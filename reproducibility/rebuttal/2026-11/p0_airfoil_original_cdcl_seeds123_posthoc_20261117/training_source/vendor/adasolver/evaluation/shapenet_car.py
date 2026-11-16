"""Manifest-bound Car evaluation with one reconstruction/metric path for all methods."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from adasolver.data.shapenet_car_runtime import CarRuntime
from adasolver.evaluation.shapenet_car_compare import summarize_records
from adasolver.inference import adaptive
from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
from adasolver.inference.cpg.shapenet_car import ShapeNetCarCPG
from adasolver.inference.cpg.shapenet_car_stratified import ShapeNetCarStratified
from adasolver.inference.cpg.shapenet_car_cd_shell import ShapeNetCarCdShell
from adasolver.inference.pi.shapenet_car import (
    sample_car_pi_exponential, sample_car_pi_two_stage, sample_car_pi_two_stage_exact,
    sample_car_pi_two_stage_tempered,
)
from adasolver.inference.shapenet_car_contract import (
    ROOT, PROTOCOL, BUDGETS, SEEDS, DEFAULT_MANIFEST, DEFAULT_SPLITS,
    coarse_count, indices_hash, json_hash, method_config, rng_for,
)
from adasolver.inference.uniform import sample_uniform
from adasolver.inference.base import SamplingResult
from adasolver.inference.native_fps import load_native_fps
from adasolver.layers import install_compensation
from adasolver.metrics.shapenet_car import reconstruct_car_fields, dual_field_metrics
from adasolver.metrics.car_cd import CarCdGeometry
from adasolver.models.checkpoint import sha256_file
from adasolver.utils import budget_count


def source_binding():
    paths = ["data/shapenet_car.py", "data/shapenet_car_runtime.py", "models/transolver3.py",
             "models/adapter.py", "models/checkpoint.py", "layers/compensation.py", "layers/block0_cache.py",
             "metrics/car_cd.py", "metrics/shapenet_car.py", "inference/adaptive.py",
             "inference/native_fps.py", "inference/_fps_native.c",
             "inference/shapenet_car_contract.py", "inference/pi/shapenet_car.py", "inference/finite_measure.py",
             "inference/cpg/shapenet_car.py", "inference/cpg/shapenet_car_stratified.py",
             "inference/cpg/shapenet_car_cd_shell.py", "inference/ca/shapenet_car.py",
             "evaluation/shapenet_car.py", "evaluation/shapenet_car_compare.py"]
    sources = {p: sha256_file(ROOT / "adasolver" / p) for p in paths}
    return sources, json_hash(sources)


def timed(call, device=None):
    cuda = device is not None and torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = call()
    if cuda:
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - start


def config_name(config):
    return f"{config['method']}_{config['selection_target']}_{json_hash(config)[:12]}"


def reweight_stratified_sampling(sampling, beta):
    """Reuse an identical stratified point set for a different compensation exponent."""
    if sampling.metadata.get("strategy") != "surface_stratified":
        raise ValueError("only unnormalized stratum count ratios can be reweighted")
    return SamplingResult(sampling.indices, sampling.sampling_density,
        np.power(1. / sampling.sampling_density, beta).astype(np.float32),
        {**sampling.metadata, "beta": beta})


@torch.no_grad()
def evaluate_configs(output, configs, *, manifest=DEFAULT_MANIFEST, data_path=None,
                     splits_path=DEFAULT_SPLITS, split="calibration", samples=None,
                     seeds=SEEDS, device="cuda:0", head_manifest=None,
                     save_prediction_ids=(), require_variable=True, sample_offset=0):
    configs = [method_config(c["method"], **{k: v for k, v in c.items() if k != "method"}) for c in configs]
    if not configs or len({config_name(c) for c in configs}) != len(configs):
        raise ValueError("configs must be nonempty and unique")
    fullmesh = any(c["method"] == "fullmesh" for c in configs)
    if fullmesh and len(configs) != 1:
        raise ValueError("fullmesh must be evaluated separately")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    runtime = CarRuntime(manifest, data_path, require_variable=require_variable)
    splits = runtime.splits(splits_path)
    if split not in {"calibration", "confirmation", "test", "head_validation"}:
        raise ValueError("unsupported evaluation split")
    ids = splits[split]
    if not 0 <= sample_offset < len(ids):
        raise ValueError("sample offset is outside the declared split")
    ids = ids[sample_offset:]
    if samples is not None:
        if samples < 1 or samples > len(ids):
            raise ValueError("samples must fit the declared split")
        ids = ids[:samples]
    operator = runtime.load_operator(device)
    context, patched = install_compensation(operator)
    ca = ShapeNetCarCA(operator, head_manifest, runtime.binding(), device) if any(c["method"] == "ca" for c in configs) else None
    if ca is not None:
        if ca.metadata["split_hash"] != splits["split_hash"]:
            raise ValueError("head dataset and evaluator use different splits")
        if samples is None and ca.metadata["status"] != "trained_head":
            raise ValueError("complete evaluations require a fully trained head")
    _, native_startup_seconds = timed(load_native_fps) if ca is not None else (None, 0.)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        item = runtime.sample(ids[0])
        for count in (coarse_count(len(item)), budget_count(len(item), .75)):
            operator(item.pos[:count][None].to(device), item.x_norm[:count][None].to(device))
        if ca is not None:
            count = coarse_count(len(item))
            ca.acquire(item.pos[:count][None].to(device), item.x_norm[:count][None].to(device), len(item))
        torch.cuda.synchronize(device)
    sources, source_hash = source_binding()
    budgets = (1.,) if fullmesh else BUDGETS
    output, runs = Path(output), []
    for config in configs:
        directory = output / config_name(config)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"run output already exists: {directory}")
        (directory / "arrays").mkdir(parents=True)
        binding = {**runtime.binding(), "protocol_version": PROTOCOL, "config": config,
                   "split_role": split, "split_hash": splits["split_hash"], "sample_ids": ids,
                   "seeds": list(seeds), "budgets": list(budgets), "source_sha256": source_hash,
                   "source_files": sources,
                   "head_sha256": ca.metadata["head_sha256"] if config["method"] == "ca" else None,
                   "reconstruction": {"backend": "knn_idw", "k": 4, "coordinates": "physical_raw", "pressure": "surface_only"}}
        binding["run_id"] = json_hash(binding)
        binding.update(status="running", patched_attention_layers=patched,
            implementation={"cache":"coarse_prefix_v2", "ca_acquisition":"cuda_graph",
                            "ca_fps":"native_float64", "native_startup_seconds":native_startup_seconds,
                            "tuning_sampling_cache":"beta_independent_indices_and_geometry_only_zero_guidance",
                            "reconstruction":"vectorized_exact_hits"},
            environment={"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "device": str(device),
                         "hardware": torch.cuda.get_device_name(device) if cuda else platform.processor()},
            timing_accounting="total_seconds sums standalone costs; cached geometry, acquisition and beta-independent sampled sets are charged at their originally measured cost for each query. wall_seconds measures actual per-record time.",
            head_manifest=str(head_manifest) if config["method"] == "ca" else None)
        (directory / "run_manifest.json").write_text(json.dumps(binding, indent=2) + "\n")
        runs.append({"directory": directory, "manifest": binding, "config": config, "rows": [],
                     "stream": (directory / "raw.jsonl").open("w")})
    start_run = time.perf_counter()
    try:
        for sample_index, sid in enumerate(ids):
            item = runtime.sample(sid)
            coords, mask, truth = item.pos.numpy(), item.surface_mask.numpy(), item.target_phys.numpy()
            geom, geom_time = timed(lambda: CarCdGeometry.load(
                runtime.data_path / "training_data" / sid / "quadpress_smpl.vtk",
                ROOT / "temp/shapenet_car/budget10_v1/geometry"))
            geom.validate_surface(coords[mask])
            gt_cd = geom.compute(truth[mask, 3], truth[mask, :3])
            gt_cd_pressure = float(geom.pressure_coefficients @ truth[mask, 3])
            gt_cd_shear = float(geom.velocity_z_coefficients @ truth[mask, 2])
            patches = {}
            for config in configs:
                if (config["method"] in {"cpg", "ca"} and config.get("strategy") not in {"surface_stratified", "cd_shell"}
                        and config["patch_count"] not in patches):
                    k = config["patch_count"]
                    patches[k] = timed(lambda: adaptive.build_patches(coords, k, seed=0))
            for seed in seeds:
                # Beta does not change these samplers' points or stratum masses.
                # Reuse across tuning candidates; still charge the original
                # measured standalone sampling cost to every result row.
                stratified_cache = {}
                coarse = rng_for(sid, seed).choice(len(item), coarse_count(len(item)), replace=False)
                cp, cx = item.pos[coarse][None].to(device), item.x_norm[coarse][None].to(device)
                pred0, coarse_seconds = None, 0.
                if any(c["method"] == "cpg" and c.get("strategy") != "cd_shell" for c in configs):
                    context.clear()
                    pred0, coarse_seconds = timed(lambda: operator(cp, cx), device)
                    pred0 = runtime.output_normalizer.decode(pred0[0].cpu()).numpy()
                head_pressure, cache, ca_seconds = None, None, 0.
                if any(c["method"] == "ca" and c.get("strategy") != "cd_shell" for c in configs):
                    context.clear()
                    (head_pressure, cache), ca_seconds = timed(lambda: ca.acquire(cp, cx, len(item)), device)
                    head_pressure = head_pressure.cpu().numpy()
                uniform_coarse, uniform_cache = coarse, cache
                uniform_coarse_seconds, uniform_ca_seconds = coarse_seconds, ca_seconds
                budget_acquisition = {}
                if any(c.get("strategy") == "cd_shell" for c in configs):
                    for budget in budgets:
                        subset = rng_for(sid, seed, "refinement", budget).permutation(len(item))[:coarse_count(len(item))]
                        bp, bx = item.pos[subset][None].to(device), item.x_norm[subset][None].to(device)
                        acquisition = {"coarse": subset, "coarse_seconds": 0., "ca_seconds": 0., "cache": None}
                        if any(c["method"] == "cpg" and c.get("strategy") == "cd_shell" for c in configs):
                            context.clear()
                            fields, acquisition["coarse_seconds"] = timed(lambda: operator(bp, bx), device)
                            acquisition["fields"] = runtime.output_normalizer.decode(fields[0].cpu()).numpy()
                        if any(c["method"] == "ca" and c.get("strategy") == "cd_shell" for c in configs):
                            context.clear()
                            (pressure, acache), acquisition["ca_seconds"] = timed(lambda: ca.acquire(bp, bx, len(item)), device)
                            acquisition.update(pressure=pressure.cpu().numpy(), cache=acache)
                        budget_acquisition[budget] = acquisition
                for run in runs:
                    cfg, method = run["config"], run["config"]["method"]
                    coarse, cache = uniform_coarse, uniform_cache
                    coarse_seconds, ca_seconds = uniform_coarse_seconds, uniform_ca_seconds
                    sampler, score, score_seconds, patch_seconds = None, None, 0., 0.
                    cd_shell = cfg.get("strategy") == "cd_shell"
                    if method in {"cpg", "ca"}:
                        stratified = cfg.get("strategy") == "surface_stratified"
                        patch_data, patch_seconds = ((np.zeros(len(item), dtype=int), np.ones(1)), 0.) if stratified or cd_shell else patches[cfg["patch_count"]]
                        score_options = {k: cfg[k] for k in ("coarse_fraction", "patch_count", "neighborhood", "feature",
                            "normalization", "score_floor", "score_power", "kappa", "beta", "signal", "surface_band")}
                        sampler = ShapeNetCarCPG(coords, mask, patches=patch_data,
                            fps_distance_backend="native" if method == "ca" else "numpy",
                            placement="fps", **score_options)
                        influence = geom.pressure_influence if cfg["cd_weight"] else None
                        if cd_shell:
                            scorer = sampler
                            sampler, patch_seconds = timed(lambda: ShapeNetCarCdShell(coords, mask,
                                **{k: cfg[k] for k in ("shell_radius", "frac_extra", "physics_strength", "beta", "compensation")}))
                        else:
                            values = head_pressure if method == "ca" else (
                                pred0[:, 3] if cfg["signal"] == "pressure" else np.linalg.norm(pred0[:, :3], axis=1))
                            score, score_seconds = timed(lambda: sampler._score(coarse, values, influence=influence))
                        if stratified:
                            sampler, patch_seconds = timed(lambda: ShapeNetCarStratified(coords, mask,
                                **{k: cfg[k] for k in ("wall_scale", "wall_amplitude", "physics_strength",
                                                      "beta", "placement", "score_bins")}))
                    for budget in budgets:
                        record_start = time.perf_counter()
                        keep = budget_count(len(item), budget)
                        rng = rng_for(sid, seed, "refinement", budget)
                        if method == "fullmesh":
                            sampling = SamplingResult(np.arange(len(item)), np.ones(len(item))/len(item),
                                                      np.ones(len(item), dtype=np.float32), {"method": "fullmesh"})
                            refinement_seconds = 0.
                        elif method == "uniform":
                            sampling, refinement_seconds = timed(lambda: sample_uniform(len(item), keep, rng))
                        elif method == "pi":
                            if cfg["strategy"] == "two_stage":
                                sampling, refinement_seconds = timed(lambda: sample_car_pi_two_stage(
                                    coords, mask, keep, rng, r=cfg["r"], frac_extra=cfg["frac_extra"], beta=cfg["beta"]))
                            elif cfg["strategy"] == "two_stage_exact":
                                sampling, refinement_seconds = timed(lambda: sample_car_pi_two_stage_exact(
                                    coords, mask, keep, rng, r=cfg["r"], frac_extra=cfg["frac_extra"], beta=cfg["beta"]))
                            elif cfg["strategy"] == "two_stage_tempered":
                                sampling, refinement_seconds = timed(lambda: sample_car_pi_two_stage_tempered(
                                    coords, mask, keep, rng, r=cfg["r"], frac_extra=cfg["frac_extra"],
                                    alpha=cfg["proposal_alpha"], beta=cfg["beta"]))
                            elif cfg["strategy"] == "exponential":
                                sampling, refinement_seconds = timed(lambda: sample_car_pi_exponential(
                                    coords, mask, keep, rng, amp=cfg["amp"], ell=cfg["ell"], beta=cfg["beta"]))
                            else:
                                raise ValueError("unsupported PI strategy")
                        elif cd_shell:
                            acquisition = budget_acquisition[budget]
                            coarse, cache = acquisition["coarse"], acquisition["cache"]
                            coarse_seconds, ca_seconds = acquisition["coarse_seconds"], acquisition["ca_seconds"]
                            base, prepare_seconds = timed(lambda: sampler.prepare(keep, rng))
                            np.testing.assert_array_equal(base[:len(coarse)], coarse)
                            values = acquisition["pressure"] if method == "ca" else acquisition["fields"][:, 3]
                            score, score_seconds = timed(lambda: scorer._score(coarse, values, influence=influence))
                            sampling, refinement_seconds = timed(lambda: sampler.refine_from_base(base, score, keep, rng))
                            refinement_seconds += prepare_seconds
                        elif cfg.get("strategy") == "surface_stratified":
                            if cfg["physics_strength"] == 0:
                                # With guidance off, the allocator ignores all
                                # scores and both methods have the same S0.
                                key_config = {k: cfg[k] for k in ("strategy", "coarse_fraction", "wall_scale",
                                    "wall_amplitude", "placement", "physics_strength")}
                            else:
                                key_config = {k: v for k, v in cfg.items() if k not in {"beta", "selection_target"}}
                            sample_key = (json_hash(key_config), budget)
                            if sample_key in stratified_cache:
                                reference_sampling, refinement_seconds = stratified_cache[sample_key]
                                sampling = reweight_stratified_sampling(reference_sampling, cfg["beta"])
                            else:
                                sampling, refinement_seconds = timed(lambda: sampler.refine_score(coarse, score, keep, rng))
                                stratified_cache[sample_key] = (sampling, refinement_seconds)
                        else:
                            sampling, refinement_seconds = timed(lambda: sampler.refine_score(coarse, score, keep, rng))
                        sampling.validate(len(item), keep)
                        indices = sampling.indices
                        pos, fx = item.pos[indices][None].to(device), item.x_norm[indices][None].to(device)
                        weights = None if method in {"uniform", "fullmesh"} else torch.as_tensor(sampling.compensation, device=device)
                        if cuda:
                            torch.cuda.reset_peak_memory_stats(device)
                        context.set(weights)
                        try:
                            prediction, infer_seconds = timed(
                                lambda: ca.final(pos, fx, coarse, indices, cache, weights) if method == "ca" else operator(pos, fx), device)
                        finally:
                            context.clear()
                        prediction = prediction[0].cpu().numpy()
                        (velocity, pressure, si), recon_seconds = timed(lambda: reconstruct_car_fields(
                            indices, prediction, coords, mask, prediction_is_normalized=True,
                            output_mean=runtime.output_normalizer.mean.numpy(),
                            output_std=runtime.output_normalizer.std.numpy(), k=4))
                        reconstructed = np.zeros((len(item), 4))
                        reconstructed[:, :3], reconstructed[si, 3] = velocity, pressure
                        metrics = dual_field_metrics(reconstructed, truth, mask)
                        cd, cd_seconds = timed(lambda: geom.compute(pressure, velocity[si]))
                        array_name = f"{sid.replace('/', '_')}_s{seed}_b{budget:.3f}.npz"
                        arrays = {"indices": indices, "compensation": sampling.compensation,
                                  "sampling_density": sampling.sampling_density}
                        if sampler is not None:
                            arrays.update(coarse_indices=coarse, score=score)
                        if sid in save_prediction_ids:
                            arrays.update(prediction=reconstructed.astype(np.float32), target=truth, pos=coords, surface_mask=mask)
                        np.savez_compressed(run["directory"] / "arrays" / array_name, **arrays)
                        times = {"geometry_seconds": geom_time + patch_seconds,
                                 "coarse_seconds": coarse_seconds if method == "cpg" else 0.,
                                 "feature_head_seconds": ca_seconds if method == "ca" else 0.,
                                 "score_seconds": score_seconds, "refinement_seconds": refinement_seconds,
                                 "final_inference_seconds": infer_seconds, "reconstruction_seconds": recon_seconds, "cd_seconds": cd_seconds}
                        row = {k: run["manifest"][k] for k in ("run_id", "operator_sha256", "normalizer_sha256", "protocol_version", "split_hash")}
                        row.update(benchmark="shapenet_car", method=method, selection_target=cfg["selection_target"],
                            sample_id=sid, seed=seed, budget_fraction=budget, budget_count=keep,
                            full_point_count=len(item), selected_count=len(indices), coarse_count=len(coarse) if sampler is not None else 0,
                            surface_count=int(mask.sum()), surface_selected_count=int(mask[indices].sum()),
                            selected_indices_sha256=indices_hash(indices),
                            coarse_indices_sha256=indices_hash(coarse) if sampler is not None else None,
                            arrays_path=f"arrays/{array_name}", velocity_rel_l2=metrics.velocity_rel_l2,
                            pressure_rel_l2=metrics.pressure_rel_l2, field_error=metrics.field_error,
                            cd_pred=cd, cd_gt=gt_cd, cd_relative_error=abs(cd-gt_cd)/max(abs(gt_cd), 1e-12),
                            cd_pressure_pred=float(geom.pressure_coefficients @ pressure),
                            cd_shear_pred=float(geom.velocity_z_coefficients @ velocity[si, 2]),
                            cd_pressure_gt=gt_cd_pressure, cd_shear_gt=gt_cd_shear,
                            cd_status="ok", status="ok", sampling_metadata=sampling.metadata,
                            interpolation_fallback_count=0, reconstruction_k=4,
                            reconstruction_surface_effective_k=min(4, int(mask[indices].sum())), mesh_sha256=geom.mesh_sha256,
                            total_seconds=sum(times.values()), wall_seconds=time.perf_counter()-record_start,
                            peak_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0, **times)
                        run["rows"].append(row)
                        run["stream"].write(json.dumps(row, allow_nan=False) + "\n")
                        run["stream"].flush()
            print(f"{split}: {sample_index+1}/{len(ids)} {sid} ({time.perf_counter()-start_run:.1f}s)", flush=True)
        results = []
        for run in runs:
            summary = summarize_records(run["rows"], budgets=budgets)
            run["manifest"].update(status="complete", records=len(run["rows"]))
            (run["directory"] / "run_manifest.json").write_text(json.dumps(run["manifest"], indent=2) + "\n")
            (run["directory"] / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            results.append({"directory": str(run["directory"]), "config": run["config"], "summary": summary})
        return results
    except Exception as error:
        for run in runs:
            run["manifest"].update(status="failed", error=f"{type(error).__name__}: {error}")
            (run["directory"] / "run_manifest.json").write_text(json.dumps(run["manifest"], indent=2) + "\n")
        raise
    finally:
        for run in runs:
            run["stream"].close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--data-path")
    p.add_argument("--splits", default=str(DEFAULT_SPLITS))
    p.add_argument("--split", choices=("calibration", "confirmation", "test", "head_validation"), default="calibration")
    p.add_argument("--samples", type=int)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--method", choices=("uniform", "pi", "cpg", "ca", "fullmesh"))
    p.add_argument("--configs", type=Path, help="JSON/YAML config or list of configs")
    p.add_argument("--head-manifest")
    p.add_argument("--save-prediction-ids", nargs="*", default=[])
    args = p.parse_args()
    if bool(args.method) == bool(args.configs):
        p.error("provide exactly one of --method or --configs")
    configs = yaml.safe_load(args.configs.read_text()) if args.configs else [method_config(args.method)]
    if isinstance(configs, dict):
        configs = [configs]
    result = evaluate_configs(args.output, configs, manifest=args.manifest, data_path=args.data_path,
        splits_path=args.splits, split=args.split, samples=args.samples, seeds=args.seeds,
        device=args.device, head_manifest=args.head_manifest, save_prediction_ids=args.save_prediction_ids,
        sample_offset=args.sample_offset)
    print(json.dumps([{k: r[k] for k in ("directory", "config")} for r in result], indent=2))


if __name__ == "__main__":
    main()
