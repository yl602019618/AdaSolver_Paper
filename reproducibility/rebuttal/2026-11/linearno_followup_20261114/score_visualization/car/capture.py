#!/usr/bin/env python3
"""Trace native Car acquisition on CPU, preserving historical sampled arrays.

No training, target-field score, GPU request, or new sampling experiment occurs.
The archived final scores, selected indices and compensation remain authoritative.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import datetime
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    e5 = root / "rebuttal_20261112/experiments/E5_training_seeds"
    vendor = e5 / "code/vendor"
    sys.path.insert(0, str(vendor))
    import torch
    from scipy.spatial import cKDTree
    from adasolver.data.shapenet_car_runtime import CarRuntime
    from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
    from adasolver.inference.cpg.shapenet_car import surface_scalar_score, normalize_cpg_score
    from adasolver.inference.cpg.shapenet_car_stratified import allocate_counts
    from adasolver.inference.pi.shapenet_car import car_surface_distance, sample_car_pi_two_stage
    from adasolver.inference.shapenet_car_contract import rng_for
    torch.set_num_threads(4)
    sid, budget, seed = "param0/100715345ee54d7ae38b52b4ee9d36a3", .5, 0
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data"
    data.mkdir(exist_ok=True)
    started = time.perf_counter()
    model_manifest = e5 / "checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json"
    head_manifest = e5 / "checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json"
    runtime = CarRuntime(model_manifest)
    sample = runtime.sample(sid)
    xyz, surface = sample.pos.numpy(), sample.surface_mask.numpy()
    n, count = len(xyz), round(len(xyz)*budget)
    distance = car_surface_distance(xyz, surface)
    operator = runtime.load_operator("cpu")
    ca = ShapeNetCarCA(operator, head_manifest, runtime.binding(), "cpu", use_cuda_graph=False)
    source_hashes = {}
    for relative in ["data/shapenet_car_runtime.py", "data/shapenet_car.py", "models/adapter.py", "models/transolver3.py",
                     "inference/pi/shapenet_car.py", "inference/cpg/shapenet_car.py", "inference/cpg/shapenet_car_stratified.py",
                     "inference/ca/shapenet_car.py", "inference/shapenet_car_contract.py", "layers/block0_cache.py"]:
        path = vendor / "adasolver" / relative
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    configurations = {"uniform": "uniform_field_26ccb3fbd7ed", "pi": "pi_field_0a7a08b11413",
                      "cpg": "cpg_field_6a0eb5f218c2", "ca": "ca_field_99d5b859e461"}
    methods, validation = {}, []

    def array_hash(a):
        return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

    for method, run in configurations.items():
        start = time.perf_counter()
        directory = e5 / "records/seed0/adaptive" / run
        manifest = json.loads((directory / "run_manifest.json").read_text())
        cfg = manifest["config"]
        record = next(r for r in map(json.loads, (directory/"raw.jsonl").read_text().splitlines())
                      if r["sample_id"] == sid and r["seed"] == seed and r["budget_fraction"] == budget)
        path = directory / record["arrays_path"]
        with np.load(path) as z:
            saved = {key: np.array(z[key], copy=True) for key in z.files}
        selected_indices = saved["indices"]
        selected = np.zeros(n, dtype=bool)
        selected[selected_indices] = True
        coarse = np.zeros(n, dtype=bool)
        if "coarse_indices" in saved:
            coarse[saved["coarse_indices"]] = True
        assert array_hash(selected_indices) == record["selected_indices_sha256"]
        assert len(selected_indices) == count and len(np.unique(selected_indices)) == count
        arrays = {"index": np.arange(n), "xyz": xyz, "surface": surface, "selected": selected, "coarse": coarse,
                  "actual_selected_indices": selected_indices, "distance_to_surface": distance,
                  "native_compensation": saved["compensation"], "native_sampling_density": saved["sampling_density"]}
        blank = lambda: np.full(n, np.nan)
        raw_prior, raw_score, normalized_native, normalized_cpu, allocation = blank(), blank(), blank(), blank(), blank()
        native_density, compensation = blank(), blank()
        native_density[selected_indices] = saved["sampling_density"]
        compensation[selected_indices] = saved["compensation"]
        definition = {"native_compensation_density": "recorded sampling_density on selected indices only; no values imputed outside its support",
                      "selected": "exact archived ordered selected indices; no rerun or filtering",
                      "normalized_score_native": "not applicable"}
        identity_check = {"method": method, "selected_hash_match": True,
                          "native_density_hash": array_hash(saved["sampling_density"]),
                          "native_compensation_hash": array_hash(saved["compensation"])}
        if method == "uniform":
            raw_prior[:] = 1.
            allocation[:] = count/n
            definition.update(raw_prior="constant equal-weight baseline; no model score evaluated",
                              allocation_density="uniform inclusion probability K/N; not native1/N sampling_density",
                              normalized_score_native="N/A: uniform sampler does not compute a normalized score")
        elif method == "pi":
            # Replay the real hard-shell sampler only to verify its stored arrays.
            replay = sample_car_pi_two_stage(xyz, surface, count, rng_for(sid, seed, "refinement", budget),
                                            r=cfg["r"], frac_extra=cfg["frac_extra"], beta=cfg["beta"])
            assert np.array_equal(replay.indices, selected_indices)
            assert np.array_equal(replay.sampling_density, saved["sampling_density"])
            assert np.array_equal(replay.compensation, saved["compensation"])
            raw_prior[:] = distance < cfg["r"]
            nb = record["sampling_metadata"]["base_count"]
            base_mask = np.zeros(n, dtype=bool)
            base_mask[selected_indices[:nb]] = True
            stage2 = selected & ~base_mask
            arrays.update(pi_uniform_base=base_mask, pi_second_stage_selected=stage2)
            identity_check.update(pi_native_replay_equal=True)
            definition.update(raw_prior=f"hard-shell membership: distance_to_surface < {cfg['r']}; true native PI prior",
                              normalized_score_native="N/A: historical two-stage PI computes no global normalized score",
                              allocation_density="N/A: native two-stage code defines no full-cloud density; selected compensation density remains separate",
                              native_compensation_density="selected shell-mass weights divided by sum of selected weights; fallback points can receive shell weight; not full-cloud proposal/inclusion probability")
        else:
            ci = saved["coarse_indices"]
            assert array_hash(ci) == record["coarse_indices_sha256"]
            cp, cx = sample.pos[ci][None], sample.x_norm[ci][None]
            with torch.no_grad():
                if method == "cpg":
                    prediction_norm = operator(cp, cx)[0]
                    prediction = runtime.output_normalizer.decode(prediction_norm).numpy()
                    values = prediction[:, 3]
                    arrays.update(coarse_prediction_physical_cpu=prediction,
                                  coarse_prediction_normalized_cpu=prediction_norm.numpy())
                else:
                    features, _ = ca.feature_pass(cp, cx, n, capture=False)
                    head_raw = ca.head(features)[0, :, 0]
                    pressure = head_raw * ca.metadata["target_std"] + ca.metadata["target_mean"]
                    values = pressure.numpy()
                    arrays.update(ca_block0_features_cpu=features[0].numpy(), ca_head_output_raw_cpu=head_raw.numpy(),
                                  ca_head_pressure_physical_cpu=values)
            raw_score = surface_scalar_score(xyz, surface, ci, values, feature=cfg["feature"],
                                             k=cfg["neighborhood"], surface_band=cfg["surface_band"])
            normalized_cpu = normalize_cpg_score(raw_score, method=cfg["normalization"],
                                                 floor=cfg["score_floor"], power=cfg["score_power"])
            normalized_native = saved["score"]
            delta = np.abs(normalized_cpu-normalized_native)
            tolerance = 1e-6 + 1e-5*np.abs(normalized_native)
            identity_check.update(cpu_score_replay_max_abs=float(delta.max()),
                                  cpu_score_replay_relative_norm=float(np.linalg.norm(delta)/np.linalg.norm(normalized_native)),
                                  cpu_score_replay_violations=int(np.sum(delta > tolerance)),
                                  cpu_score_replay_atol=1e-6, cpu_score_replay_rtol=1e-5,
                                  cpu_score_replay_pass=bool(np.all(delta <= tolerance)))
            # Recover exactly the allocation defined by the archived score. The
            # actual FPS points are reused, not recomputed from the CPU score.
            anchors = coarse | surface
            strata = np.zeros(n, dtype=np.int64)
            shell = np.searchsorted(cfg["wall_scale"] * np.array([.25, .5, 1., 2., 4., 8., 16.]), distance)
            next_id = 1
            for band in np.unique(shell[~surface]):
                ids = np.flatnonzero((shell == band) & ~surface)
                cuts = np.unique(np.quantile(normalized_native[ids], np.arange(1, cfg["score_bins"])/cfg["score_bins"]))
                groups = np.searchsorted(cuts, normalized_native[ids], side="left")
                for rank in np.unique(groups):
                    strata[ids[groups == rank]] = next_id
                    next_id += 1
            mass = np.bincount(strata, minlength=next_id)
            lower = np.bincount(strata[anchors], minlength=next_id)
            scale = max(float(np.quantile(normalized_native[surface], .95)), 1e-12)
            physics = np.clip(normalized_native/scale, 0., 1.)
            allocation_priority = (1.+cfg["wall_amplitude"]*np.exp(-distance/cfg["wall_scale"])) * (1.+cfg["physics_strength"]*physics)
            priority = np.bincount(strata, weights=allocation_priority, minlength=next_id)
            counts = allocate_counts(mass, lower, priority, count)
            allocation = counts[strata]/mass[strata]
            assert np.array_equal(counts, np.bincount(strata[selected_indices], minlength=next_id))
            assert np.array_equal(allocation[selected_indices], saved["sampling_density"])
            assert np.array_equal(counts, record["sampling_metadata"]["stratum_selected_counts"])
            assert np.array_equal(mass, record["sampling_metadata"]["stratum_reference_counts"])
            arrays.update(coarse_indices=ci, coarse_input_pressure_cpu=values, stratum=strata,
                          stratum_mass=mass, stratum_selected_counts=counts, allocation_priority=allocation_priority,
                          physics_term=physics)
            identity_check.update(stratum_counts_match_archived=True, full_allocation_matches_selected_density=True)
            definition.update(raw_score_cpu="CPU replay: mean squared coarse-surface pressure difference divided by neighbor distance squared (k4); nearest lift to surface, then exp(-surface_distance/0.1)",
                              normalized_score_native="archived GPU score: max(raw/q95(raw),0.1)^1 on full native cloud; copied without alteration",
                              normalized_score_cpu="separate CPU replay with frozen checkpoint; parity result reported; not substituted for archived score",
                              allocation_density="native stratum quota/reference count, extended to each original point in that stratum; sum is K, not1; FPS plus anchors makes it an allocation ratio, not point inclusion probability",
                              allocation_priority="(1+64*exp(-distance/0.1))*(1+2*clip(saved_score/q95(surface_saved_score),0,1))")
        # Diagnostic realized retention at the same64-neighbor geometry support.
        neighbor = cKDTree(xyz).query(xyz, k=64)[1]
        realized = selected[neighbor].mean(axis=1)
        arrays.update(raw_prior=raw_prior, raw_score_cpu=raw_score,
                      normalized_score_native=normalized_native, normalized_score_cpu=normalized_cpu,
                      allocation_density=allocation, compensation_density_selected=native_density,
                      compensation_weight_selected=compensation, realized_local_retained_fraction=realized)
        npz_path = data / f"{method}.npz"
        np.savez_compressed(npz_path, **arrays)
        csv_path = data / f"{method}_perpoint.csv"
        fields = ["index", "x", "y", "z", "surface", "coarse", "selected", "raw_prior", "raw_score_cpu",
                  "normalized_score_native", "normalized_score_cpu", "allocation_density", "compensation_density_selected",
                  "compensation_weight_selected", "realized_local_retained_fraction"]
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for i in range(n):
                row = [i, *xyz[i], int(surface[i]), int(coarse[i]), int(selected[i]), raw_prior[i], raw_score[i],
                       normalized_native[i], normalized_cpu[i], allocation[i], native_density[i], compensation[i], realized[i]]
                writer.writerow(["" if isinstance(value, (float, np.floating)) and np.isnan(value) else value for value in row])
        elapsed = time.perf_counter()-start
        identity_check["cpu_trace_seconds"] = elapsed
        validation.append(identity_check)
        methods[method] = {"perpoint_csv": str(csv_path.relative_to(output)), "npz": str(npz_path.relative_to(output)),
            "columns": {"index": "index", "xyz": ["x", "y", "z"], "selected": "selected", "coarse": "coarse",
                        "rawscore": "raw_score_cpu" if method in ["cpg", "ca"] else "raw_prior",
                        "normalizedscore": "normalized_score_native" if method in ["cpg", "ca"] else None,
                        "density": "allocation_density" if method != "pi" else None,
                        "native_selected_density": "compensation_density_selected"},
            "definitions": definition, "configuration": cfg, "source_npz": str(path.relative_to(root)),
            "source_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_manifest": str((directory/"run_manifest.json").relative_to(root)),
            "source_manifest_sha256": hashlib.sha256((directory/"run_manifest.json").read_bytes()).hexdigest(),
            "actual_selected_ordered_sha256": record["selected_indices_sha256"],
            "actual_coarse_ordered_sha256": record.get("coarse_indices_sha256"),
            "selected_count": count, "coarse_count": int(coarse.sum()), "surface_selected_count": int((surface&selected).sum()),
            "native_online_generation_seconds": {k:record[k] for k in ["coarse_seconds", "feature_head_seconds", "score_seconds", "refinement_seconds"]},
            "cpu_trace_seconds": elapsed, "missing": ["native normalized score", "native full-cloud sampling distribution"] if method == "pi" else []}
        methods[method]["field_semantics"] = {
            "raw_prior": {"support":"full" if method in ["uniform","pi"] else "not_applicable", "semantic_type":"prior", "normalization_rule":"none"},
            "raw_score_cpu": {"support":"full" if method in ["cpg","ca"] else "not_applicable", "semantic_type":"score", "normalization_rule":"none", "execution":"separate CPU trace"},
            "normalized_score_native": {"support":"full" if method in ["cpg","ca"] else "not_applicable", "semantic_type":"score", "normalization_rule":"max(raw/q95(raw),0.1)^1; not sum1 or mean1", "execution":"archived GPU output"},
            "normalized_score_cpu": {"support":"full" if method in ["cpg","ca"] else "not_applicable", "semantic_type":"score", "normalization_rule":"same native q95 rule; separately recomputed on CPU"},
            "allocation_density": {"support":"not_applicable" if method=="pi" else "full", "semantic_type":"selection_rate" if method=="uniform" else "stratum_allocation_ratio", "normalization_rule":"none; sum over full reference cloud=K" if method!="pi" else "not_defined_by_native_algorithm"},
            "compensation_density_selected": {"support":"selected", "semantic_type":"native_selected_density", "normalization_rule":"selected weights/sum(selected weights)" if method=="pi" else "constant1/N" if method=="uniform" else "stratum_selected_count/stratum_reference_count; none"},
            "compensation_weight_selected": {"support":"selected", "semantic_type":"compensation", "normalization_rule":"constant1" if method=="uniform" else "native shell weight^(-beta), with fallback assignments" if method=="pi" else "allocation_density^(-beta)"},
            "realized_local_retained_fraction": {"support":"full", "semantic_type":"realized_selection_rate_diagnostic", "normalization_rule":"selected_count/64 on same nearest64 native neighbors; never used for acquisition"}}
        print(method, identity_check, flush=True)
    report = {"status": "PARTIAL" if any(r.get("cpu_score_replay_pass") is False for r in validation) else "VERIFIED_SCOPED_CAPTURE",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "case_id": sid, "dataset": "ShapeNet-Car",
        "backbone": "Transolver3", "training_seed": 0, "head_seed": 0, "sampling_seed": 0, "budget": budget,
        "points": n, "surface_points": int(surface.sum()), "checkpoint_sha256": runtime.metadata["checkpoint_sha256"],
        "head_sha256": ca.metadata["head_sha256"], "normalizer_sha256": runtime.metadata["normalizer_sha256"],
        "split_hash": next(iter(json.loads((e5/'config/splits_seed0.json').read_text()).get(k, '') for k in ['split_hash'])),
        "coordinate_space": "native physical xyz; no jitter, smoothing or coordinate remeshing",
        "truth_loaded_by_dataset_loader": True, "truth_read_by_capture_or_score": False,
        "truth_boundary": "CarRuntime.sample loads the target as part of its dataset object. This script never accesses target_phys or target_norm; score functions receive coordinates, surface mask and predictions/own-head outputs only.",
        "methods": methods, "checks": validation, "source_functions_sha256": source_hashes,
        "total_cpu_trace_seconds": time.perf_counter()-started, "GPU_execution": False,
        "density_caveat": "Allocation ratios, uniform inclusion probability, native selected compensation density, and realized local retention are distinct quantities; none is relabeled as a common normalized probability.",
        "paper_formula_status": "PARTIAL",
        "paper_formula_reason": "No claim is made that native selected compensation density or native stratum allocation equals the paper full-domain lambda=t/mean_rho(t). Full-reference mean1 target-density mapping is not implemented by these archived native paths, and none is invented here.",
        "files": {str(p.relative_to(output)): {"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                    "allow_public":False,"reason":"native point coordinates, intermediate features and per-point internal data; publish figures and approved reduced sources separately"}
                  for p in sorted(data.iterdir())}}
    (output/'score_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OUTPUT', output/'score_manifest.json', 'STATUS', report['status'], flush=True)


if __name__ == "__main__":
    main()
