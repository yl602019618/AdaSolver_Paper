from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.inference.airfoil2d_contract import BUDGETS, coarse_count, sample_rng, validate_coarse_indices
from adasolver.inference.airfoil2d_config import METHODS, resolve_config, at_budget
from adasolver.inference.acquisition import grad_score
from adasolver.inference.ca.airfoil2d import Airfoil2dCA, Airfoil2dPressureHead, load_airfoil2d_ca_head
from adasolver.inference.cpg.airfoil2d import sample_airfoil_cpg, sample_airfoil_cpg_force_ratio, sample_airfoil_cpg_lift_support
from adasolver.inference.cpg.airfoil2d_features import pressure_acquisition
from adasolver.inference.cpg.airfoil2d_single_feature import sample_airfoil_cpg_single_feature
from adasolver.inference.pi.airfoil2d import sample_airfoil_pi, sample_airfoil_pi_voronoi
from adasolver.inference.uniform.airfoil2d import sample_airfoil_uniform
from adasolver.layers import install_compensation
from adasolver.layers.block0_cache import cached_operator_forward
from adasolver.metrics import pressure_surface_force, relative_area_under_error_curve, relative_l2
from adasolver.models.checkpoint import load_airfoil2d_operator, sha256_file
from adasolver.utils.budget import budget_count
from adasolver.utils.interpolation import index_grid, reconstruct_field


def synchronized_time(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def force_record(force):
    ratio = force.drag_coefficient / force.lift_coefficient if force.lift_coefficient != 0 else None
    if ratio is not None and not np.isfinite(ratio):
        ratio = None
    return dict(Dp=force.drag_pressure, Lp=force.lift_pressure, Cd=force.drag_coefficient,
                Cl=force.lift_coefficient, Cd_over_Cl=ratio)


class AirfoilEvaluator:
    """The same sample/config/metric implementation for calibration and test."""

    def __init__(self, checkpoint, manifest, data_path, device="cuda:0"):
        self.device = device
        self.model = load_airfoil2d_operator(checkpoint, manifest, device=device)
        self.model.requires_grad_(False)
        self.context, self.patched = install_compensation(self.model)
        self.data = AirfoilRuntimeData(data_path, manifest)
        self.operator_hash = sha256_file(checkpoint)
        self.checkpoint, self.manifest = str(checkpoint), str(manifest)
        self.reference = index_grid((221, 51))
        self.heads = {}
        self.coarse_cache = {}
        self.patch_cache = {}
        self.feature_cache = {}
        self.last_visualization = None

    def selected_ids(self, ids, protocol="positive_lift", min_true_cl=.01):
        ids = [int(i) for i in ids]
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(self.data.x):
            raise ValueError("sample IDs must be nonempty, unique, and in the dataset")
        if protocol not in ("positive_lift", "all"):
            raise ValueError("unknown Airfoil test protocol")
        if not np.isfinite(min_true_cl):
            raise ValueError("minimum reference lift must be finite")
        return [i for i in ids if protocol == "all" or
                pressure_surface_force(self.data.coords(i), self.data.target(i)).lift_coefficient > min_true_cl]

    def head(self, config):
        path, manifest = config.get("head_checkpoint"), config.get("head_manifest")
        if not path or not manifest:
            raise ValueError("CA config requires head_checkpoint and head_manifest")
        key = (path, manifest)
        if key not in self.heads:
            head = load_airfoil2d_ca_head(self.model, path, manifest, self.device, self.checkpoint)
            metadata = json.loads(Path(manifest).read_text())
            if isinstance(head, Airfoil2dPressureHead):
                if metadata.get("normalizer_sha256") != self.data.normalizer_sha256:
                    raise ValueError("pressure head/operator normalizers differ")
            self.heads[key] = (head, sha256_file(path), metadata)
        return self.heads[key]

    @torch.no_grad()
    def sampling(self, sample_id, seed, budget, config):
        method = config["method"]
        cfg = at_budget(config, budget)
        coords, mandatory = self.data.coords(sample_id), self.data.mandatory
        count = budget_count(len(coords), budget)
        rng = sample_rng(sample_id, seed)
        metadata = dict(coarse_count=0, coarse_cap_count=1127, coarse_forward_calls=0,
                        acquisition_input_point_count=0, acquisition_seconds=0.,
                        coarse_sampling_seconds=0., coarse_seconds=0., head_seconds=0.,
                        feature_seconds=0., operator_stage1_cache=False)
        started = time.perf_counter()
        if method == "uniform":
            result = sample_airfoil_uniform(len(coords), count, mandatory, rng)
            cache, coarse = None, None
        elif method in ("pi", "pi_voronoi"):
            sampler = sample_airfoil_pi if method == "pi" else sample_airfoil_pi_voronoi
            result = sampler(coords, count, rng, amp=cfg["amp"], ell=cfg["ell"], beta=cfg["beta"], mandatory_indices=mandatory)
            cache, coarse = None, None
        else:
            n0 = coarse_count(len(coords), cfg["coarse_fraction"], len(mandatory))
            coarse = sample_airfoil_uniform(len(coords), n0, mandatory, rng).indices
            validate_coarse_indices(coarse, len(coords), mandatory)
            coarse_sampling_seconds = time.perf_counter() - started
            head_hash, head, head_metadata = None, None, {}
            if method == "ca":
                head, head_hash, head_metadata = self.head(cfg)
            cache_key = (sample_id, seed, n0, head_hash, cfg["operator_stage1_cache"])
            if cache_key not in self.coarse_cache:
                pos = torch.as_tensor(coords[coarse], device=self.device)[None]
                self.context.clear()
                t = synchronized_time(self.device)
                if head is None:
                    values = self.data.decode(self.model(pos, pos))[0].cpu().numpy()
                    cache = None
                else:
                    prediction, cache = head.forward_with_cache(pos, pos, capture=cfg["operator_stage1_cache"])
                    standardized = prediction[0].cpu().numpy()
                    values = head.decode_pressure(standardized) if isinstance(head, Airfoil2dPressureHead) else head.decode_score(standardized).reshape(-1)
                seconds = synchronized_time(self.device) - t
                self.coarse_cache[cache_key] = (values, cache, seconds, coarse_sampling_seconds)
            values, cache, seconds, coarse_sampling_seconds = self.coarse_cache[cache_key]
            metadata.update(coarse_count=n0, coarse_requested_fraction=cfg["coarse_fraction"],
                            coarse_realized_fraction=n0 / len(coords), coarse_forward_calls=1,
                            acquisition_input_point_count=n0, coarse_sampling_seconds=coarse_sampling_seconds,
                            coarse_seconds=seconds if head is None else 0., head_seconds=seconds if head else 0.,
                            head_sha256=head_hash, feature_source="frozen_block0_on_S0" if head else "coarse_operator_pressure",
                            feature_dim=head_metadata.get("feature_dim", 128 if head else 1),
                            acquisition_output_count=1, operator_stage1_cache=cfg["operator_stage1_cache"])
            t = time.perf_counter()
            if method in ("cpg_force_ratio", "cpg_force_ratio_lift_support"):
                common = {key: cfg[key] for key in ("ell", "geometry_weight", "uniform_floor", "score_gamma", "beta")}
                if method == "cpg_force_ratio":
                    result = sample_airfoil_cpg_force_ratio(coords, coarse, values, count, rng, mandatory, **common)
                else:
                    common.update({key: cfg[key] for key in ("ratio_weight", "lift_support_weight", "lift_balance_weight", "lift_reliability_scale")})
                    result = sample_airfoil_cpg_lift_support(coords, coarse, values, count, rng, mandatory, **common)
            elif method == "cpg":
                score = grad_score(coords[coarse], values, k=8)
                score = score[cKDTree(coords[coarse]).query(coords, k=1)[1]]
                result = sample_airfoil_cpg(coords, count, rng, score, mandatory)
            else:
                fkey = cache_key + (cfg["feature"], cfg["feature_k"], cfg["ell"], cfg["pressure_offset"])
                if fkey not in self.feature_cache:
                    if cfg["feature"] == "gradient_head":
                        if isinstance(head, Airfoil2dPressureHead) or head is None:
                            raise ValueError("gradient_head feature requires the trained gradient head")
                        score = values.reshape(-1)
                    else:
                        if head is not None and not isinstance(head, Airfoil2dPressureHead):
                            raise ValueError("pressure-derived acquisition requires a pressure head")
                        score = pressure_acquisition(coords, coarse, values, mandatory,
                                feature=cfg["feature"], k=cfg["feature_k"], ell=cfg["ell"], pressure_offset=cfg["pressure_offset"])
                    self.feature_cache[fkey] = (score, time.perf_counter()-t)
                score, feature_seconds = self.feature_cache[fkey]
                metadata["feature_seconds"] = feature_seconds
                if cfg["sampler"] == "patch":
                    pkey = (sample_id, n0, cfg["patch_count"], cfg["placement"], cfg["kappa"], cfg["beta"])
                    if pkey not in self.patch_cache:
                        setup_started = time.perf_counter()
                        sampler = Airfoil2dCA(coords, mandatory, cfg["coarse_fraction"], cfg["patch_count"],
                                             {b: cfg["kappa"] for b in BUDGETS}, {b: cfg["beta"] for b in BUDGETS}, cfg["placement"])
                        self.patch_cache[pkey] = (sampler, time.perf_counter()-setup_started)
                    sampler, setup_seconds = self.patch_cache[pkey]
                    t = time.perf_counter()
                    result = sampler.refine(coarse, score, count, budget, rng)
                    metadata["patch_setup_seconds"] = max(setup_seconds, 0.)
                else:
                    if len(score) != len(coords):
                        score = score[cKDTree(coords[coarse]).query(coords, k=1)[1]]
                    t = time.perf_counter()
                    result = sample_airfoil_cpg_single_feature(score, count, rng, mandatory,
                            uniform_floor=cfg["uniform_floor"], score_gamma=cfg["score_gamma"], beta=cfg["beta"],
                            normalization=cfg["normalization"], feature_name=cfg["feature"],
                            base_indices=coarse if cfg["sampler"] == "weighted_nested" else None)
            metadata["refinement_seconds"] = time.perf_counter()-t
            if method in ("cpg_force_ratio", "cpg_force_ratio_lift_support"):
                metadata["feature_seconds"] = result.metadata["feature_seconds"]
                metadata["refinement_seconds"] -= metadata["feature_seconds"]
            metadata["acquisition_seconds"] = coarse_sampling_seconds + seconds + metadata["feature_seconds"]
        if method in ("uniform", "pi", "pi_voronoi"):
            metadata["refinement_seconds"] = time.perf_counter()-started
        metadata["sampling_seconds"] = metadata["acquisition_seconds"] + metadata["refinement_seconds"] + metadata.get("patch_setup_seconds", 0.)
        result.validate(len(coords), count)
        if method == "ca" and not np.isin(coarse, result.indices).all():
            raise ValueError("CA final selection lost coarse points")
        result.metadata.update(metadata)
        result.metadata.update(nfull=len(coords), final_count=len(result.indices),
            new_point_count=len(result.indices) - (len(coarse) if coarse is not None and np.isin(coarse, result.indices).all() else 0),
            coarse_indices_hash=hashlib.sha256(coarse.tobytes()).hexdigest() if coarse is not None else None)
        return result, coarse, cache

    @torch.no_grad()
    def record(self, sample_id, seed, budget, config, *, visualization=False):
        sampling, coarse, cache = self.sampling(sample_id, seed, budget, config)
        coords, truth = self.data.coords(sample_id), self.data.target(sample_id)
        indices = sampling.indices
        pos = torch.as_tensor(coords[indices], device=self.device)[None]
        weights = None if config["method"] == "uniform" else torch.as_tensor(sampling.compensation, dtype=torch.float32, device=self.device)
        self.context.set(weights)
        start = synchronized_time(self.device)
        try:
            if cache is not None:
                output = cached_operator_forward(self.model, pos, pos, coarse, indices, cache, weights)
            else:
                output = self.model(pos, pos)
            prediction = self.data.decode(output)[0].cpu().numpy()
        finally:
            self.context.clear()
        final_seconds = synchronized_time(self.device)-start
        start = time.perf_counter()
        reconstructed, fallback = reconstruct_field(indices, prediction, self.reference)
        reconstruction_seconds = time.perf_counter()-start
        field_error = float(relative_l2(torch.from_numpy(reconstructed[None]), torch.from_numpy(truth[None]))[0])
        if not np.isfinite(field_error):
            field_error = None
        force_true = force_record(pressure_surface_force(coords, truth))
        force_pred = force_record(pressure_surface_force(coords, reconstructed))
        ratio_error = None if force_true["Cd_over_Cl"] is None or force_pred["Cd_over_Cl"] is None else abs(force_true["Cd_over_Cl"]-force_pred["Cd_over_Cl"])
        meta = sampling.metadata
        row = dict(method=config["name"], method_family=config["method"], sample_id=sample_id, seed=seed,
                   budget_fraction=budget, budget_count=len(indices), selected_count=len(indices),
                   selected_indices_hash=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
                   pressure_field_relative_l2=field_error, ratio_abs_error=ratio_error,
                   truth=force_true, prediction=force_pred, truth_access=False,
                   status="ok" if ratio_error is not None and field_error is not None else "nonfinite_metric",
                   interpolation_fallback_count=fallback, interpolation_backend="index_grid_linear_nearest",
                   operator_sha256=self.operator_hash, normalizer_sha256=self.data.normalizer_sha256,
                   config_hash=config_digest(config), resolved_method_config=config,
                   sampling_metadata=meta, sampling_seconds=meta["sampling_seconds"],
                   coarse_seconds=meta["coarse_seconds"], head_seconds=meta["head_seconds"],
                   acquisition_seconds=meta["acquisition_seconds"], inference_seconds=final_seconds,
                   reconstruction_seconds=reconstruction_seconds,
                   standalone_total_seconds=meta["sampling_seconds"]+final_seconds+reconstruction_seconds,
                   peak_memory_scope="process including resident heads and acquisition caches",
                   peak_memory_bytes=torch.cuda.max_memory_allocated(self.device) if torch.device(self.device).type == "cuda" else None)
        if visualization:
            self.last_visualization = dict(coords=coords, truth=truth, indices=indices,
                                           prediction=reconstructed, density=sampling.sampling_density)
        return row

    def evaluate(self, config, ids, seeds=(0,1,2), protocol="positive_lift", output=None):
        config = resolve_config(config["method"], config)
        selected = self.selected_ids(ids, protocol)
        if not selected:
            raise ValueError("no samples retained by test protocol")
        stream = None
        if output is not None:
            output = Path(output)
            raw_path = output.with_name(output.stem+"_raw.jsonl")
            if output.exists() or raw_path.exists():
                raise FileExistsError(f"evaluation artifact exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = raw_path.open("x")
        rows = []
        try:
            for sid in selected:
                # Reuse acquisition only across budgets/configs, not unbounded samples.
                self.coarse_cache.clear(); self.feature_cache.clear(); self.patch_cache.clear()
                for seed in seeds:
                    for budget in BUDGETS:
                        row = self.record(sid, seed, budget, config)
                        row["evaluation_protocol"] = protocol
                        rows.append(row)
                        if stream:
                            stream.write(json.dumps(row, allow_nan=False)+"\n")
                            stream.flush()
                if len(rows) % 90 == 0:
                    print(f"{config['name']}: {len(rows)} records", flush=True)
        finally:
            if stream:
                stream.close()
        summary = summarize_airfoil_records(rows, config["name"])
        summary.update(samples=len(selected), seeds=list(seeds), sample_ids=selected,
                       protocol=protocol, config=config, operator_sha256=self.operator_hash,
                       normalizer_sha256=self.data.normalizer_sha256, patched_attention_layers=self.patched)
        if output is not None:
            output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary, rows


def summarize_airfoil_records(records, method):
    if not records:
        raise ValueError("no Airfoil records")
    keys = [(r["sample_id"], r["seed"], r["budget_fraction"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Airfoil sample/seed/budget records")
    result = {"method": method, "budgets": {}}
    expected_pairs = {(r["sample_id"],r["seed"]) for r in records}
    for budget in BUDGETS:
        rows = [r for r in records if r["budget_fraction"] == budget]
        if {(r["sample_id"],r["seed"]) for r in rows} != expected_pairs:
            raise ValueError("missing Airfoil sample/seed/budget records")
        field = np.asarray([r["pressure_field_relative_l2"] for r in rows], dtype=float)
        ratio = np.asarray([r["ratio_abs_error"] if r["ratio_abs_error"] is not None else np.nan for r in rows], dtype=float)
        finite = np.isfinite(ratio).all()
        item = dict(count=rows[0]["budget_count"], records=len(rows),
                    mean_pressure_field_relative_l2=float(field.mean()) if np.isfinite(field).all() else None,
                    mean_ratio_abs_error=float(ratio.mean()) if finite else None,
                    median_ratio_abs_error=float(np.median(ratio)) if finite else None,
                    p90_ratio_abs_error=float(np.quantile(ratio,.9)) if finite else None,
                    max_ratio_abs_error=float(ratio.max()) if finite else None,
                    std_pressure_field_relative_l2=float(field.std()) if np.isfinite(field).all() else None,
                    nonfinite_pressure_count=int((~np.isfinite(field)).sum()),
                    std_ratio_abs_error=float(ratio.std()) if finite else None,
                    nonfinite_ratio_count=int((~np.isfinite(ratio)).sum()))
        for key in ("sampling_seconds", "coarse_seconds", "head_seconds", "acquisition_seconds", "inference_seconds", "reconstruction_seconds", "standalone_total_seconds"):
            item["mean_"+key] = float(np.mean([r.get(key,0.) for r in rows]))
        result["budgets"][f"{budget:.3f}"] = item
    return result


def airfoil_rauec(method_summary, uniform_summary, metric):
    for field in ("sample_ids", "seeds", "protocol", "operator_sha256", "normalizer_sha256"):
        if field in method_summary or field in uniform_summary:
            if method_summary.get(field) != uniform_summary.get(field):
                raise ValueError(f"method and Uniform differ in {field}")
    method = [method_summary["budgets"][f"{b:.3f}"][metric] for b in BUDGETS]
    uniform = [uniform_summary["budgets"][f"{b:.3f}"][metric] for b in BUDGETS]
    if any(x is None or not np.isfinite(x) for x in method+uniform):
        return None
    return relative_area_under_error_curve(BUDGETS, method, uniform)


def evaluate(method, checkpoint, manifest, data_path, output, device="cuda:0", samples=200,
             seeds=(0,1,2), ca_head_checkpoint=None, ca_head_manifest=None, *, method_config=None,
             sample_ids=None, protocol="positive_lift"):
    config = resolve_config(method, method_config)
    if ca_head_checkpoint:
        config["head_checkpoint"] = ca_head_checkpoint
    if ca_head_manifest:
        config["head_manifest"] = ca_head_manifest
    runner = AirfoilEvaluator(checkpoint, manifest, data_path, device)
    ids = list(range(1000, 1000+min(samples,200))) if sample_ids is None else list(sample_ids)
    return runner.evaluate(config, ids, seeds, protocol, output)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-path", default="/mnt/ai-material-data/shuaibinqu/NACA/data/naca")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--ca-head-checkpoint")
    parser.add_argument("--ca-head-manifest")
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--protocol", choices=("positive_lift","all"), default="positive_lift")
    a = parser.parse_args()
    ids = json.loads(a.sample_ids.read_text()) if a.sample_ids else None
    if isinstance(ids, dict):
        ids = ids.get("sample_ids", ids.get("selected_sample_ids"))
    print(json.dumps(evaluate(a.method, a.checkpoint, a.manifest, a.data_path, a.output, a.device,
                             a.samples, tuple(a.seeds), a.ca_head_checkpoint, a.ca_head_manifest,
                             method_config=a.method_config, sample_ids=ids, protocol=a.protocol), indent=2))


if __name__ == "__main__":
    main()
