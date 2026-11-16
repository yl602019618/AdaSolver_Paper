"""Official full20 evaluation; acquisition never receives a target field."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from adasolver.inference.plasticity2d_contract import (
    BUDGETS, METRIC_ID, N_COARSE, N_FULL, indices_hash, load_data, object_hash,
    rng_for, validate_indices, write_json, validate_method_config,
)
from adasolver.inference.plasticity2d_sampling import (
    REFERENCE, variation_features, solution_score, sample_pi_full20, refine_full20,
)
from adasolver.inference.ca.plasticity2d_full import load_full20_head
from adasolver.layers import install_compensation
from adasolver.layers.first_block import first_block_features, cached_first_block_forward
from adasolver.metrics.plasticity2d import full20_metrics
from adasolver.models.plasticity2d import load_plasticity2d_operator
from adasolver.models.checkpoint import sha256_file
from adasolver.utils.interpolation import reconstruct_field


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Full20Evaluator:
    def __init__(self, manifest, device="cuda:0", *, allow_smoke=False):
        self.manifest_path = Path(manifest).resolve()
        self.metadata = json.loads(self.manifest_path.read_text())
        if self.metadata.get("status") != "trained" and not allow_smoke:
            raise ValueError("evaluation requires a completed operator, or explicit smoke mode")
        if self.metadata.get("task_version") != "full20_v1":
            raise ValueError("full20 operator manifest required")
        self.device = device
        self.operator = load_plasticity2d_operator(self.metadata["checkpoint"], manifest, device)
        self.context, _ = install_compensation(self.operator)
        self.data = load_data(self.metadata["effective_config"], self.metadata["input_normalizer"])
        self.heads = {}
        self.acquisition_cache = {}

    def head(self, path):
        path = str(Path(path).resolve())
        if path not in self.heads:
            self.heads[path] = load_full20_head(path, self.metadata, self.device)
        return self.heads[path]

    @torch.no_grad()
    def acquire(self, method, sample_id, seed, pos, fx, config, reuse=False):
        coarse = rng_for(seed, sample_id, 10).choice(N_FULL, N_COARSE, replace=False)
        head_id = self.head(config["head_manifest"])[1]["head_sha256"] if method == "ca" else None
        key = (method, sample_id, seed, head_id, config.get("neighbors", 16))
        if reuse and key in self.acquisition_cache:
            return self.acquisition_cache[key]
        self.context.clear()
        sync(self.device)
        start = time.perf_counter()
        cp, cf = pos[:, coarse].to(self.device), fx[:, coarse].to(self.device)
        cache = None
        direct_score = False
        if method == "cpg":
            prediction = self.operator(cp, cf)[0].cpu().numpy()
        else:
            h, cache = first_block_features(self.operator, torch.cat((cp, cf), dim=-1), capture=True)
            prediction = self.head(config["head_manifest"])[0](h)[0].cpu().numpy()
            head_metadata = self.head(config["head_manifest"])[1]
            direct_score = head_metadata["target"] == "zscore_log1p_full20_gradient"
        sync(self.device)
        acquisition_seconds = time.perf_counter()-start
        start = time.perf_counter()
        if direct_score:
            log_score = prediction.reshape(-1)*head_metadata["target_std"]+head_metadata["target_mean"]
            features = {"g":np.expm1(np.maximum(log_score,0.))}
        else:
            features = variation_features(REFERENCE[coarse], prediction, config.get("neighbors", 16))
        score_seconds = time.perf_counter()-start
        result = (coarse, features, cache, acquisition_seconds, score_seconds, head_id)
        if reuse:
            self.acquisition_cache[key] = result
        return result

    @torch.no_grad()
    def predict(self, split_name, local_index, seed, budget_id, config, *, reuse_acquisition=False):
        validate_method_config(config)
        split = self.data.splits[split_name]
        sid = split.sample_ids[local_index]
        method = config["method"]
        budget = BUDGETS[budget_id]
        count = round(budget*N_FULL)
        pos, fx = split.pos[local_index:local_index+1], split.fx[local_index:local_index+1]
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        sync(self.device)
        started = time.perf_counter()
        acquisition_seconds = score_seconds = compensation_seconds = 0.
        coarse = cache = head_id = None
        self.context.clear()
        if method in ("cpg", "ca"):
            coarse, features, cache, acquisition_seconds, score_seconds, head_id = self.acquire(
                method, sid, seed, pos, fx, config, reuse_acquisition)
            score = solution_score(features, config)
        sample_start = time.perf_counter()
        sampling_metadata = {}
        if method == "uniform":
            indices = rng_for(seed, sid, 20, budget_id).choice(N_FULL, count, replace=False)
            compensation = np.ones(count, np.float32)
            density = np.ones(count)
        elif method == "pi":
            raw_fx = self.data.input_normalizer.decode(fx)[0, :, 0].numpy().reshape(101, 31)[:, 0]
            sampling = sample_pi_full20(count, rng_for(seed, sid, 21, budget_id), raw_fx, config)
        elif method in ("cpg", "ca"):
            refine_seed = int(rng_for(seed, sid, 22, budget_id).integers(2**31))
            sampling = refine_full20(coarse, score, count, refine_seed, config)
        else:
            raise ValueError(f"unknown full20 method {method}")
        if method != "uniform":
            indices, compensation, density = sampling.indices, sampling.compensation, sampling.sampling_density
            sampling_metadata = sampling.metadata
            compensation_seconds = sampling_metadata["compensation_seconds"]
        validate_indices(indices, count, coarse)
        sampling_seconds = time.perf_counter()-sample_start-compensation_seconds
        final_pos, final_fx = pos[:, indices].to(self.device), fx[:, indices].to(self.device)
        weights = torch.as_tensor(compensation, device=self.device)
        self.context.set(weights)
        sync(self.device)
        final_start = time.perf_counter()
        if method == "ca":
            mapping = np.empty(N_FULL, dtype=np.int64)
            mapping[indices] = np.arange(count)
            prediction = cached_first_block_forward(
                self.operator, torch.cat((final_pos, final_fx), dim=-1), mapping[coarse], cache, weights,
                chunk_size=count,
            )
        else:
            prediction = self.operator(final_pos, final_fx)
        sync(self.device)
        final_seconds = time.perf_counter()-final_start
        sparse = prediction[0].cpu().numpy()
        self.context.clear()
        reconstruct_start = time.perf_counter()
        reconstructed, fallback = reconstruct_field(indices, sparse, REFERENCE)
        reconstruction_seconds = time.perf_counter()-reconstruct_start
        total_seconds = time.perf_counter()-started
        metrics = full20_metrics(reconstructed, split.target[local_index].numpy())
        record = {
            "benchmark": "plasticity2d", "task_version": "full20_v1", "metric_id": METRIC_ID,
            "method": method, "config_id": object_hash(config), "effective_config": config,
            "sample_id": sid, "split": split_name, "seed": seed,
            "rng_scheme": "SeedSequence(seed,sample_id,stage,budget_id)_v1",
            "budget_fraction": budget, "budget_count": count, "selected_count": len(indices),
            "selected_indices": indices.tolist(), "selected_indices_hash": indices_hash(indices),
            "coarse_count": 0 if coarse is None else N_COARSE,
            "coarse_fraction": 0 if coarse is None else .1,
            "coarse_indices_hash": None if coarse is None else indices_hash(coarse),
            "operator_id": self.metadata["checkpoint_sha256"], "head_id": head_id,
            "truth_access_inference": False,
            "input_normalizer_id": self.metadata["input_normalizer_sha256"],
            "interpolation_backend": "scipy_linear_nearest_fallback",
            "reconstruction_coordinates": "index_lattice",
            "interpolation_fallback_count": fallback, "fallback_fraction": fallback/N_FULL,
            "compensation_type": sampling_metadata.get("compensation_type", "uniform_design_density"),
            "beta": config.get("beta", 0 if method == "uniform" else 1),
            "acquisition_seconds": acquisition_seconds, "score_seconds": score_seconds,
            "sampling_seconds": sampling_seconds, "compensation_seconds": compensation_seconds,
            "final_inference_seconds": final_seconds, "reconstruction_seconds": reconstruction_seconds,
            "total_seconds": total_seconds, "timing_mode": "search_cached" if reuse_acquisition else "independent_request",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if torch.device(self.device).type == "cuda" else None,
            "sampling_metadata": sampling_metadata, "status": "ok", **metrics,
        }
        arrays = {"indices": indices, "density": density, "compensation": compensation,
                  "prediction": reconstructed, "coarse": coarse}
        return record, arrays

    def evaluate(self, config, split, output, *, seeds=(0, 1, 2), sample_ids=None,
                 reuse_acquisition=False):
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        raw_path = destination / "raw.jsonl"
        if raw_path.exists():
            raise FileExistsError(raw_path)
        available = self.data.splits[split].sample_ids
        ids = list(available if sample_ids is None else sample_ids)
        if len(set(ids)) != len(ids) or not set(ids) <= set(available):
            raise ValueError("sample IDs must be unique members of the requested split")
        # Warm the exact method before measuring independent requests.
        self.predict(split, available.index(ids[0]), seeds[0], 0, config, reuse_acquisition=reuse_acquisition)
        records = []
        with raw_path.open("x", buffering=1) as stream:
            for budget_id in range(3):
                for seed in seeds:
                    for sid in ids:
                        try:
                            record, _ = self.predict(split, available.index(sid), seed, budget_id,
                                                     config, reuse_acquisition=reuse_acquisition)
                        except Exception as error:
                            stream.write(json.dumps({"sample_id": sid, "seed": seed,
                                "budget_fraction": BUDGETS[budget_id], "status": "failed",
                                "failure_reason": repr(error)}) + "\n")
                            raise
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        records.append(record)
        summary = {
            "method": config["method"], "config_id": object_hash(config), "effective_config": config,
            "metric_id": METRIC_ID, "operator_id": self.metadata["checkpoint_sha256"],
            "split": split, "sample_ids": ids, "seeds": list(seeds), "records": len(records),
            "budgets": list(BUDGETS),
            "mean_errors": [float(np.mean([r["full_field_relative_l2"] for r in records if r["budget_fraction"] == b])) for b in BUDGETS],
            "raw_file": str(raw_path.resolve()), "raw_sha256": sha256_file(raw_path),
        }
        write_json(destination / "summary.json", summary)
        return summary

    @torch.no_grad()
    def fullmesh(self, split_name, output):
        split = self.data.splits[split_name]
        self.context.clear()
        records = []
        for i, sid in enumerate(split.sample_ids):
            pred = self.operator(split.pos[i:i+1].to(self.device), split.fx[i:i+1].to(self.device))[0].cpu().numpy()
            records.append({"sample_id": sid, **full20_metrics(pred, split.target[i].numpy())})
        result = {"metric_id": METRIC_ID, "split": split_name, "operator_id": self.metadata["checkpoint_sha256"],
                  "mean_full_field_relative_l2": float(np.mean([r["full_field_relative_l2"] for r in records])),
                  "records": records}
        write_json(output, result)
        return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--method-config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "calibration", "confirmation", "test"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sample-ids", type=int, nargs="+")
    p.add_argument("--allow-smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(a.method_config).read_text())
    evaluator = Full20Evaluator(a.manifest, a.device, allow_smoke=a.allow_smoke)
    print(evaluator.evaluate(config, a.split, a.output, seeds=a.seeds, sample_ids=a.sample_ids))


if __name__ == "__main__":
    main()
