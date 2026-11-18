"""Recompute full20 curves and paired sample-cluster intervals from raw records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from adasolver.inference.plasticity2d_contract import BUDGETS, METRIC_ID, indices_hash, validate_indices, write_json, rng_for, object_hash
from adasolver.metrics.rauec import relative_area_under_error_curve


def compare(directories, output, *, bootstrap_repeats=2000):
    records, matrices, ids = {}, {}, None
    seeds = None
    operator_ids, normalizers, method_configs = set(), set(), {}
    for directory in directories:
        directory = Path(directory)
        rows = [json.loads(line) for line in (directory/"raw.jsonl").read_text().splitlines()]
        if not rows or any(r.get("status") != "ok" for r in rows):
            raise ValueError("missing or failed raw evaluation")
        method = rows[0]["method"]
        if method in records:
            raise ValueError("compare requires one frozen configuration per method")
        sample_ids = sorted({r["sample_id"] for r in rows})
        seed_ids = sorted({r["seed"] for r in rows})
        if ids is None:
            ids, seeds = sample_ids, seed_ids
        if ids != sample_ids or seeds != seed_ids:
            raise ValueError("methods use different samples or seeds")
        expected = {(sid, seed, b) for sid in ids for seed in seeds for b in BUDGETS}
        actual = [(r["sample_id"], r["seed"], r["budget_fraction"]) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("missing or duplicate raw record keys")
        config_ids = {r["config_id"] for r in rows}
        if len(config_ids) != 1:
            raise ValueError("raw file mixes method configurations")
        if len({r["head_id"] for r in rows}) != 1:
            raise ValueError("raw file mixes head checkpoints")
        matrix = np.empty((len(ids), len(seeds), 3))
        for r in rows:
            if r["metric_id"] != METRIC_ID or r["method"] != method:
                raise ValueError("mixed metric or method")
            if object_hash(r["effective_config"]) != r["config_id"]:
                raise ValueError("raw config hash mismatch")
            count = round(r["budget_fraction"]*3131)
            validate_indices(r["selected_indices"], count)
            if r["selected_count"] != count or indices_hash(r["selected_indices"]) != r["selected_indices_hash"]:
                raise ValueError("raw selected point count/hash mismatch")
            if method in ("cpg", "ca") and r["coarse_count"] != 313:
                raise ValueError("coarse budget differs from 10%")
            if method in ("cpg", "ca"):
                coarse = rng_for(r["seed"], r["sample_id"], 10).choice(3131, 313, replace=False)
                validate_indices(r["selected_indices"], count, coarse)
                if indices_hash(coarse) != r["coarse_indices_hash"]:
                    raise ValueError("coarse indices do not match the declared RNG")
            recomputed = np.sqrt(r["squared_error_total"])/max(np.sqrt(r["target_energy_total"]), 1e-12)
            if not np.isclose(recomputed, r["full_field_relative_l2"], rtol=1e-10, atol=1e-12):
                raise ValueError("full-field metric does not match raw energies")
            matrix[ids.index(r["sample_id"]), seeds.index(r["seed"]), BUDGETS.index(r["budget_fraction"])] = recomputed
            operator_ids.add(r["operator_id"])
            normalizers.add(r["input_normalizer_id"])
        records[method], matrices[method] = rows, matrix
        method_configs[method] = rows[0]["effective_config"]
    if "uniform" not in matrices or len(operator_ids) != 1 or len(normalizers) != 1:
        raise ValueError("one shared operator/normalizer and Uniform denominator are required")
    if "cpg" in records and "ca" in records:
        def coarse_map(rows):
            return {(r["sample_id"], r["seed"], r["budget_fraction"]): r["coarse_indices_hash"] for r in rows}
        if coarse_map(records["cpg"]) != coarse_map(records["ca"]):
            raise ValueError("CPG and CA coarse sets differ")
    uniform = matrices["uniform"].mean(axis=(0, 1))
    draws = np.random.default_rng(731).integers(len(ids), size=(bootstrap_repeats, len(ids)))
    boot_uniform = matrices["uniform"].mean(axis=1)[draws].mean(axis=1)
    denominator = np.trapezoid(boot_uniform, x=BUDGETS, axis=1)
    result = {"metric_id": METRIC_ID, "operator_id": next(iter(operator_ids)),
              "sample_ids": ids, "seeds": seeds, "budgets": list(BUDGETS),
              "bootstrap": {"unit": "sample_cluster_preserving_seeds_budgets_methods", "repeats": bootstrap_repeats, "seed": 731},
              "methods": {}}
    boot_ratios = {}
    timing_keys = ("acquisition_seconds", "score_seconds", "sampling_seconds", "compensation_seconds",
                   "final_inference_seconds", "reconstruction_seconds", "total_seconds")
    for method, values in matrices.items():
        curve = values.mean(axis=(0, 1))
        boot_curve = values.mean(axis=1)[draws].mean(axis=1)
        boot_ratio = np.trapezoid(boot_curve, x=BUDGETS, axis=1)/denominator
        boot_ratios[method] = boot_ratio
        per_seed = values.mean(axis=0)
        result["methods"][method] = {
            "mean_errors": curve.tolist(), "per_seed_mean_errors": per_seed.tolist(),
            "seed_std": per_seed.std(axis=0, ddof=1 if len(seeds)>1 else 0).tolist(),
            "rauec": relative_area_under_error_curve(BUDGETS, curve, uniform),
            "rauec_ci95": np.quantile(boot_ratio, [.025,.975]).tolist(),
            "records": len(records[method]), "config": method_configs[method],
            "mean_timing": {k: float(np.mean([r[k] for r in records[method]])) for k in timing_keys},
            "mean_fallback_fraction": float(np.mean([r["fallback_fraction"] for r in records[method]])),
        }
    if "pi" in boot_ratios:
        for method in ("cpg", "ca"):
            if method in boot_ratios:
                result["methods"][method]["rauec_difference_vs_pi_ci95"] = np.quantile(
                    boot_ratios[method]-boot_ratios["pi"], [.025,.975]).tolist()
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.runs, a.output), indent=2))


if __name__ == "__main__":
    main()
