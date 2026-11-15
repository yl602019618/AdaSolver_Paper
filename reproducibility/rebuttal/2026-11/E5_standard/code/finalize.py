"""Aggregate completed runs while keeping training and sampling variance separate."""
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

from common import ROOT, configuration, digest, read, write
import numpy as np

BENCHMARKS = ("elasticity", "pipe", "darcy")
METHODS = ("uniform", "pi", "cpg", "ca")
BUDGETS = (.375, .5, .75)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trapezoid(values):
    return sum((BUDGETS[i+1] - BUDGETS[i]) * (values[i] + values[i+1]) / 2 for i in range(2))


def main():
    all_rows, registry, checks, means = [], [], [], {}
    baseline_indices = {}
    from adasolver.inference.elasticity_contract import sample_rng
    from adasolver.utils.seed import make_numpy_rng
    for benchmark in BENCHMARKS:
        cfg = configuration(benchmark)
        expected_ids = cfg["split"]["test_ids"]
        expected = {(sid, None, "full", 1.) for sid in expected_ids}
        expected.update((sid, s, m, b) for sid in expected_ids for s in (0, 1, 2)
                        for m in METHODS for b in BUDGETS)
        backbone_ids = []
        for training_seed in (0, 1, 2):
            directory = ROOT / "records" / benchmark / f"seed{training_seed}"
            completion = read(directory / "completion.json")
            assert completion["status"] == "completed" and completion["records"] == 7400
            checkpoint = ROOT / "checkpoints" / benchmark / f"seed{training_seed}"
            model_meta = read(checkpoint / "backbone/manifest.json")
            head_meta = read(checkpoint / "head/manifest.json")
            assert model_meta["status"] == head_meta["status"] == "completed"
            assert model_meta["epochs_completed"] == cfg["training"]["epochs"]
            assert head_meta["epochs_completed"] == cfg["head"]["epochs"]
            assert model_meta["training_seed"] == head_meta["training_seed"] == head_meta["head_seed"] == training_seed
            assert not model_meta["test_used_for_selection"] and not head_meta["test_used_for_selection"]
            assert model_meta["split"] == cfg["split"]
            assert head_meta["train_ids"] == cfg["split"]["train_ids"]
            assert head_meta["validation_ids"] == cfg["split"]["validation_ids"]
            assert digest(checkpoint / "backbone/best.pt") == model_meta["best_sha256"] == head_meta["operator_sha256"]
            assert digest(checkpoint / "head/best.pt") == head_meta["best_sha256"]
            assert digest(ROOT / "data" / benchmark / "normalizer.npz") == model_meta["normalizer_sha256"] == head_meta["normalizer_sha256"]
            assert completion["operator_sha256"] == model_meta["best_sha256"]
            assert completion["head_sha256"] == head_meta["best_sha256"]
            backbone_ids.append(model_meta["best_sha256"])
            rows = [json.loads(line) for name in completion["files"]
                    for line in (directory / name).read_text().splitlines() if line]
            keys = [(r["sample_id"], r["sampling_seed"], r["method"], r["budget_fraction"]) for r in rows]
            assert len(keys) == len(set(keys)) == 7400 and set(keys) == expected
            groups = defaultdict(list)
            for row in rows:
                assert row["benchmark"] == benchmark and row["training_seed"] == training_seed
                assert row["status"] == "ok" and row["nfull"] == cfg["nfull"]
                assert row["selected_count"] == round(cfg["nfull"] * row["budget_fraction"])
                assert row["operator_sha256"] == model_meta["best_sha256"]
                assert row["head_sha256"] == (head_meta["best_sha256"] if row["method"] == "ca" else None)
                n0 = 97 if benchmark == "elasticity" else round(.1 * cfg["nfull"])
                assert row["coarse_count"] == (n0 if row["method"] in ("cpg", "ca") else 0)
                assert math.isfinite(row["relative_l2"]) and row["relative_l2"] >= 0
                recomputed = row["relative_l2_numerator"] / max(row["relative_l2_denominator"], 1e-12)
                assert math.isclose(recomputed, row["relative_l2"], abs_tol=1e-14)
                groups[row["method"], row["budget_fraction"], row["sampling_seed"]].append(row["relative_l2"])
                all_rows.append(row)
            for key, values in groups.items():
                assert len(values) == 200
                average = float(np.mean(values))
                assert abs(average - math.fsum(values) / 200) <= 1e-12
                means[(benchmark, training_seed, *key)] = average
            for sampling_seed in (0, 1, 2):
                for budget in BUDGETS:
                    count = round(cfg["nfull"] * budget)
                    archive = np.load(directory / f"sampling{sampling_seed}_n{count}.npz")
                    n0 = 97 if benchmark == "elasticity" else round(.1 * cfg["nfull"])
                    coarse = []
                    for local, sid in enumerate(expected_ids):
                        rng = (sample_rng(sid, sampling_seed) if benchmark == "elasticity" else
                               make_numpy_rng(sampling_seed, sid) if benchmark == "darcy" else
                               np.random.default_rng(1000 * sampling_seed + local))
                        coarse.append(rng.choice(cfg["nfull"], n0, replace=False))
                    coarse = np.stack(coarse)
                    for method in METHODS:
                        indices = archive[method]
                        assert indices.shape == (200, count)
                        assert int(indices.max()) < cfg["nfull"]
                        assert np.all(np.diff(np.sort(indices.astype(np.int32), axis=1), axis=1) > 0)
                        if method in ("cpg", "ca"):
                            assert np.array_equal(indices[:, :n0], coarse)
                        if method in ("uniform", "pi"):
                            key = benchmark, sampling_seed, budget, method
                            if training_seed == 0:
                                baseline_indices[key] = indices.copy()
                            else:
                                assert np.array_equal(indices, baseline_indices[key])
            registry.append({
                "dataset": benchmark, "training_seed": training_seed, "head_seed": training_seed,
                "backbone": "Transolver3", "backbone_epochs": model_meta["epochs_completed"],
                "head_epochs": head_meta["epochs_completed"], "backbone_best_epoch": model_meta["best_epoch"],
                "head_best_epoch": head_meta["best_epoch"], "train_cases": 1000, "validation_cases": 200,
                "test_cases": 200, "nfull": cfg["nfull"], "training_min_points": cfg["training"]["nmin_count"],
                "checkpoint_sha256": model_meta["best_sha256"], "head_sha256": head_meta["best_sha256"],
                "normalizer_sha256": model_meta["normalizer_sha256"],
                "backbone_training_seconds": model_meta["elapsed_seconds"],
                "head_training_seconds": head_meta["elapsed_seconds"],
                "head_training_device": head_meta.get("training_device", "cuda:0"),
                "checkpoint_path": f"checkpoints/{benchmark}/seed{training_seed}/backbone/best.pt",
                "head_path": f"checkpoints/{benchmark}/seed{training_seed}/head/best.pt",
                "test_used_for_checkpoint_selection": False, "verified": True,
            })
            checks.append({"dataset": benchmark, "training_seed": training_seed, "records": len(rows),
                           "condition_matrix_complete": True, "saved_indices_valid": True})
        assert len(set(backbone_ids)) == 3, benchmark
    assert len(all_rows) == 66600
    replay = read(ROOT / "internal/canonical_evaluator_check.json")
    assert replay["status"] == "passed" and len(replay["checks"]) == 12
    assert all(r["indices_equal"] and r["absolute_error_difference"] < 2e-6 for r in replay["checks"])

    absolute, sampling, rauec, paired = [], [], [], []
    model_means = {}
    for benchmark in BENCHMARKS:
        for method in ("full", *METHODS):
            for budget in ((1.,) if method == "full" else BUDGETS):
                across = []
                for training_seed in (0, 1, 2):
                    within = [means[benchmark, training_seed, method, budget, s]
                              for s in ((None,) if method == "full" else (0, 1, 2))]
                    value = statistics.mean(within)
                    across.append(value)
                    model_means[benchmark, training_seed, method, budget] = value
                    sampling.append({"dataset": benchmark, "training_seed": training_seed,
                        "head_seed": training_seed if method == "ca" else None,
                        "method": method, "budget_fraction": budget, "mean_error": value,
                        "sampling_seed0": within[0], "sampling_seed1": within[1] if len(within)>1 else None,
                        "sampling_seed2": within[2] if len(within)>1 else None,
                        "sampling_seed_sd": statistics.stdev(within) if len(within)>1 else None,
                        "cases_per_sampling_seed": 200})
                absolute.append({"dataset": benchmark, "metric": "field_relative_l2", "method": method,
                    "budget_fraction": budget, "training_seed0": across[0], "training_seed1": across[1],
                    "training_seed2": across[2], "mean": statistics.mean(across),
                    "training_seed_sd": statistics.stdev(across), "independent_training_runs": 3,
                    "cases_per_training_run": 200, "sampling_seeds_per_training_run": 1 if method == "full" else 3})
        for method in METHODS:
            training_values = []
            for training_seed in (0, 1, 2):
                ratios = []
                for sampling_seed in (0, 1, 2):
                    v = [means[benchmark, training_seed, method, b, sampling_seed] for b in BUDGETS]
                    u = [means[benchmark, training_seed, "uniform", b, sampling_seed] for b in BUDGETS]
                    ratio = trapezoid(v) / trapezoid(u)
                    independently = float(np.trapezoid(v, BUDGETS) / np.trapezoid(u, BUDGETS))
                    assert abs(ratio - independently) <= 1e-12
                    ratios.append(ratio)
                training_values.append(statistics.mean(ratios))
            rauec.append({"dataset": benchmark, "metric": "field", "method": method,
                "training_seed0": training_values[0], "training_seed1": training_values[1],
                "training_seed2": training_values[2], "mean": statistics.mean(training_values),
                "training_seed_sd": statistics.stdev(training_values), "independent_training_runs": 3,
                "aggregation": "matched sampling-seed area ratios averaged within training run; sample SD across training runs"})
        for method, reference in (("pi", "uniform"), ("cpg", "uniform"), ("ca", "uniform"),
                                  ("cpg", "pi"), ("ca", "pi"), ("ca", "cpg")):
            for budget in BUDGETS:
                values = [model_means[benchmark, t, method, budget] - model_means[benchmark, t, reference, budget]
                          for t in (0, 1, 2)]
                paired.append({"dataset": benchmark, "method": method, "reference": reference,
                    "budget_fraction": budget, "training_seed0_delta": values[0],
                    "training_seed1_delta": values[1], "training_seed2_delta": values[2],
                    "mean_paired_delta": statistics.mean(values), "training_seed_sd": statistics.stdev(values),
                    "independent_training_runs": 3, "negative_means": "lower error than reference"})
    destination = ROOT / "results"
    for table, suffix in ((absolute, ""), (rauec, ""), (paired, "_delta")):
        for row in table:
            values = [row[f"training_seed{t}{suffix}"] for t in (0, 1, 2)]
            average = row["mean"] if "mean" in row else row["mean_paired_delta"]
            assert abs(average - float(np.mean(values))) <= 1e-12
            assert abs(row["training_seed_sd"] - float(np.std(values, ddof=1))) <= 1e-12
    csv_write(destination / "training_seed_summary.csv", absolute)
    csv_write(destination / "sampling_variability.csv", sampling)
    csv_write(destination / "training_seed_rauec.csv", rauec)
    csv_write(destination / "paired_training_differences.csv", paired)
    csv_write(destination / "checkpoint_registry.csv", registry)
    csv_write(destination / "matrix_checks.csv", checks)
    csv_write(destination / "metrics_records.csv.gz", all_rows)
    write(destination / "verification.json", {
        "status": "verified", "backbones": 9, "heads": 9, "benchmarks": list(BENCHMARKS),
        "raw_records": len(all_rows), "complete_case_seed_budget_matrices": True,
        "distinct_initialization_checkpoints": True, "saved_indices_valid": True,
        "uniform_pi_indices_match_across_training_models": True,
        "coarse_prefixes_match_declared_sampling_rngs": True,
        "test_used_for_checkpoint_selection": False,
        "aggregation_checked_with_numpy_and_standard_library": True,
        "canonical_darcy_evaluator_checks": len(replay["checks"]),
        "absolute_summary_rows": len(absolute), "rauec_rows": len(rauec),
        "sampling_variability_rows": len(sampling), "paired_summary_rows": len(paired),
    })
    state = read(ROOT / "run_manifest.json")
    state.update(execution_status="completed", evidence_status="verified", publication_status="internal_only")
    write(ROOT / "run_manifest.json", state)
    print(json.dumps(read(destination / "verification.json"), indent=2))


if __name__ == "__main__":
    main()
