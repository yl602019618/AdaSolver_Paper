#!/usr/bin/env python3
"""Validate the 97-point measurements and replace the Elasticity timing rows."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (.375, .5, .75)
METHODS = ("uniform", "pi", "cpg", "ca")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def group_stats(records):
    means = [r["mean_ms"] for r in records]
    return {
        "cases": len(means), "mean_ms": statistics.mean(means),
        "case_std_ms": statistics.stdev(means) if len(means) > 1 else None,
        "repeat_std_ms": math.sqrt(statistics.mean(statistics.variance(r["raw_ms"]) for r in records)),
        "repeats_per_case_min": min(r["n"] for r in records),
        "repeats_per_case_max": max(r["n"] for r in records),
    }


def simple_stats(values):
    return {"n": len(values), "mean_ms": statistics.mean(values),
            "case_std_ms": statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--adasolver-root", type=Path, default=ROOT.parent / "AdaSolver_official")
    args = parser.parse_args()
    run = args.run.resolve()
    rows = read_rows(run / "raw.jsonl")
    state = json.loads((run / "status.json").read_text())
    telemetry = json.loads((run / "gpu_telemetry.json").read_text())
    assert state["status"] == "complete" and state["records"] == len(rows) == 111
    assert telemetry["exclusive"] and not telemetry["errors"]
    expected = {(sid, seed, b, m) for sid in (1800, 1899, 1999)
                for seed in (0, 1, 2) for b in BUDGETS for m in METHODS}
    expected.update((sid, 0, 1., "full") for sid in (1800, 1899, 1999))
    assert {(r["sample_id"], r["seed"], r["budget"], r["method"]) for r in rows} == expected
    configurations = rows[0]["binding"]["configurations"]
    max_field_difference = 0.
    for row in rows:
        assert row["benchmark"] == "elasticity" and row["n_full"] == 972
        assert row["n_final"] == round(972 * row["budget"])
        assert row["n_coarse"] == (97 if row["method"] in {"cpg", "ca"} else 0)
        assert row["binding"]["configurations"] == configurations
        assert row["binding"]["head_parameters"] == 521
        assert row["binding"]["head_bottleneck"] == 4
        assert row["validation"]["indices_equal"], (row["sample_id"], row["method"], row["budget"])
        difference = row["validation"]["full_field_relative_difference"]
        assert difference <= 2e-5, difference
        max_field_difference = max(max_field_difference, difference)
        for category in ("timings", "stages"):
            for s in row[category].values():
                assert len(s["raw_ms"]) == s["n"] >= 2
                assert all(math.isfinite(t) and t >= 0 for t in s["raw_ms"])
                assert math.isclose(statistics.mean(s["raw_ms"]), s["mean_ms"], abs_tol=1e-8)
                assert math.isclose(statistics.stdev(s["raw_ms"]), s["std_ms"], abs_tol=1e-8)

    # Match the saved accuracy experiment, independently of the profiling adapter.
    accuracy_root = args.adasolver_root / "results/elasticity/budget10/final_evaluation/historical_test"
    saved = {}
    for method in METHODS:
        for row in read_rows(accuracy_root / f"{method}.raw.jsonl"):
            saved[(row["sample_id"], row["seed"], row["budget_fraction"], method)] = row["relative_l2"]
    max_error_difference = 0.
    for row in rows:
        if row["method"] == "full":
            continue
        key = row["sample_id"], row["seed"], row["budget"], row["method"]
        delta = abs(row["errors"]["original"]["relative_l2"] - saved[key])
        assert delta <= 2e-5, (key, delta)
        max_error_difference = max(max_error_difference, delta)

    tables = {name: [] for name in ("summary.csv", "stages.csv", "preparation.csv", "accuracy.csv")}
    uniform = {(r["sample_id"], r["seed"], r["budget"]): r for r in rows if r["method"] == "uniform"}
    for budget, method in [(b, m) for b in BUDGETS for m in METHODS] + [(1., "full")]:
        group = [r for r in rows if r["budget"] == budget and r["method"] == method]
        base = {"benchmark": "elasticity", "budget": budget, "method": method}
        for mode in ("sampled", "original", "fast"):
            ratios = [1. if method == "full" else
                      r["timings"][mode]["mean_ms"] /
                      uniform[r["sample_id"], r["seed"], budget]["timings"][mode]["mean_ms"] for r in group]
            tables["summary.csv"].append({**base, "mode": mode,
                **group_stats([r["timings"][mode] for r in group]),
                "paired_ratio_to_uniform_mean": statistics.mean(ratios),
                "paired_ratio_to_uniform_std": statistics.stdev(ratios)})
        for stage in group[0]["stages"]:
            tables["stages.csv"].append({**base, "stage": stage,
                                         **group_stats([r["stages"][stage] for r in group])})
        for component in ("adapter_increment_ms", "case_preparation_ms", "reported_shared_input_preparation_ms",
                          "gpu_runtime_and_reconstruction_preparation_ms", "first_sampled_call_ms", "graph_setup_ms"):
            tables["preparation.csv"].append({**base, "component": component,
                                               **simple_stats([r["preparation"][component] for r in group])})
        for mode in ("original", "fast"):
            tables["preparation.csv"].append({**base, "component": f"first_full_{mode}_after_sampled_ms",
                **simple_stats([r["preparation"]["first_full_calls_after_sampled_ms"][mode] for r in group])})
        for mode, metrics in group[0]["errors"].items():
            for metric in metrics:
                values = [r["errors"][mode][metric] for r in group]
                tables["accuracy.csv"].append({**base, "mode": mode, "metric": metric,
                    "cases": len(values), "mean": statistics.mean(values), "case_std": statistics.stdev(values)})
    loads = json.loads((run / "shared_loads.json").read_text())
    tables["shared_loads.csv"] = [{"benchmark": "elasticity",
                                   **simple_stats([r["suite_load_ms"] for r in loads])}]

    # Prepare every replacement before writing any of the existing data tables.
    replacements = {}
    for name, new in tables.items():
        path = ROOT / "data/inference_timing" / name
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, old = reader.fieldnames, list(reader)
        dimensions = ["benchmark"] + [k for k in ("budget", "method", "mode", "stage", "component", "metric") if k in fields]
        def key(row):
            return tuple(float(row[k]) if k == "budget" else row[k] for k in dimensions)
        lookup = {key(r): r for r in new}
        old_keys = {key(r) for r in old if r["benchmark"] == "elasticity"}
        assert old_keys == set(lookup), (name, old_keys ^ set(lookup))
        combined = [lookup[key(r)] if r["benchmark"] == "elasticity" else r for r in old]
        replacements[path] = fields, combined, "\r\n" if b"\r\n" in path.read_bytes() else "\n"

    destination = ROOT / "data/elasticity_cpg10_timing"
    destination.mkdir(exist_ok=True)
    for path, (fields, values, line_end) in replacements.items():
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator=line_end)
            writer.writeheader()
            writer.writerows(values)
    for name in ("raw.jsonl", "environment.json", "gpu_telemetry.json"):
        shutil.copy2(run / name, destination / name)
    (destination / "configurations.json").write_text(json.dumps(rows[0]["binding"], indent=2) + "\n")
    verification = {
        "records": 111, "samples": [1800, 1899, 1999], "seeds": [0, 1, 2],
        "final_counts": [364, 486, 729], "coarse_count": 97,
        "all_reference_indices_equal": True, "gpu_exclusive": True,
        "max_full_field_relative_difference": max_field_difference,
        "max_absolute_error_difference_from_saved_accuracy": max_error_difference,
    }
    (destination / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
