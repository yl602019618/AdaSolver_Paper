"""Independent, paired raw-record summaries for the two Car metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adasolver.inference.shapenet_car_contract import BUDGETS, PROTOCOL, sample_fold

METRICS = {"field": "field_error", "cd": "cd_relative_error"}


def record_key(row):
    return str(row["sample_id"]), int(row["seed"]), float(row["budget_fraction"])


def summarize_records(rows, *, budgets=BUDGETS):
    if not rows:
        raise ValueError("empty Car records")
    keys = [record_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample/seed/budget keys")
    ids, seeds = sorted({k[0] for k in keys}), sorted({k[1] for k in keys})
    expected = {(sid, seed, float(b)) for sid in ids for seed in seeds for b in budgets}
    if set(keys) != expected:
        raise ValueError("missing or unexpected sample/seed/budget keys")
    result = {"samples": len(ids), "seeds": seeds, "records": len(rows), "budgets": {}}
    for budget in budgets:
        records = [r for r in rows if r["budget_fraction"] == budget]
        if any(r.get("cd_status") != "ok" for r in records):
            raise ValueError("missing valid Cd records; cannot change the denominator")
        entry = {"records": len(records), "cd_valid_records": len(records)}
        for metric in ("velocity_rel_l2", "pressure_rel_l2", "field_error", "cd_relative_error"):
            values = np.asarray([r.get(metric, np.nan) for r in records], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"missing finite {metric} records")
            means = [float(np.mean([r[metric] for r in records if r["seed"] == seed])) for seed in seeds]
            entry["mean_" + metric] = float(np.mean(means))
            entry["per_seed_" + metric] = dict(zip(map(str, seeds), means))
            entry["std_" + metric] = float(np.std(means, ddof=1)) if len(means) > 1 else None
        for key in ("geometry_seconds", "coarse_seconds", "head_seconds", "feature_head_seconds", "score_seconds",
                    "refinement_seconds", "final_inference_seconds", "reconstruction_seconds",
                    "cd_seconds", "total_seconds", "peak_memory_bytes"):
            entry["mean_" + key] = float(np.mean([r.get(key, 0.) for r in records]))
        result["budgets"][f"{budget:.3f}"] = entry
    return result


def load_run(path, *, calibration=False):
    path = Path(path)
    root = path if path.is_dir() else path.parent
    manifest = json.loads((root / "run_manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines() if line.strip()]
    if manifest.get("status") != "complete" or manifest.get("protocol_version") != PROTOCOL:
        raise ValueError("ranking requires a completed budget10 run")
    expected = {(sid, seed, float(b)) for sid in manifest["sample_ids"]
                for seed in manifest["seeds"] for b in manifest["budgets"]}
    if {record_key(r) for r in rows} != expected:
        raise ValueError("raw keys do not match declared run sample/seed/budget set")
    for row in rows:
        for key in ("run_id", "operator_sha256", "normalizer_sha256", "protocol_version", "split_hash"):
            if row.get(key) != manifest.get(key):
                raise ValueError(f"mixed raw provenance: {key}")
        if row.get("selected_count") != row.get("budget_count"):
            raise ValueError("selected_count differs from budget")
        if calibration and sample_fold(row["sample_id"]) == 0:
            raise ValueError("calibration artifact contains fold 0")
    if calibration and manifest.get("split_role") != "calibration":
        raise ValueError("selection requires the calibration split")
    return manifest, rows, summarize_records(rows, budgets=manifest["budgets"])


def rauec(values, uniform, budgets=BUDGETS):
    integrate = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
    denominator = float(integrate(uniform, budgets))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("invalid Uniform integration denominator")
    return float(integrate(values, budgets) / denominator)


def compare(uniform_path, method_path, *, calibration=False):
    u_manifest, u_rows, uniform = load_run(uniform_path, calibration=calibration)
    m_manifest, m_rows, method = load_run(method_path, calibration=calibration)
    if u_manifest["config"]["method"] != "uniform":
        raise ValueError("denominator must be a Uniform run")
    for key in ("operator_sha256", "normalizer_sha256", "protocol_version", "split_hash",
                "split_role", "reconstruction", "source_sha256"):
        if u_manifest.get(key) != m_manifest.get(key):
            raise ValueError(f"incompatible Uniform/method provenance: {key}")
    if {record_key(r) for r in u_rows} != {record_key(r) for r in m_rows}:
        raise ValueError("Uniform and method sample/seed/budget keys differ")
    if tuple(m_manifest["budgets"]) != BUDGETS:
        raise ValueError("rAUEC requires all three prescribed budgets")
    ratios = {name: rauec([method["budgets"][f"{b:.3f}"]["mean_" + key] for b in BUDGETS],
                         [uniform["budgets"][f"{b:.3f}"]["mean_" + key] for b in BUDGETS])
              for name, key in METRICS.items()}
    return {"uniform_run": str(Path(uniform_path).resolve()), "method_run": str(Path(method_path).resolve()),
            "config": m_manifest["config"], "rAUEC": ratios, "summary": method,
            "uniform_run_id": u_manifest["run_id"], "method_run_id": m_manifest["run_id"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = [compare(args.uniform, path, calibration=args.calibration) for path in args.methods]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: r[k] for k in ("config", "rAUEC")} for r in result], indent=2))


if __name__ == "__main__":
    main()
