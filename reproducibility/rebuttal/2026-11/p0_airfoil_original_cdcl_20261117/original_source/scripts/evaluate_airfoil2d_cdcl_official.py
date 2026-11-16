from __future__ import annotations

"""Official Airfoil2D evaluation for the frozen single-feature Cd/Cl sampler.

The deployable sampler receives only coordinates and a coarse frozen-operator
pressure prediction. Truth is used only after sampling for offline metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adasolver.data import Airfoil2DAdapter
from adasolver.inference.cpg.airfoil2d_single_feature import sample_airfoil_cpg_single_feature
from adasolver.layers import install_compensation
from adasolver.models import load_airfoil2d_operator
from adasolver.metrics import relative_area_under_error_curve
from scripts.deep_search_airfoil_cdcl_single_feature import (
    BUDGETS,
    _summary,
    evaluate,
    scalar_feature,
)

OFFICIAL_START, OFFICIAL_END = 1000, 1200
FROZEN = {
    "name": "signed_f0.5_g0.5_b1",
    "kind": "signed_cancellation",
    "ell": 0.2,
    "denominator_policy": "scale_invariant",
    "extension": "kernel_max",
    "coarse_fraction": 0.25,
    "normalization": "q95",
    "uniform_floor": 0.5,
    "score_gamma": 0.5,
    "beta": 1.0,
}


def _posthoc_signed(rows: list[dict], name: str) -> dict:
    selected = [r for r in rows if r["method"] == name and r["truth_Cl"] > 0.01]
    uniform = [r for r in rows if r["method"] == "uniform" and r["truth_Cl"] > 0.01]
    summary = _summary(uniform + selected, ["uniform", name])
    return {
        "filter": "truth_Cl > 0.01",
        "records": len(selected),
        "summary": summary[name],
        "uniform_summary": summary["uniform"],
        "selection_use": "posthoc reporting only; excluded from sampling and candidate choice",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/mnt/ai-material-data/shuaibinqu/NACA/data/naca")
    parser.add_argument("--checkpoint", default="checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_best.pt")
    parser.add_argument("--manifest", default="checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_manifest.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if FROZEN["coarse_fraction"] > .10:
        raise SystemExit("This archived 25% candidate exceeds the current 10% cap. "
                         "Use python -m adasolver.evaluation.airfoil2d with a budget10 method config.")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    data = Airfoil2DAdapter(args.data_path).load()
    model = load_airfoil2d_operator(args.checkpoint, args.manifest, device=args.device)
    context, patched = install_compensation(model)
    data.y_normalizer.to(args.device)
    x = np.load(Path(args.data_path) / "NACA_Cylinder_X.npy", mmap_mode="r")
    y = np.load(Path(args.data_path) / "NACA_Cylinder_Y.npy", mmap_mode="r")
    q = np.load(Path(args.data_path) / "NACA_Cylinder_Q.npy", mmap_mode="r")
    ids = np.arange(OFFICIAL_START, OFFICIAL_END, dtype=np.int64)
    if len(x) < OFFICIAL_END:
        raise SystemExit("dataset does not contain the exact official interval [1000,1200)")
    coords = np.stack((np.asarray(x[ids]), np.asarray(y[ids])), axis=-1).reshape(len(ids), -1, 2)
    truth = np.asarray(q[ids, 3]).reshape(len(ids), -1, 1)
    rows, summary = evaluate(model, context, data.y_normalizer, coords, truth, ids,
                             [FROZEN], tuple(args.seeds), data.mandatory_indices)
    method = FROZEN["name"]
    raw_path = args.output.with_name(args.output.stem + "_raw.jsonl")
    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    uniform_summary = _summary(rows, ["uniform"])["uniform"]
    result = {
        "status": "official_frozen_evaluation",
        "benchmark": "airfoil2d",
        "official_test_ids": [OFFICIAL_START, OFFICIAL_END],
        "official_test_ids_accessed": True,
        "truth_access": "offline metrics only; sampler uses coordinates and coarse pressure prediction",
        "ratio_protocol": "raw Cd/Cl without epsilon, clipping, or silent filtering",
        "selected_candidate": FROZEN,
        "candidate_reselected": False,
        "candidate_frozen_before_official": True,
        "seeds": list(args.seeds),
        "budgets": list(BUDGETS),
        "samples": len(ids),
        "patched_attention_layers": patched,
        "summary": summary[method],
        "uniform_summary": uniform_summary,
        "rAUEC": {
            "pressure": summary[method]["rAUEC_pressure"],
            "Cd_over_Cl": summary[method]["rAUEC_Cd_over_Cl"],
        },
        "positive_lift_posthoc": _posthoc_signed(rows, method),
        "raw_path": str(raw_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    audit = {
        "status": "official_frozen_evaluation",
        "official_interval_exact": True,
        "ids": ids.tolist(),
        "id_count": len(ids),
        "candidate": FROZEN,
        "raw_records": len(rows),
        "truth_access_sampler": False,
        "metrics_truth_access": "offline",
        "raw_path": str(raw_path),
        "summary_path": str(args.output),
        "no_search_or_retuning": True,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("OUTPUT", args.output)
    print("RAW", raw_path)
    print("AUDIT", audit_path)
    print("R-AUEC Cd/Cl", result["rAUEC"]["Cd_over_Cl"])
    print("R-AUEC pressure", result["rAUEC"]["pressure"])


if __name__ == "__main__":
    main()
