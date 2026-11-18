from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/pipe/summary"
FIGURES = ROOT / "results/pipe/figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks = {}
    budgets = np.array([0.375, 0.5, 0.75])
    final = json.loads((SUMMARY / "official_comparison_cpg_ca_tuned.json").read_text())
    for method in ("uniform", "pi", "cpg_tuned", "ca_tuned"):
        curve = final["methods"][method]["curve"]
        area = float(np.trapezoid(curve, budgets))
        uniform_area = float(np.trapezoid(final["methods"]["uniform"]["curve"], budgets))
        expected = final["methods"][method]["rAUEC"]
        actual = area / uniform_area
        assert abs(actual - expected) < 1e-6, (method, actual, expected)
    checks["rauec_recompute"] = "PASS"

    for name in ("official_uniform", "official_pi", "official_cpg", "official_ca"):
        summary = json.loads((SUMMARY / f"{name}.json").read_text())
        assert all(row["records"] == 600 for row in summary["budgets"].values())
    checks["official_summary_records"] = "PASS"

    for name in ("ca_tuning_full_confirmation", "cpg_tuning_full_confirmation"):
        summary = json.loads((SUMMARY / f"{name}.json").read_text())
        assert summary["raw_records"] == 1800
        assert sum(1 for _ in (SUMMARY / f"{name}_raw.jsonl").open()) == 1800
    checks["tuned_raw_records"] = "PASS"

    operator = ROOT / "checkpoints/pipe/pipe_transolver3_varpts_nmin040_best.pt"
    head = ROOT / "checkpoints/pipe/pipe_ca_head_nmin040_best.pt"
    assert sha256(operator) == "7b98fb8e0e59cfbdd897b6a33d83fb545565f0aa3b2857688ae58507a65278c1"
    assert sha256(head) == "09926b14bb14c70ab41a694f2e1ef6801a6192eac1a703f149a1fe8889a4ca13"
    checks["checkpoint_hashes"] = "PASS"

    sample_dirs = [FIGURES / f"sample_{sample_id}" for sample_id in (1149, 1174, 1040)]
    for directory in sample_dirs:
        assert directory.is_dir()
        pngs = list(directory.glob("*.png"))
        assert len(pngs) >= 51, (directory, len(pngs))
        assert all(path.stat().st_size > 0 for path in pngs)
    checks["visualizations"] = "PASS"

    output = {"benchmark": "pipe", "status": "PASS", "checks": checks}
    path = ROOT / "results/pipe/done_gate.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
