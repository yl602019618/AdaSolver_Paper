from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .airfoil2d import BUDGETS, airfoil_rauec


def compare_airfoil_summaries(uniform_path: str | Path, method_paths: dict[str, str | Path]) -> dict:
    uniform = json.loads(Path(uniform_path).read_text(encoding="utf-8"))
    result = {"uniform": uniform, "methods": {}}
    for method, path in method_paths.items():
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        result["methods"][method] = {
            "summary": summary,
            "rAUEC_pressure": airfoil_rauec(summary, uniform, "mean_pressure_field_relative_l2"),
            "rAUEC_Cd_over_Cl": airfoil_rauec(summary, uniform, "mean_ratio_abs_error"),
        }
    return result
