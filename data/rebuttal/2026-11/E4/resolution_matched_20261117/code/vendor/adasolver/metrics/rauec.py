from __future__ import annotations

import numpy as np


def relative_area_under_error_curve(budgets, method_errors, uniform_errors) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    method = np.asarray(method_errors, dtype=np.float64)
    uniform = np.asarray(uniform_errors, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or method.shape != x.shape or uniform.shape != x.shape:
        raise ValueError("budgets and error curves must be equal-length one-dimensional arrays")
    if not np.all(np.diff(x) > 0):
        raise ValueError("budgets must be strictly increasing")
    denominator = np.trapezoid(uniform, x)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("uniform curve must have positive finite area")
    return float(np.trapezoid(method, x) / denominator)
