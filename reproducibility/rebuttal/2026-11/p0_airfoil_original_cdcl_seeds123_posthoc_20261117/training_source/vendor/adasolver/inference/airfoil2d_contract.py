"""Airfoil acquisition budgets shared by training, search, and inference."""
from __future__ import annotations

import math
import numpy as np

NFULL = 221 * 51
COARSE_CAP_FRACTION = 0.10
BUDGETS = (0.375, 0.5, 0.75)


def coarse_count(nfull: int, fraction: float = .10, mandatory_count: int = 0) -> int:
    if nfull < 1 or not math.isfinite(fraction) or not 0 < fraction <= COARSE_CAP_FRACTION:
        raise ValueError("Airfoil coarse fraction must be in (0, 0.10]")
    count = math.floor(nfull * fraction)
    if count < max(2, mandatory_count):
        raise ValueError("Airfoil coarse budget cannot contain mandatory points and two samples")
    return count


def validate_coarse_count(count: int, nfull: int = NFULL) -> None:
    if not 2 <= count <= math.floor(COARSE_CAP_FRACTION * nfull):
        raise ValueError(f"Airfoil coarse input {count} exceeds the 10% cap or is too small")


def validate_coarse_indices(indices, nfull: int, mandatory=()) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("coarse indices must be a one-dimensional integer array")
    if len(np.unique(values)) != len(values):
        raise ValueError("coarse indices must not contain duplicates")
    if np.any(values < 0) or np.any(values >= nfull):
        raise ValueError("coarse index outside full grid")
    validate_coarse_count(len(values), nfull)
    if not np.isin(mandatory, values).all():
        raise ValueError("coarse set must contain all mandatory surface points")
    return values.astype(np.int64, copy=False)


def sample_rng(sample_id: int, seed: int) -> np.random.Generator:
    # Preserve the historical official-test stream, independently of filtering.
    index = sample_id - 1000 if 1000 <= sample_id < 1200 else sample_id
    return np.random.default_rng(1000 * seed + index)
