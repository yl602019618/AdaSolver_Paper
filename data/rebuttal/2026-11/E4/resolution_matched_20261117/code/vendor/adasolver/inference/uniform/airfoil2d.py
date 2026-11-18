from __future__ import annotations

import numpy as np

from adasolver.inference.base import SamplingResult


def sample_airfoil_uniform(
    full_count: int,
    count: int,
    mandatory_indices: np.ndarray,
    rng: np.random.Generator,
) -> SamplingResult:
    """Sample an Airfoil set while always retaining the complete surface ring."""
    mandatory = np.asarray(mandatory_indices, dtype=np.int64)
    if mandatory.ndim != 1 or len(np.unique(mandatory)) != len(mandatory):
        raise ValueError("mandatory indices must be a one-dimensional unique array")
    if np.any(mandatory < 0) or np.any(mandatory >= full_count):
        raise ValueError("mandatory index outside full point set")
    if count < len(mandatory) or count > full_count:
        raise ValueError("sample count must contain the mandatory surface and fit the full set")
    available = np.setdiff1d(np.arange(full_count, dtype=np.int64), mandatory, assume_unique=True)
    extra = rng.choice(available, count - len(mandatory), replace=False)
    indices = np.concatenate((mandatory, extra))
    # The evaluator's compensation is deliberately neutral for the uniform baseline.
    result = SamplingResult(
        indices=indices,
        sampling_density=np.full(count, 1.0 / full_count, dtype=np.float64),
        compensation=np.ones(count, dtype=np.float64),
        metadata={
            "method": "uniform",
            "mandatory_count": int(len(mandatory)),
            "mandatory_indices": mandatory.tolist(),
        },
    )
    result.validate(full_count, count)
    return result
