from __future__ import annotations

import numpy as np

from adasolver.inference.base import SamplingResult


def sample_uniform(full_count: int, count: int, rng: np.random.Generator) -> SamplingResult:
    if count > full_count:
        raise ValueError("sample count exceeds full point count")
    indices = rng.choice(full_count, count, replace=False)
    result = SamplingResult(indices, np.full(count, 1.0 / full_count), np.ones(count),
                            {"method": "uniform"})
    result.validate(full_count, count)
    return result
