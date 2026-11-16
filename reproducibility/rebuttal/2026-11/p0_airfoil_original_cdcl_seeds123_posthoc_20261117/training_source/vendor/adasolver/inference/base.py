from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamplingResult:
    indices: np.ndarray
    sampling_density: np.ndarray
    compensation: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, full_count: int, expected_count: int) -> None:
        idx = np.asarray(self.indices)
        density = np.asarray(self.sampling_density)
        compensation = np.asarray(self.compensation)
        if idx.ndim != 1 or len(idx) != expected_count:
            raise ValueError(f"expected {expected_count} sampled indices, got {idx.shape}")
        if len(np.unique(idx)) != expected_count:
            raise ValueError("sampled indices must be unique")
        if np.any(idx < 0) or np.any(idx >= full_count):
            raise ValueError("sampled index outside full point set")
        if density.shape != (expected_count,) or compensation.shape != (expected_count,):
            raise ValueError("density and compensation must align with sampled indices")
        if not np.all(np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("sampling density must be positive and finite")
        if not np.all(np.isfinite(compensation)) or np.any(compensation <= 0):
            raise ValueError("compensation must be positive and finite")
