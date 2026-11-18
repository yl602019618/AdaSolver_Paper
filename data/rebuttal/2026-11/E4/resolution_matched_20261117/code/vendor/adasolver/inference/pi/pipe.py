from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult


def pipe_wall_distance(coords: np.ndarray, shape: tuple[int, int] = (129, 129)) -> np.ndarray:
    coords = np.asarray(coords)
    if coords.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"unexpected Pipe coordinate shape: {coords.shape}")
    point_ids = np.arange(len(coords))
    j = point_ids % shape[1]
    wall = (j == 0) | (j == shape[1] - 1)
    return cKDTree(coords[wall]).query(coords, k=1)[0]


def sample_pipe_pi(distance: np.ndarray, count: int, rng: np.random.Generator,
                   wall_band_fraction: float = 0.05, refinement_fraction: float = 0.25,
                   beta: float = 0.75) -> SamplingResult:
    full_count = len(distance)
    extra_count = int(round(count * refinement_fraction))
    base_count = count - extra_count
    permutation = rng.permutation(full_count)
    base = permutation[:base_count]
    available = np.ones(full_count, dtype=bool)
    available[base] = False
    shell = np.flatnonzero((distance < wall_band_fraction) & available)
    if len(shell) >= extra_count:
        extra = rng.choice(shell, extra_count, replace=False)
        supplemented = 0
    else:
        needed = extra_count - len(shell)
        remainder = np.flatnonzero(available & (distance >= wall_band_fraction))
        extra = np.concatenate((shell, remainder[np.argsort(distance[remainder])[:needed]]))
        supplemented = needed
    indices = np.concatenate((base, extra))
    shell_fraction = max(float(np.mean(distance < wall_band_fraction)), 1e-12)
    base_rate = base_count / full_count
    extra_rate = extra_count / full_count
    density_ratio = np.ones(count, dtype=np.float64)
    wall_ratio = (base_rate + extra_rate / shell_fraction) / max(base_rate, 1e-12)
    density_ratio[distance[indices] < wall_band_fraction] = wall_ratio
    if supplemented:
        density_ratio[base_count + len(shell):] = wall_ratio
    compensation = np.power(1.0 / density_ratio, beta)
    result = SamplingResult(indices, density_ratio, compensation.astype(np.float32), {
        "method": "pi", "kind": "two_stage", "wall_band_fraction": wall_band_fraction,
        "refinement_fraction": refinement_fraction, "beta": beta,
        "supplemented_points": supplemented,
    })
    result.validate(full_count, count)
    return result
