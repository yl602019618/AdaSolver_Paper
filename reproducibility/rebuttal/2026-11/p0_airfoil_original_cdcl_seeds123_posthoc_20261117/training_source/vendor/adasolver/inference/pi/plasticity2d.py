from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult


def plasticity_contact_indices(shape: tuple[int, int] = (101, 31)) -> np.ndarray:
    return np.arange(shape[0], dtype=np.int64) * shape[1]


def contact_distance(shape: tuple[int, int] = (101, 31)) -> np.ndarray:
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    # normalized distance to the j=0 contact row
    return j.reshape(-1).astype(np.float64) / max(shape[1] - 1, 1)


def sample_plasticity_pi(count: int, rng: np.random.Generator,
                         shape: tuple[int, int] = (101, 31), amp: float = 4.0,
                         ell: float = 0.1, beta: float = 0.5,
                         mandatory_indices: np.ndarray | None = None) -> SamplingResult:
    full_count = shape[0] * shape[1]
    mandatory = plasticity_contact_indices(shape) if mandatory_indices is None else np.asarray(mandatory_indices, dtype=np.int64)
    if count < len(mandatory) or count > full_count or amp < 0 or ell <= 0:
        raise ValueError("invalid Plasticity PI parameters or budget")
    distance = contact_distance(shape)
    weight = 1.0 + amp * np.exp(-distance / ell)
    available = np.setdiff1d(np.arange(full_count, dtype=np.int64), mandatory)
    probabilities = weight[available] / weight[available].sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    # This is the historical position-weight approximation; expose it explicitly.
    density = weight[indices] / weight.sum()
    compensation = (1.0 / weight[indices]) ** beta
    result = SamplingResult(indices, density, compensation, {
        "method": "pi", "prior": "contact_distance", "amp": amp,
        "ell": ell, "beta": beta, "mandatory_count": int(len(mandatory)),
        "truth_access": False,
    })
    result.validate(full_count, count)
    return result
