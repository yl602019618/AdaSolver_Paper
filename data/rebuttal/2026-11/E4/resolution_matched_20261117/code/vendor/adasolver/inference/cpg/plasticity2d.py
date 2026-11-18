from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult
from adasolver.inference.acquisition import grad_score


def coarse_gradient_density(coarse_pos: np.ndarray, coarse_prediction: np.ndarray,
                            reference: np.ndarray, k: int = 8,
                            amp: float = 8.0) -> np.ndarray:
    """Lift a coarse all-time prediction gradient score to the full point lattice."""
    pos = np.asarray(coarse_pos, dtype=np.float64)
    pred = np.asarray(coarse_prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or pred.ndim != 2 or len(pos) != len(pred):
        raise ValueError("coarse positions/prediction must be aligned (points,2)/(points,channels)")
    if not np.isfinite(pos).all() or not np.isfinite(pred).all():
        raise ValueError("coarse positions and predictions must be finite")
    score = grad_score(pos, pred, k=k)
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("coarse gradient score must be nonnegative and finite")
    nearest = cKDTree(pos).query(ref, k=1)[1]
    lifted = score[nearest]
    normalized = lifted / (lifted.mean() + 1e-12)
    return 1.0 + amp * normalized


def sample_plasticity_cpg(coarse_indices: np.ndarray, density: np.ndarray,
                          count: int, rng: np.random.Generator,
                          beta: float = 1.0) -> SamplingResult:
    """Select a final set from a coarse-gradient density (legacy analogue)."""
    density = np.asarray(density, dtype=np.float64)
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
    full_count = len(density)
    if count <= len(coarse_indices) or count > full_count:
        raise ValueError("CPG count must exceed coarse count and fit the lattice")
    if not np.isfinite(density).all() or np.any(density <= 0) or beta < 0:
        raise ValueError("CPG density must be positive and finite")
    coarse_indices = np.unique(coarse_indices)
    available = np.setdiff1d(np.arange(full_count, dtype=np.int64), coarse_indices)
    probability = density[available] / density[available].sum()
    extra = rng.choice(available, count - len(coarse_indices), replace=False, p=probability)
    indices = np.concatenate((coarse_indices, extra))
    inclusion = len(coarse_indices) / full_count + (count - len(coarse_indices)) * density / density[available].sum()
    inclusion = np.clip(inclusion, 1e-12, None)
    compensation = (1.0 / (inclusion[indices] / (len(coarse_indices) / full_count))) ** beta
    result = SamplingResult(indices, inclusion[indices], compensation, {
        "method": "cpg", "prior": "coarse_gradient_analogue",
        "coarse_count": int(len(coarse_indices)), "beta": beta,
        "truth_access": False,
    })
    result.validate(full_count, count)
    return result
