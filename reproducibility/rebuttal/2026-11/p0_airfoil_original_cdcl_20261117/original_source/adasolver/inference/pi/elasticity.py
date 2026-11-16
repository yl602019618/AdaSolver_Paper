from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult


def cavity_boundary_distance(coords: np.ndarray, grid_n: int = 64,
                             gap_mult: float = 1.8) -> np.ndarray:
    """Approximate distance to the internal cavity using a per-sample void grid."""
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"expected non-empty coordinates with shape (N, 2), got {points.shape}")
    if grid_n < 2 or gap_mult <= 0:
        raise ValueError("grid_n must be >= 2 and gap_mult must be positive")
    tree = cKDTree(points)
    gx, gy = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_n),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_n),
    )
    grid = np.column_stack((gx.reshape(-1), gy.reshape(-1)))
    gaps = tree.query(grid, k=1)[0]
    void = grid[gaps > gap_mult * np.median(gaps)]
    if len(void) == 0:
        center = grid[int(np.argmax(gaps))]
        return np.linalg.norm(points - center, axis=1)
    return cKDTree(void).query(points, k=1)[0]


def sample_elasticity_pi(coords: np.ndarray, count: int, rng: np.random.Generator,
                         amp: float = 64.0, ell: float = 0.012, beta: float = 0.5,
                         grid_n: int = 64, gap_mult: float = 1.8,
                         correction: str = "proposal") -> SamplingResult:
    """Sample the cavity prior with an explicit correction convention.

    ``proposal`` retains the published inverse-proposal heuristic. It is not
    inverse inclusion probability for weighted sampling without replacement.
    ``reference_voronoi`` uses the selected set's reference-cell masses.
    """
    points = np.asarray(coords, dtype=np.float64)
    n = len(points)
    if count < 1 or count > n or amp < 0 or ell <= 0 or beta < 0:
        raise ValueError("invalid Elasticity PI parameters or budget")
    if correction not in {"proposal", "reference_voronoi"}:
        raise ValueError("correction must be proposal or reference_voronoi")
    distance = cavity_boundary_distance(points, grid_n, gap_mult)
    weight = 1.0 + amp * np.exp(-distance / ell)
    probabilities = weight / weight.sum()
    indices = rng.choice(n, count, replace=False, p=probabilities)
    density = weight[indices] / weight.sum()
    compensation = np.power(1.0 / weight[indices], beta)
    metadata = {
        "method": "pi", "kind": "elasticity_cavity_boundary_void_grid",
        "amp": amp, "ell": ell, "beta": beta, "grid_n": grid_n, "gap_mult": gap_mult,
    }
    if correction == "reference_voronoi":
        from adasolver.inference.finite_measure import reference_cell_weights
        cell_correction = reference_cell_weights(points, indices)
        density = 1.0 / cell_correction
        compensation = np.power(cell_correction, beta)
        metadata.update(compensation_kind="reference_voronoi",
                        density_semantics="inverse represented reference-cell mass, not inclusion probability")
    result = SamplingResult(
        indices.astype(np.int64), density, compensation.astype(np.float32),
        metadata,
    )
    result.validate(n, count)
    return result


def sample_elasticity_pi_voronoi(coords: np.ndarray, count: int,
                                 rng: np.random.Generator, amp: float = 64.0,
                                 ell: float = 0.012, beta: float = 1.0,
                                 grid_n: int = 64, gap_mult: float = 1.8) -> SamplingResult:
    """Keep the historical PI points and use their reference-cell masses.

    This is deterministic quadrature, not exact inclusion-probability
    weighting. At beta=1 the represented reference mass is conserved.
    The original entry point retains the published proposal-weight behavior.
    """
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    return sample_elasticity_pi(coords, count, rng, amp, ell, beta, grid_n, gap_mult,
                                correction="reference_voronoi")
