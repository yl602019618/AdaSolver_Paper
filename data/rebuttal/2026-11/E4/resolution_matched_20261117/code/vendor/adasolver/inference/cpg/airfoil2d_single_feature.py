from __future__ import annotations

"""Exploratory single-feature CPG acquisition for Airfoil2D."""

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult
from adasolver.inference.pi.airfoil2d import voronoi_mass, realized_density_from_mass


def coarse_gradient_feature(
    coarse_coords: np.ndarray,
    coarse_pressure: np.ndarray,
    reference_coords: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Lift a local coarse-pressure gradient-energy proxy to the full grid."""
    points = np.asarray(coarse_coords, dtype=np.float64)
    pressure = np.asarray(coarse_pressure, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_coords, dtype=np.float64)
    if points.shape != (len(pressure), 2) or k < 1:
        raise ValueError("coarse coordinates, pressure, and k are invalid")
    kq = min(k + 1, len(points))
    distance, neighbor = cKDTree(points).query(points, k=kq)
    score = np.zeros(len(points), dtype=np.float64)
    for column in range(1, kq):
        delta = pressure - pressure[neighbor[:, column]]
        score += delta * delta / (distance[:, column] ** 2 + 1e-12)
    score /= max(kq - 1, 1)
    nearest = cKDTree(points).query(reference, k=1)[1]
    return np.maximum(score[nearest], 0.0)


def coarse_curvature_feature(
    coarse_coords: np.ndarray,
    coarse_pressure: np.ndarray,
    reference_coords: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Lift an absolute local pressure-Laplacian proxy to the full grid."""
    points = np.asarray(coarse_coords, dtype=np.float64)
    pressure = np.asarray(coarse_pressure, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_coords, dtype=np.float64)
    if points.shape != (len(pressure), 2) or k < 1:
        raise ValueError("coarse coordinates, pressure, and k are invalid")
    kq = min(k + 1, len(points))
    distance, neighbor = cKDTree(points).query(points, k=kq)
    distance = distance[:, 1:]
    neighbor = neighbor[:, 1:]
    scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-12)
    weight = np.exp(-np.square(distance / scale))
    local_mean = np.sum(weight * pressure[neighbor], axis=1) / np.maximum(
        weight.sum(axis=1), 1e-12
    )
    score = np.abs(pressure - local_mean)
    nearest = cKDTree(points).query(reference, k=1)[1]
    return score[nearest]


def coarse_ratio_sensitivity_feature(
    coords: np.ndarray,
    coarse_indices: np.ndarray,
    coarse_pressure: np.ndarray,
    mandatory_indices: np.ndarray,
    ell: float = 0.1,
) -> np.ndarray:
    """Diffuse one scalar field: coarse-predicted local D/L sensitivity."""
    points = np.asarray(coords, dtype=np.float64)
    coarse = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
    pressure = np.asarray(coarse_pressure, dtype=np.float64).reshape(-1)
    mandatory = np.asarray(mandatory_indices, dtype=np.int64).reshape(-1)
    if len(coarse) != len(pressure) or ell <= 0:
        raise ValueError("coarse prediction and ell are invalid")
    nearest = cKDTree(points[coarse]).query(points[mandatory], k=1)[1]
    surface_pressure = pressure[nearest]
    surface = points[mandatory]
    edge_drag = surface[:-1, 1] - surface[1:, 1]
    edge_lift = surface[1:, 0] - surface[:-1, 0]
    c_drag = np.zeros(len(mandatory), dtype=np.float64)
    c_lift = np.zeros(len(mandatory), dtype=np.float64)
    c_drag[:-1] += edge_drag / 2.0
    c_drag[1:] += edge_drag / 2.0
    c_lift[:-1] += edge_lift / 2.0
    c_lift[1:] += edge_lift / 2.0
    drag = float(np.dot(c_drag, surface_pressure))
    lift = float(np.dot(c_lift, surface_pressure))
    if abs(lift) <= 1e-12:
        # A near-zero coarse lift cannot define quotient sensitivity.  Keep the
        # feature neutral rather than smuggling a geometry-only fallback into a
        # pressure-derived acquisition.
        surface_score = np.zeros_like(c_lift)
    else:
        # The omitted global 1/|lift| factor cancels under score normalization.
        surface_score = np.abs(c_drag - (drag / lift) * c_lift)
    scale = max(float(np.quantile(surface_score, 0.95)), 1e-12)
    surface_score = surface_score / scale
    distance = np.linalg.norm(points[:, None, :] - surface[None, :, :], axis=2)
    return np.max(np.exp(-distance / ell) * surface_score[None, :], axis=1)


def normalize_single_feature(score: np.ndarray, mode: str = "q95") -> np.ndarray:
    values = np.maximum(np.asarray(score, dtype=np.float64).reshape(-1), 0.0)
    if not np.all(np.isfinite(values)):
        raise ValueError("single-feature score must be finite")
    if mode == "q95":
        return values / max(float(np.quantile(values, 0.95)), 1e-12)
    if mode == "rank":
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
        return ranks
    if mode == "robust":
        q95 = max(float(np.quantile(values, 0.95)), 1e-12)
        q99 = max(float(np.quantile(values, 0.99)), q95)
        return np.minimum(values, q99) / q95
    raise ValueError(f"unsupported score normalization: {mode}")


def sample_airfoil_cpg_single_feature(
    score: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mandatory_indices: np.ndarray,
    *,
    uniform_floor: float = 0.1,
    score_gamma: float = 0.5,
    beta: float = 1.0,
    normalization: str = "q95",
    shape: tuple[int, int] = (221, 51),
    feature_name: str,
    base_indices: np.ndarray | None = None,
) -> SamplingResult:
    """Sample from exactly one coarse-inference scalar feature."""
    values = normalize_single_feature(score, normalization)
    mandatory = np.asarray(mandatory_indices, dtype=np.int64).reshape(-1)
    n = len(values)
    if n != shape[0] * shape[1] or count < len(mandatory) or count > n:
        raise ValueError("invalid single-feature Airfoil sampling contract")
    if uniform_floor <= 0 or score_gamma <= 0 or beta < 0:
        raise ValueError("floor and gamma must be positive and beta nonnegative")
    base = mandatory if base_indices is None else np.asarray(base_indices, dtype=np.int64)
    if len(np.unique(base)) != len(base) or len(base) > count or not np.isin(mandatory, base).all():
        raise ValueError("final base must be unique, contain mandatory points, and fit budget")
    available = np.setdiff1d(np.arange(n, dtype=np.int64), base)
    probability = np.power(uniform_floor + values[available], score_gamma)
    probability /= probability.sum()
    extra = rng.choice(available, count - len(base), replace=False, p=probability)
    indices = np.concatenate((base, extra))
    mass = voronoi_mass(indices, shape)
    realized_density = realized_density_from_mass(mass)
    compensation = np.power(realized_density, -beta)
    result = SamplingResult(
        indices,
        (uniform_floor + values[indices]) / np.sum(uniform_floor + values),
        compensation,
        {
            "method": "cpg_single_feature",
            "feature": feature_name,
            "normalization": normalization,
            "uniform_floor": uniform_floor,
            "score_gamma": score_gamma,
            "voronoi_beta": beta,
            "truth_access": False,
            "feature_min": float(values.min()),
            "feature_max": float(values.max()),
        },
    )
    result.validate(n, count)
    return result
