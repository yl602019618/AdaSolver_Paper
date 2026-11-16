from __future__ import annotations

import numpy as np
import time
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult
from adasolver.inference.pi.airfoil2d import surface_distance, voronoi_mass
from adasolver.inference.uniform.airfoil2d import sample_airfoil_uniform
from adasolver.inference.airfoil2d_contract import validate_coarse_indices


def sample_airfoil_cpg(coords: np.ndarray, count: int, rng: np.random.Generator,
                       score: np.ndarray | None = None,
                       mandatory_indices: np.ndarray | None = None) -> SamplingResult:
    """Deterministic score-guided Airfoil refinement hook.

    Acquisition is supplied by the caller after a coarse operator pass; this
    function only turns scores into a contract-valid final subset.
    """
    n = len(coords)
    mandatory = np.arange(50, 171, dtype=np.int64) * 51 if mandatory_indices is None else np.asarray(mandatory_indices)
    if score is None:
        return sample_airfoil_uniform(n, count, mandatory, rng)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError("CPG score must be finite and aligned with all points")
    available = np.setdiff1d(np.arange(n), mandatory)
    available_score = np.maximum(values[available], 0.0)
    if available_score.max() > 0:
        available_score = available_score / available_score.max()
    probabilities = 0.5 + available_score
    probabilities = probabilities / probabilities.sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    point_density = np.full(n, 0.5, dtype=np.float64)
    point_density[available] += available_score
    density = point_density[indices] / point_density.sum()
    compensation = np.power(point_density[indices], -0.75)
    compensation /= compensation.mean()
    result = SamplingResult(indices, density, compensation,
                            {"method": "cpg", "mandatory_count": int(len(mandatory)),
                             "placement": "score_weighted_without_replacement",
                             "uniform_floor": 0.5, "beta": 0.75})
    result.validate(n, count)
    return result


def _q95_normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    scale = float(np.quantile(values, 0.95))
    if scale <= 1e-12:
        return np.ones_like(values)
    return values / scale


def _surface_force_coefficients(coords: np.ndarray, mandatory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    surface = np.asarray(coords, dtype=np.float64)[mandatory]
    edge_d = surface[:-1, 1] - surface[1:, 1]
    edge_l = surface[1:, 0] - surface[:-1, 0]
    c_drag = np.zeros(len(mandatory), dtype=np.float64)
    c_lift = np.zeros(len(mandatory), dtype=np.float64)
    c_drag[:-1] += edge_d / 2.0
    c_drag[1:] += edge_d / 2.0
    c_lift[:-1] += edge_l / 2.0
    c_lift[1:] += edge_l / 2.0
    return c_drag, c_lift


def _diffuse_surface_feature(
    points: np.ndarray,
    surface_points: np.ndarray,
    feature: np.ndarray,
    ell: float,
) -> np.ndarray:
    kernel = np.exp(
        -np.linalg.norm(points[:, None, :] - surface_points[None, :, :], axis=2) / ell
    )
    return np.max(kernel * _q95_normalize(feature)[None, :], axis=1)


def sample_airfoil_cpg_force_ratio(
    coords: np.ndarray,
    coarse_indices: np.ndarray,
    coarse_pressure: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mandatory_indices: np.ndarray,
    ell: float = 0.1,
    geometry_weight: float = 0.5,
    uniform_floor: float = 0.1,
    score_gamma: float = 0.5,
    beta: float = 1.0,
    shape: tuple[int, int] = (221, 51),
) -> SamplingResult:
    """Sample using coarse pressure sensitivity of the raw surface Dp/Lp ratio.

    The coarse operator prediction is the only field-dependent input. Truth is
    intentionally absent from this deployable sampler boundary.
    """
    score_started = time.perf_counter()
    points = np.asarray(coords, dtype=np.float64)
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
    pressure = np.asarray(coarse_pressure, dtype=np.float64).reshape(-1)
    mandatory = np.asarray(mandatory_indices, dtype=np.int64).reshape(-1)
    n = len(points)
    if points.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"expected coordinates {(shape[0] * shape[1], 2)}, got {points.shape}")
    validate_coarse_indices(coarse_indices, n, mandatory)
    if len(coarse_indices) != len(pressure) or len(np.unique(coarse_indices)) != len(coarse_indices):
        raise ValueError("coarse indices and pressure must be aligned and unique")
    if not np.all(np.isfinite(pressure)) or ell <= 0 or geometry_weight < 0 or uniform_floor <= 0 or score_gamma <= 0 or beta < 0:
        raise ValueError("invalid force-ratio CPG parameters")
    if count < len(mandatory) or count > n or np.any(mandatory < 0) or np.any(mandatory >= n):
        raise ValueError("invalid mandatory indices or budget")
    nearest = cKDTree(points[coarse_indices]).query(points, k=1)[1]
    coarse_full_pressure = pressure[nearest]
    c_drag, c_lift = _surface_force_coefficients(points, mandatory)
    surface_pressure = coarse_full_pressure[mandatory]
    drag = float(np.dot(c_drag, surface_pressure))
    lift = float(np.dot(c_lift, surface_pressure))
    if abs(lift) <= 1e-12:
        ratio_sensitivity = np.ones(len(mandatory), dtype=np.float64)
    else:
        ratio_sensitivity = np.abs((lift * c_drag - drag * c_lift) / (lift * lift))
    surface_points = points[mandatory]
    surface_feature = _diffuse_surface_feature(points, surface_points, ratio_sensitivity, ell)
    geometry = np.exp(-surface_distance(points, shape=shape, surface_indices=mandatory) / ell)
    score = uniform_floor + surface_feature + geometry_weight * geometry
    score_seconds = time.perf_counter() - score_started
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory)
    probability = np.power(np.maximum(score[available], 1e-12), score_gamma)
    probability /= probability.sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probability)
    indices = np.concatenate((mandatory, extra))
    mass = voronoi_mass(indices, shape)
    realized_density = mass.mean() / mass
    compensation = np.power(realized_density, -beta)
    result = SamplingResult(
        indices, score[indices] / score.sum(), compensation,
        {"method": "cpg_force_ratio", "coarse_count": int(len(coarse_indices)),
         "feature_seconds": score_seconds,
         "coarse_fraction_realized": float(len(coarse_indices) / n),
         "mandatory_count": int(len(mandatory)), "feature": "ratio_sensitivity",
         "normalization": "q95", "surface_kernel_ell": ell,
         "geometry_weight": geometry_weight, "uniform_floor": uniform_floor,
         "score_gamma": score_gamma, "voronoi_beta": beta,
         "compensation_type": "final_subset_voronoi_mass_inverse_power",
         "coarse_predicted_ratio_drag": drag, "coarse_predicted_ratio_lift": lift},
    )
    result.validate(n, count)
    return result


def sample_airfoil_cpg_lift_support(
    coords: np.ndarray,
    coarse_indices: np.ndarray,
    coarse_pressure: np.ndarray,
    count: int,
    rng: np.random.Generator,
    mandatory_indices: np.ndarray,
    ell: float = 0.1,
    ratio_weight: float = 1.0,
    lift_support_weight: float = 1.0,
    lift_balance_weight: float = 0.5,
    geometry_weight: float = 0.5,
    uniform_floor: float = 0.1,
    score_gamma: float = 0.5,
    lift_reliability_scale: float = 0.02,
    beta: float = 1.0,
    shape: tuple[int, int] = (221, 51),
) -> SamplingResult:
    """Sample for raw Dp/Lp while protecting the lift denominator."""
    score_started = time.perf_counter()
    points = np.asarray(coords, dtype=np.float64)
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
    pressure = np.asarray(coarse_pressure, dtype=np.float64).reshape(-1)
    mandatory = np.asarray(mandatory_indices, dtype=np.int64).reshape(-1)
    n = len(points)
    values = (ratio_weight, lift_support_weight, lift_balance_weight,
              geometry_weight, uniform_floor, score_gamma,
              lift_reliability_scale, beta)
    if points.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"expected coordinates {(shape[0] * shape[1], 2)}, got {points.shape}")
    validate_coarse_indices(coarse_indices, n, mandatory)
    if len(coarse_indices) != len(pressure) or len(np.unique(coarse_indices)) != len(coarse_indices):
        raise ValueError("coarse indices and pressure must be aligned and unique")
    if not np.all(np.isfinite(pressure)) or ell <= 0 or any(value < 0 for value in values):
        raise ValueError("invalid lift-support CPG parameters")
    if uniform_floor <= 0 or score_gamma <= 0 or lift_reliability_scale <= 0:
        raise ValueError("floor, gamma, and reliability scale must be positive")
    if count < len(mandatory) or count > n or np.any(mandatory < 0) or np.any(mandatory >= n):
        raise ValueError("invalid mandatory indices or budget")

    nearest = cKDTree(points[coarse_indices]).query(points, k=1)[1]
    surface_pressure = pressure[nearest][mandatory]
    c_drag, c_lift = _surface_force_coefficients(points, mandatory)
    drag = float(np.dot(c_drag, surface_pressure))
    lift = float(np.dot(c_lift, surface_pressure))
    lift_reliability = min(1.0, abs(lift) / lift_reliability_scale)
    denominator = max(abs(lift), lift_reliability_scale)
    ratio_sensitivity = np.abs((lift * c_drag - drag * c_lift) / (denominator * denominator))
    lift_support = np.abs(c_lift)
    signed = c_lift * surface_pressure
    lift_balance = _q95_normalize(np.maximum(signed, 0.0)) + _q95_normalize(np.maximum(-signed, 0.0))

    surface_points = points[mandatory]
    ratio_feature = _diffuse_surface_feature(points, surface_points, ratio_sensitivity, ell)
    support_feature = _diffuse_surface_feature(points, surface_points, lift_support, ell)
    balance_feature = _diffuse_surface_feature(points, surface_points, lift_balance, ell)
    geometry = np.exp(-surface_distance(points, shape=shape, surface_indices=mandatory) / ell)
    score = (uniform_floor + ratio_weight * lift_reliability * ratio_feature
             + lift_support_weight * support_feature + lift_balance_weight * balance_feature
             + geometry_weight * geometry)
    score_seconds = time.perf_counter() - score_started
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory)
    probability = np.power(np.maximum(score[available], 1e-12), score_gamma)
    probability /= probability.sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probability)
    indices = np.concatenate((mandatory, extra))
    mass = voronoi_mass(indices, shape)
    realized_density = mass.mean() / mass
    compensation = np.power(realized_density, -beta)
    result = SamplingResult(
        indices, score[indices] / score.sum(), compensation,
        {"method": "cpg_force_ratio_lift_support", "coarse_count": int(len(coarse_indices)),
         "feature_seconds": score_seconds,
         "coarse_fraction_realized": float(len(coarse_indices) / n),
         "mandatory_count": int(len(mandatory)),
         "feature": "ratio_sensitivity_lift_support_balance", "normalization": "q95",
         "surface_kernel_ell": ell, "ratio_weight": ratio_weight,
         "lift_support_weight": lift_support_weight, "lift_balance_weight": lift_balance_weight,
         "geometry_weight": geometry_weight, "uniform_floor": uniform_floor,
         "score_gamma": score_gamma, "lift_reliability_scale": lift_reliability_scale,
         "coarse_lift_reliability": lift_reliability, "voronoi_beta": beta,
         "compensation_type": "final_subset_voronoi_mass_inverse_power",
         "coarse_predicted_ratio_drag": drag, "coarse_predicted_ratio_lift": lift,
         "ratio_feature_max": float(ratio_feature.max()),
         "lift_support_feature_max": float(support_feature.max()),
         "lift_balance_feature_max": float(balance_feature.max())},
    )
    result.validate(n, count)
    return result
