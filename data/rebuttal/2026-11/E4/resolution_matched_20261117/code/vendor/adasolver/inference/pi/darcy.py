from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult


def coefficient_interface_mask(
    coefficients: np.ndarray, shape: tuple[int, int] = (85, 85),
    threshold: float = 1e-6,
) -> np.ndarray:
    """Return nodes adjacent to a coefficient discontinuity on the native grid."""
    values = np.asarray(coefficients, dtype=np.float64).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    interface = np.zeros(shape, dtype=bool)
    interface[1:, :] |= np.abs(values[1:, :] - values[:-1, :]) > threshold
    interface[:-1, :] |= np.abs(values[1:, :] - values[:-1, :]) > threshold
    interface[:, 1:] |= np.abs(values[:, 1:] - values[:, :-1]) > threshold
    interface[:, :-1] |= np.abs(values[:, 1:] - values[:, :-1]) > threshold
    return interface.reshape(-1)


def coefficient_interface_distance(
    coefficients: np.ndarray, shape: tuple[int, int] = (85, 85),
    threshold: float = 1e-6,
) -> np.ndarray:
    """Distance to the nearest coefficient interface in normalized grid units."""
    mask = coefficient_interface_mask(coefficients, shape=shape, threshold=threshold)
    if not mask.any():
        return np.ones(mask.size, dtype=np.float64)
    # EDT returns zero on interface nodes and cell-distance elsewhere.
    return distance_transform_edt(~mask).reshape(-1) / max(shape[0] - 1, 1)


def coefficient_interface_score(
    coefficients: np.ndarray, reference_coords: np.ndarray, k: int = 8,
) -> np.ndarray:
    """Estimate local coefficient-interface contrast from coordinate neighbors."""
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    coords = np.asarray(reference_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(values) != len(coords):
        raise ValueError("coefficients and reference_coords must align as (N,) and (N, 2)")
    if not np.isfinite(values).all() or not np.isfinite(coords).all():
        raise ValueError("coefficients and reference_coords must be finite")
    if k < 1 or len(coords) < 2:
        raise ValueError("k must be positive and at least two points are required")
    distances, neighbors = cKDTree(coords).query(coords, k=min(k + 1, len(coords)))
    del distances
    score = np.zeros(len(coords), dtype=np.float64)
    for column in range(1, neighbors.shape[1]):
        scale = np.maximum(np.abs(values) + np.abs(values[neighbors[:, column]]), 1e-12)
        score += np.abs(values - values[neighbors[:, column]]) / scale
    return score / max(neighbors.shape[1] - 1, 1)


def _robust_unit(values: np.ndarray) -> np.ndarray:
    """Map a field to a nonnegative, robustly standardized signal."""
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-12)
    return np.maximum((values - center) / scale, 0.0)


def coefficient_feature_score(
    coefficients: np.ndarray,
    reference_coords: np.ndarray | None = None,
    *,
    shape: tuple[int, int] = (85, 85),
    scales: tuple[int, ...] = (1, 2, 4, 8),
    feature: str = "multiscale",
    threshold: float = 1e-6,
    log_transform: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a coefficient-only, multiscale structural score.

    The score deliberately uses no pressure prediction or target.  It combines
    local log-coefficient jumps with smoothed interface occupancy, which keeps
    scattered interfaces from becoming a thin, over-weighted shell.
    """
    values = np.asarray(coefficients, dtype=np.float64).reshape(shape)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("coefficients must be finite and positive")
    if feature not in {"multiscale", "jump", "occupancy"}:
        raise ValueError(f"unsupported Darcy coefficient feature: {feature}")
    if not scales or any(int(scale) < 1 for scale in scales):
        raise ValueError("scales must contain positive integers")
    field = np.log(np.maximum(values, 1e-12)) if log_transform else values
    jump = np.zeros_like(field)
    jump[1:, :] = np.maximum(jump[1:, :], np.abs(field[1:, :] - field[:-1, :]))
    jump[:-1, :] = np.maximum(jump[:-1, :], np.abs(field[1:, :] - field[:-1, :]))
    jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(field[:, 1:] - field[:, :-1]))
    jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(field[:, 1:] - field[:, :-1]))
    interface = coefficient_interface_mask(coefficients, shape=shape, threshold=threshold).reshape(shape)
    components: list[np.ndarray] = []
    names: list[str] = []
    if feature in {"multiscale", "jump"}:
        components.append(_robust_unit(jump))
        names.append("robust_log_jump")
    if feature in {"multiscale", "occupancy"}:
        for scale in scales:
            occupancy = gaussian_filter(interface.astype(np.float64), sigma=float(scale), mode="nearest")
            components.append(_robust_unit(occupancy))
            names.append(f"interface_occupancy_sigma_{scale}")
    score = np.mean(np.stack(components, axis=0), axis=0).reshape(-1)
    metadata = {
        "feature": feature, "feature_names": names, "scales": list(scales),
        "threshold": threshold, "log_transform": log_transform,
        "feature_score_mean": float(score.mean()),
    }
    return score, metadata


def _weight_from_prior(
    coefficients: np.ndarray, reference_coords: np.ndarray, *, prior: str,
    amp: float, ell: float, k: int, shape: tuple[int, int], score_power: float,
    feature: str, scales: tuple[int, ...], feature_rho: float, feature_cap: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if prior == "interface_distance":
        distance = coefficient_interface_distance(coefficients, shape=shape)
        weight = 1.0 + amp * np.exp(-distance / ell)
        metadata = {"prior": prior, "ell": ell, "interface_fraction": float(np.mean(distance == 0.0))}
    elif prior == "interface_contrast":
        score = coefficient_interface_score(coefficients, reference_coords, k=k)
        normalized = score / max(float(score.mean()), 1e-12)
        weight = 1.0 + amp * np.power(normalized, score_power)
        metadata = {"prior": prior, "neighbors": k, "score_power": score_power}
    elif prior == "feature_mixture":
        if not 0.0 <= feature_rho <= 1.0 or feature_cap < 1.0:
            raise ValueError("feature_rho must be in [0, 1] and feature_cap >= 1")
        score, feature_metadata = coefficient_feature_score(
            coefficients, reference_coords, shape=shape, scales=scales, feature=feature,
        )
        if float(score.mean()) <= 1e-12:
            structural = np.ones_like(score)
        else:
            normalized = score / float(score.mean())
            structural = np.clip(normalized, 0.0, feature_cap)
            structural /= max(float(structural.mean()), 1e-12)
        weight = (1.0 - feature_rho) + feature_rho * structural
        weight = np.clip(weight, 1.0 / feature_cap, feature_cap)
        metadata = {"prior": prior, "feature_rho": feature_rho,
                    "feature_cap": feature_cap, **feature_metadata}
    elif prior == "uniform":
        weight = np.ones(len(reference_coords), dtype=np.float64)
        metadata = {"prior": prior}
    else:
        raise ValueError(f"unsupported Darcy PI prior: {prior}")
    return weight, metadata


def sample_darcy_pi(
    coefficients: np.ndarray, reference_coords: np.ndarray, count: int,
    rng: np.random.Generator, amp: float = 8.0, beta: float = 0.75,
    k: int = 8, *, prior: str = "interface_contrast", ell: float = 0.05,
    shape: tuple[int, int] = (85, 85), score_power: float = 1.0,
    feature: str = "multiscale", scales: tuple[int, ...] = (1, 2, 4, 8),
    feature_rho: float = 0.25, feature_cap: float = 2.0,
) -> SamplingResult:
    """Sample a truth-free Darcy density with an explicit, tunable prior."""
    coefficients = np.asarray(coefficients).reshape(-1)
    reference_coords = np.asarray(reference_coords)
    full_count = len(reference_coords)
    if count < 1 or count > full_count or amp < 0 or beta < 0 or ell <= 0:
        raise ValueError("invalid Darcy PI parameters or budget")
    weight, metadata = _weight_from_prior(
        coefficients, reference_coords, prior=prior, amp=amp, ell=ell,
        k=k, shape=shape, score_power=score_power, feature=feature,
        scales=scales, feature_rho=feature_rho, feature_cap=feature_cap,
    )
    probability = weight / weight.sum()
    indices = rng.choice(full_count, count, replace=False, p=probability)
    density = probability[indices]
    # Relative density is sufficient for the compensated attention integral.
    relative_weight = weight[indices]
    compensation = np.power(1.0 / relative_weight, beta)
    result = SamplingResult(
        indices.astype(np.int64), density, compensation.astype(np.float32),
        {"method": "pi", "amp": amp, "beta": beta, "truth_access": False, **metadata},
    )
    result.validate(full_count, count)
    return result
