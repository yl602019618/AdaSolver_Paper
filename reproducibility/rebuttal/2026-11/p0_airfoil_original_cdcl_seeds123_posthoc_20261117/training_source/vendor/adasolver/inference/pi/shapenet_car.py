"""Geometry-only PI sampling for ShapeNet-Car."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult


def car_surface_distance(coords: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    mask = np.asarray(surface_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3 or mask.shape != (len(points),):
        raise ValueError("coords must be (N,3) and surface_mask must be (N,)")
    if not mask.any():
        raise ValueError("surface_mask contains no surface points")
    return cKDTree(points[mask]).query(points, k=1)[0]


def sample_car_pi(coords: np.ndarray, surface_mask: np.ndarray, count: int,
                  rng: np.random.Generator, amp: float = 8.0, ell: float = 0.05,
                  beta: float = 1.0) -> SamplingResult:
    points = np.asarray(coords)
    mask = np.asarray(surface_mask, dtype=bool)
    n = len(points)
    if count < 1 or count > n or amp < 0 or ell <= 0 or beta < 0:
        raise ValueError("invalid ShapeNet-Car PI parameters or budget")
    weight = 1.0 + amp * np.exp(-car_surface_distance(points, mask) / ell)
    surface = np.flatnonzero(mask)
    mandatory = np.array([surface[np.argmax(weight[surface])]], dtype=np.int64)
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory)
    probabilities = weight[available] / weight[available].sum()
    extra = rng.choice(available, count - 1, replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    compensation = np.power(1.0 / weight[indices], beta).astype(np.float32)
    result = SamplingResult(indices, weight[indices] / weight.sum(), compensation,
                            {"method": "pi", "strategy": "surface_distance_exponential",
                             "amp": amp, "ell": ell, "beta": beta,
                             "mandatory_surface_count": 1})
    result.validate(n, count)
    return result


def sample_car_pi_exponential(coords: np.ndarray, surface_mask: np.ndarray, count: int,
                              rng: np.random.Generator, amp: float = 8.0,
                              ell: float = 0.05, beta: float = 1.0,
                              mandatory_surface: bool = True) -> SamplingResult:
    """Historical single-stage exponential near-surface prior.

    The density is ``w=1+amp*exp(-d/ell)`` and sampling is weighted without
    replacement.  ``beta`` controls the inverse-density compensation applied
    to model attention.  This explicit entry point preserves the historical
    protocol while keeping :func:`sample_car_pi` as its compatibility alias.
    """
    points = np.asarray(coords)
    mask = np.asarray(surface_mask, dtype=bool)
    n = len(points)
    if count < 1 or count > n or amp < 0 or ell <= 0 or beta < 0:
        raise ValueError("invalid ShapeNet-Car exponential parameters or budget")
    distance = car_surface_distance(points, mask)
    weight = 1.0 + amp * np.exp(-distance / ell)
    mandatory = np.array([], dtype=np.int64)
    if mandatory_surface:
        surface = np.flatnonzero(mask)
        mandatory = np.array([surface[np.argmax(weight[surface])]], dtype=np.int64)
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory)
    probabilities = weight[available] / weight[available].sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    compensation = np.power(1.0 / weight[indices], beta).astype(np.float32)
    result = SamplingResult(indices, weight[indices] / weight.sum(), compensation,
                            {"method": "pi", "strategy": "historical_exponential",
                             "amp": float(amp), "ell": float(ell), "beta": float(beta),
                             "mandatory_surface_count": int(len(mandatory))})
    result.validate(n, count)
    return result


def sample_car_pi_two_stage(coords: np.ndarray, surface_mask: np.ndarray, count: int,
                            rng: np.random.Generator, r: float = 0.1,
                            frac_extra: float = 0.5, beta: float = 0.75) -> SamplingResult:
    """Historical two-stage sampler: uniform base plus a near-surface stage.

    ``frac_extra`` is the fraction of the total budget allocated to the second
    stage; that stage samples uniformly from points with distance ``< r``.
    Inverse-density compensation follows the historical shell-mass formula.
    """
    points = np.asarray(coords)
    mask = np.asarray(surface_mask, dtype=bool)
    n = len(points)
    if count < 1 or count > n or r <= 0 or not 0 <= frac_extra <= 1 or beta < 0:
        raise ValueError("invalid ShapeNet-Car two-stage parameters or budget")
    distance = car_surface_distance(points, mask)
    n_extra = int(round(count * frac_extra))
    n_base = count - n_extra
    permutation = rng.permutation(n)
    base = permutation[:n_base].astype(np.int64, copy=False)
    base_set = set(base.tolist())
    shell = np.flatnonzero(distance < r).astype(np.int64)
    shell_available = np.asarray([i for i in shell if i not in base_set], dtype=np.int64)
    if len(shell_available) >= n_extra:
        extra = rng.choice(shell_available, n_extra, replace=False)
        fallback_count = 0
    else:
        fallback_count = n_extra - len(shell_available)
        rest = np.asarray([i for i in range(n) if i not in base_set and distance[i] >= r], dtype=np.int64)
        if fallback_count > len(rest):
            raise ValueError("two-stage sampler cannot satisfy requested count")
        # Historical fallback takes the nearest available non-shell points.
        extra = np.concatenate((shell_available, rest[np.argsort(distance[rest])[:fallback_count]]))
    indices = np.concatenate((base, extra.astype(np.int64, copy=False)))
    shell_fraction = max(len(shell) / n, 1e-12)
    base_fraction = n_base / n
    extra_fraction = n_extra / n
    shell_weight = (base_fraction + extra_fraction / shell_fraction) / max(base_fraction, 1e-12)
    weights = np.ones(count, dtype=np.float64)
    near = distance[indices] < r
    weights[near] = shell_weight
    if fallback_count:
        weights[n_base + len(shell_available):] = shell_weight
    compensation = np.power(1.0 / weights, beta).astype(np.float32)
    result = SamplingResult(indices, weights / weights.sum(), compensation,
                            {"method": "pi", "strategy": "historical_two_stage", "r": float(r),
                             "frac_extra": float(frac_extra), "beta": float(beta),
                             "base_count": int(n_base), "extra_count": int(n_extra),
                             "shell_count": int(len(shell)), "fallback_count": int(fallback_count)})
    result.validate(n, count)
    return result


def sample_car_pi_two_stage_exact(coords: np.ndarray, surface_mask: np.ndarray,
                                  count: int, rng: np.random.Generator,
                                  r: float = 0.1, frac_extra: float = 0.5,
                                  beta: float = 1.0) -> SamplingResult:
    """Two-stage PI with a fixed outside priority and exact marginal weights.

    Unlike the historical shell-mass formula, probabilities include shell
    saturation and deterministic nearest-outside additions. The uniform base
    must be nonempty, so every reference point has positive probability.
    Equal outside distances are ordered by original point ID before filtering
    the random base. The legacy entry point remains available for replay.
    """
    from adasolver.inference.finite_measure import two_stage_probabilities

    points = np.asarray(coords)
    n = len(points)
    if (not 1 <= count <= n or not np.isfinite(r) or r <= 0
            or not np.isfinite(frac_extra) or not 0 <= frac_extra <= 1
            or not np.isfinite(beta) or beta < 0):
        raise ValueError("invalid exact two-stage parameters or budget")
    distance = car_surface_distance(points, surface_mask)
    n_extra = int(round(count * frac_extra))
    n_base = count - n_extra
    if n_base < 1:
        raise ValueError("exact correction requires a nonempty uniform base")
    base = rng.permutation(n)[:n_base].astype(np.int64, copy=False)
    available = np.ones(n, dtype=bool)
    available[base] = False
    shell = np.flatnonzero(distance < r)
    shell_left = shell[available[shell]]
    fallback = max(0, n_extra - len(shell_left))
    if fallback:
        outside = np.flatnonzero(distance >= r)
        outside = outside[np.lexsort((outside, distance[outside]))]
        extra = np.concatenate((shell_left, outside[available[outside]][:fallback]))
    else:
        extra = rng.choice(shell_left, n_extra, replace=False)
    indices = np.concatenate((base, extra))
    inclusion = two_stage_probabilities(distance, r, n_base, n_extra)
    relative_density = n * inclusion[indices] / count
    result = SamplingResult(
        indices, relative_density, np.power(1.0 / relative_density, beta).astype(np.float32),
        {"method": "pi", "strategy": "two_stage_exact_inclusion",
         "r": float(r), "frac_extra": float(frac_extra), "beta": float(beta),
         "base_count": n_base, "extra_count": n_extra, "shell_count": len(shell),
         "fallback_count": fallback, "outside_priority": "distance_then_original_id",
         "compensation_kind": "exact_marginal_inclusion", "expected_count": float(inclusion.sum())},
    )
    result.validate(n, count)
    return result


def sample_car_pi_two_stage_tempered(coords: np.ndarray, surface_mask: np.ndarray,
                                     count: int, rng: np.random.Generator,
                                     r: float = 0.1, frac_extra: float = 0.5,
                                     alpha: float = 1.0, beta: float = 1.0) -> SamplingResult:
    """Temper the proposal before sampling, then correct its finite design.

    Alpha controls allocation. Beta=1 uses complete inclusion correction.
    No physical-accuracy optimality follows from choosing either exponent.
    """
    from adasolver.inference.finite_measure import temper_shell_fraction

    distance = car_surface_distance(coords, surface_mask)
    shell_fraction = float(np.mean(distance < r))
    fraction = temper_shell_fraction(shell_fraction, frac_extra, alpha)
    result = sample_car_pi_two_stage_exact(coords, surface_mask, count, rng, r, fraction, beta)
    return SamplingResult(
        result.indices, result.sampling_density, result.compensation,
        {**result.metadata, "strategy": "two_stage_tempered_exact",
         "proposal_alpha": float(alpha), "original_frac_extra": float(frac_extra)},
    )
