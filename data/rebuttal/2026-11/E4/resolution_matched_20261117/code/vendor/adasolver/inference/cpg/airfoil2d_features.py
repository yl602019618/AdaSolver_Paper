"""Single scalar acquisitions from decoded, budget-limited pressure."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from adasolver.inference.airfoil2d_contract import validate_coarse_indices
from .airfoil2d import _surface_force_coefficients
from .airfoil2d_single_feature import coarse_gradient_feature, coarse_curvature_feature


def pressure_gradient(coords, pressure, k=16):
    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if points.shape != (len(values), 2) or not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError("finite coordinates and pressure must align")
    # Zero-distance C-grid duplicates do not define an independent derivative.
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    value = np.bincount(inverse, weights=values) / np.bincount(inverse)
    if len(unique) < 3 or k < 2:
        raise ValueError("pressure gradient needs three distinct points and k >= 2")
    dist, neighbors = cKDTree(unique).query(unique, k=min(k + 1, len(unique)))
    offset = unique[neighbors[:, 1:]] - unique[:, None]
    difference = value[neighbors[:, 1:]] - value[:, None]
    radius = np.maximum(np.median(dist[:, 1:], axis=1), 1e-12)
    scaled = offset / radius[:, None, None]
    weight = np.exp(-np.square(dist[:, 1:] / radius[:, None]))
    matrix = np.einsum("nki,nkj,nk->nij", scaled, scaled, weight)
    rhs = np.einsum("nki,nk,nk->ni", scaled, difference, weight)
    ridge = 1e-3 * np.trace(matrix, axis1=1, axis2=2) / 2 + 1e-12
    matrix += ridge[:, None, None] * np.eye(2)
    gradient = np.linalg.solve(matrix, rhs[..., None])[..., 0] / radius[:, None]
    return np.linalg.norm(gradient, axis=1)[inverse]


def pressure_acquisition(coords, indices, pressure, mandatory, *, feature="gradient_lsq",
                         k=16, ell=.1, pressure_offset=0.0):
    points = np.asarray(coords, dtype=np.float64)
    coarse = validate_coarse_indices(indices, len(points), mandatory)
    p = np.asarray(pressure, dtype=np.float64).reshape(-1) - pressure_offset
    if len(p) != len(coarse) or not np.isfinite(p).all():
        raise ValueError("coarse pressure must be finite and aligned")
    if feature == "gradient_lsq":
        score = pressure_gradient(points[coarse], p, k)
        return score[cKDTree(points[coarse]).query(points, k=1)[1]]
    if feature == "gradient_energy":
        return coarse_gradient_feature(points[coarse], p, points, k)
    if feature == "curvature":
        return coarse_curvature_feature(points[coarse], p, points, k)
    order = {int(value): i for i, value in enumerate(coarse)}
    surface_p = p[[order[int(i)] for i in mandatory]]
    cd, cl = _surface_force_coefficients(points, np.asarray(mandatory))
    if feature == "lift_contribution":
        surface_score = np.abs(cl * surface_p)
    elif feature == "ratio_sensitivity":
        drag, lift = np.dot(cd, surface_p), np.dot(cl, surface_p)
        surface_score = (np.abs(cd - drag / lift * cl) if abs(lift) > 1e-12
                         else np.zeros_like(cl))
    else:
        raise ValueError(f"unknown single pressure feature: {feature}")
    if ell <= 0:
        raise ValueError("surface kernel scale must be positive")
    distance = np.linalg.norm(points[:, None] - points[np.asarray(mandatory)][None], axis=2)
    return np.max(np.exp(-distance / ell) * surface_score[None], axis=1)
