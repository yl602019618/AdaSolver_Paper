"""SuperWing cell-centered field metrics and canonical aerodynamic integration.

Implements the geometry/force equations in SuperWing (arXiv:2512.14397) and
the dataset postprocessing convention. Geometry is vertex-centered; targets
are cell-centered [Cp, 150 Cf_tau, 300 Cf_z]. No legacy code import is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adasolver.metrics.field import relative_l2


@dataclass(frozen=True)
class SurfaceGeometry:
    normals: np.ndarray
    area: np.ndarray
    centers: np.ndarray
    tangent_xy: np.ndarray


def surface_geometry(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or min(vertices.shape[:2]) < 3:
        raise ValueError('vertices must have shape (H+1,W+1,3)')
    if not np.isfinite(vertices).all():
        raise ValueError('non-finite surface vertices')
    p0, p1 = vertices[:-1, :-1], vertices[:-1, 1:]
    p2, p3 = vertices[1:, 1:], vertices[1:, :-1]
    normals = np.cross(p2-p0, p3-p1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-20
    area = 0.5*(np.linalg.norm(np.cross(p1-p0, p2-p0), axis=-1) +
                np.linalg.norm(np.cross(p2-p0, p3-p0), axis=-1))
    if np.any(area <= 0):
        raise ValueError('surface cells must have positive area')
    tangent = np.diff(vertices[..., :2], axis=1)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-20
    # Average neighboring spanwise tangents, matching the supplied closure.
    tangent = 0.5*(tangent[:-1] + tangent[1:])
    return SurfaceGeometry(normals, area, 0.25*(p0+p1+p2+p3), tangent)


def force_basis(geometry):
    """Per-cell force vector for each of the three stored field channels."""
    n, area = geometry.normals, geometry.area
    tangential = np.concatenate((geometry.tangent_xy/150.0, np.zeros_like(area[..., None])), axis=-1)
    spanwise = np.zeros_like(n)
    spanwise[..., 2] = 1.0/300.0
    projected = [n]
    for direction in (tangential, spanwise):
        projected.append(direction - np.sum(direction*n, axis=-1, keepdims=True)*n)
    return np.stack(projected, axis=-2)*area[..., None, None]


def coefficient_weights(geometry, aoa, s_ref, c_ref=1.0):
    """Weights (3,N,3) for CL, CD, CMz applied to the stored field (N,3)."""
    if not np.isfinite([aoa, s_ref, c_ref]).all() or s_ref <= 0 or c_ref <= 0:
        raise ValueError('finite AoA and positive reference area/chord are required')
    alpha = np.deg2rad(aoa)
    drag = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    lift = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    basis = force_basis(geometry)
    cd = np.einsum('hwcv,v->hwc', basis, drag)/s_ref
    cl = np.einsum('hwcv,v->hwc', basis, lift)/s_ref
    arm = geometry.centers - np.array([0.25, 0.0, 0.0])
    cm = np.cross(arm[..., None, :], basis)[..., 2]/(s_ref*c_ref)
    return np.stack([cl, cd, cm]).reshape(3, -1, 3)


def integrate_field(field, weights):
    field, weights = np.asarray(field), np.asarray(weights)
    if field.ndim != 2 or weights.shape != (3, *field.shape) or field.shape[1] != 3:
        raise ValueError('field (N,3) and CL/CD/CMz weights (3,N,3) must align')
    if not np.isfinite(field).all():
        raise ValueError('non-finite predicted field')
    return np.einsum('knc,nc->k', weights, field, dtype=np.float64)


def evaluate_field(prediction, target, weights, structured):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError('SuperWing full-field predictions and targets must be matching (N,3)')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('non-finite field evaluation')
    pred_t = torch.from_numpy(np.asarray(prediction, dtype=np.float64).copy())
    true_t = torch.from_numpy(np.asarray(target, dtype=np.float64).copy())
    errors = relative_l2(pred_t[None], true_t[None])
    channels = relative_l2(pred_t.T, true_t.T)
    coef = integrate_field(prediction, weights)
    true_coef = integrate_field(target, weights)
    structured = np.asarray(structured, dtype=np.float64)
    if structured.shape != (3,) or not np.isfinite(structured).all():
        raise ValueError('structured labels must be finite [CL,CD,CMz]')
    absolute = np.abs(coef - structured)
    relative = absolute/np.maximum(np.abs(structured), 1e-12)
    ranges = np.maximum(np.ptp(target, axis=0), 1e-12)
    paper = np.abs(prediction-target).mean(axis=0)/ranges
    contributions = (weights[1]*prediction).sum(axis=0)
    result = dict(field_relative_l2=float(errors[0]), cd_relative_error=float(relative[1]),
                  cd_absolute_error=float(absolute[1]), cd_drag_counts=float(absolute[1]*1e4),
                  pressure_Cd=float(contributions[0]), friction_Cd=float(contributions[1:].sum()),
                  Cd_pred=float(coef[1]), Cd_gt_structured=float(structured[1]),
                  Cd_gt_field_closure=float(true_coef[1]),
                  Cl_pred=float(coef[0]), Cl_gt_structured=float(structured[0]),
                  cl_relative_error=float(relative[0]), cl_absolute_error=float(absolute[0]),
                  CMz_pred=float(coef[2]), CMz_gt_structured=float(structured[2]))
    for i, name in enumerate(('Cp', 'Cf_tau', 'Cf_z')):
        result[f'{name}_relative_l2'] = float(channels[i])
        result[f'{name}_range_normalized_mae'] = float(paper[i])
    return result
