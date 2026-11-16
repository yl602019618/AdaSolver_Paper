from __future__ import annotations

"""Calibration-only Airfoil acquisition and reconstruction diagnostics.

This module deliberately uses ground-truth pressure to construct oracle scores. It is
not a deployable sampler and every caller must label its records as calibration-only.
"""

from scipy.spatial import cKDTree
import numpy as np
import torch

from adasolver.inference.cpg.airfoil2d_single_feature import (
    coarse_curvature_feature, coarse_gradient_feature,
    coarse_ratio_sensitivity_feature, sample_airfoil_cpg_single_feature,
)
from adasolver.metrics import pressure_surface_force, relative_l2

FEATURES = ("true_gradient", "coarse_gradient", "coarse_curvature", "coarse_ratio")


def gradient_magnitude_lsq(coords, values, k=16, ridge_fraction=1e-3,
                           clip_percentile=99.5):
    pos = np.asarray(coords, dtype=np.float64)
    field = np.asarray(values, dtype=np.float64).reshape(len(pos), -1)
    if pos.ndim != 2 or pos.shape[1] != 2 or field.shape[0] != len(pos):
        raise ValueError("coords must be (N,2) and values must align with N")
    if k < 1 or ridge_fraction < 0 or not np.isfinite(pos).all() or not np.isfinite(field).all():
        raise ValueError("invalid coordinates, values, or gradient parameters")
    kq = min(k + 1, len(pos))
    distance, neighbors = cKDTree(pos).query(pos, k=kq)
    offsets = pos[neighbors[:, 1:]] - pos[:, None, :]
    differences = field[neighbors[:, 1:]] - field[:, None, :]
    radius = np.maximum(np.median(distance[:, 1:], axis=1, keepdims=True), 1e-9)
    scaled = offsets / radius[..., None]
    weights = np.exp(-np.square(distance[:, 1:] / radius))
    weighted = scaled * weights[..., None]
    matrix = np.einsum("nmd,nme->nde", weighted, scaled)
    rhs = np.einsum("nmd,nmc->ndc", weighted, differences)
    trace = np.einsum("nii->n", matrix) / 2.0
    matrix += (ridge_fraction * trace + 1e-12)[:, None, None] * np.eye(2)[None]
    gradient = np.linalg.solve(matrix, rhs) / radius[..., None]
    score = np.sqrt(np.square(gradient).sum(axis=1)).sum(axis=1)
    if clip_percentile is not None and clip_percentile < 100:
        if clip_percentile <= 0:
            raise ValueError("clip_percentile must be positive")
        score = np.minimum(score, np.percentile(score, clip_percentile))
    return np.maximum(score, 0.0)


def feature_score(feature, coords, full_pressure, coarse_indices, mandatory_indices,
                  *, k=8, gradient_k=16):
    points = np.asarray(coords, dtype=np.float64)
    pressure = np.asarray(full_pressure, dtype=np.float64).reshape(-1)
    coarse = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
    mandatory = np.asarray(mandatory_indices, dtype=np.int64).reshape(-1)
    if points.shape != (len(pressure), 2) or len(np.unique(coarse)) != len(coarse):
        raise ValueError("coordinates, pressure, and coarse indices are misaligned")
    if np.any(coarse < 0) or np.any(coarse >= len(points)):
        raise ValueError("coarse index outside reference grid")
    if feature == "true_gradient":
        return gradient_magnitude_lsq(points, pressure, k=gradient_k)
    coarse_points, coarse_pressure = points[coarse], pressure[coarse]
    if feature == "coarse_gradient":
        return coarse_gradient_feature(coarse_points, coarse_pressure, points, k=k)
    if feature == "coarse_curvature":
        return coarse_curvature_feature(coarse_points, coarse_pressure, points, k=k)
    if feature == "coarse_ratio":
        return coarse_ratio_sensitivity_feature(points, coarse, coarse_pressure, mandatory)
    raise ValueError(f"unsupported Airfoil oracle feature: {feature}")


def metric_record(indices, reconstructed, target, coords, *, cl_threshold=0.01):
    points = np.asarray(coords)
    if points.shape == (11271, 2):
        prediction_force = pressure_surface_force(points, reconstructed)
        target_force = pressure_surface_force(points, target)
        force_record = {
            "ratio_abs_error": float(abs(prediction_force.ratio - target_force.ratio)),
            "truth_Cl": float(target_force.lift_coefficient),
            "pred_Cl": float(prediction_force.lift_coefficient),
            "truth_Cd_over_Cl": float(target_force.ratio),
            "pred_Cd_over_Cl": float(prediction_force.ratio),
            "cdcl_included": bool(target_force.lift_coefficient > cl_threshold),
        }
    else:
        # Small synthetic grids are useful for feature tests but do not define
        # the Airfoil surface-force discretization.
        force_record = {
            "ratio_abs_error": float("nan"), "truth_Cl": float("nan"),
            "pred_Cl": float("nan"), "truth_Cd_over_Cl": float("nan"),
            "pred_Cd_over_Cl": float("nan"), "cdcl_included": False,
        }
    return {
        "pressure_field_relative_l2": float(relative_l2(
            torch.as_tensor(np.asarray(reconstructed)[None]),
            torch.as_tensor(np.asarray(target)[None]))[0]),
        **force_record,
        "selected_count": int(len(indices)),
    }


def summarize_oracle_records(records, budgets, *, cl_threshold=0.01):
    result = {"truth_access": "calibration_only", "budgets": {}}
    for budget in budgets:
        rows = [row for row in records if row["budget_fraction"] == budget]
        field = np.asarray([row["pressure_field_relative_l2"] for row in rows], dtype=float)
        ratio_rows = [row for row in rows if row["cdcl_included"]]
        ratio = np.asarray([row["ratio_abs_error"] for row in ratio_rows], dtype=float)
        result["budgets"][f"{budget:.3f}"] = {
            "records": len(rows), "count": int(rows[0]["budget_count"]) if rows else 0,
            "mean_pressure_field_relative_l2": float(field.mean()) if len(field) else None,
            "mean_ratio_abs_error_cl_gt_threshold": float(ratio.mean()) if len(ratio) else None,
            "cdcl_records": len(ratio_rows), "cl_threshold": cl_threshold,
        }
    return result


def select_feature(coords, target, feature, coarse_indices, count, rng,
                   mandatory_indices, *, k=8, gradient_k=16):
    score = feature_score(feature, coords, target, coarse_indices, mandatory_indices,
                          k=k, gradient_k=gradient_k)
    sampling = sample_airfoil_cpg_single_feature(score, count, rng, mandatory_indices,
                                                  feature_name=feature)
    sampling.validate(len(coords), count)
    return sampling.indices, {"feature": feature, **sampling.metadata,
                              "truth_access": "calibration_only"}
