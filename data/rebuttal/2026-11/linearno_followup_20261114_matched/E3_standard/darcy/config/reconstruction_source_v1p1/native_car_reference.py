"""Mask-aware ShapeNet-Car reconstruction and dual-field metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CarFieldMetrics:
    velocity_rel_l2: float
    pressure_rel_l2: float
    field_error: float


def knn_idw(
    source_positions,
    source_values,
    target_positions,
    *,
    k: int = 4,
    eps: float = 1e-12,
):
    """Interpolate scalar or vector values with deterministic kNN inverse-distance weights."""
    source_positions = np.asarray(source_positions)
    source_values = np.asarray(source_values)
    target_positions = np.asarray(target_positions)
    if (
        source_positions.ndim != 2
        or target_positions.ndim != 2
        or source_positions.shape[1] != target_positions.shape[1]
        or len(source_positions) == 0
    ):
        raise ValueError(
            "source_positions and target_positions must be 2-D arrays with matching dimensions"
        )
    if not np.isfinite(source_positions).all() or not np.isfinite(target_positions).all():
        raise ValueError("interpolation positions contain NaN or Inf")
    if len(source_values) != len(source_positions):
        raise ValueError("source values and positions must align")
    scalar = source_values.ndim == 1
    values = source_values[:, None] if scalar else source_values
    if values.ndim != 2:
        raise ValueError("source_values must have shape (N,) or (N,F)")
    if not np.isfinite(values).all():
        raise ValueError("interpolation values contain NaN or Inf")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")

    effective_k = min(int(k), len(source_positions))
    distances, indices = cKDTree(source_positions).query(target_positions, k=effective_k)
    if effective_k == 1:
        output = values[indices]
        return output[:, 0] if scalar else output

    distances = np.asarray(distances)
    indices = np.asarray(indices)
    weights = 1.0 / (distances + eps)
    neighbor_values = values[indices]
    output = (neighbor_values * weights[..., None]).sum(axis=1) / weights.sum(axis=1)[..., None]

    exact = distances <= eps
    exact_rows = np.flatnonzero(exact.any(axis=1))
    first_exact = np.argmax(exact[exact_rows], axis=1)
    output[exact_rows] = neighbor_values[exact_rows, first_exact]
    return output[:, 0] if scalar else output


def decode_outputs(prediction_norm, output_mean, output_std):
    prediction_norm = np.asarray(prediction_norm)
    return prediction_norm * np.asarray(output_std) + np.asarray(output_mean)


def dual_field_metrics(
    prediction_phys,
    target_phys,
    surface_mask,
    *,
    pressure_weight: float = 0.5,
    eps: float = 1e-12,
) -> CarFieldMetrics:
    prediction = np.asarray(prediction_phys)
    target = np.asarray(target_phys)
    mask = np.asarray(surface_mask, dtype=bool)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 4
        or mask.shape != (len(prediction),)
    ):
        raise ValueError("expected prediction/target (N,4) and surface_mask (N,)")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction or target contains NaN or Inf")
    if not mask.any():
        raise ValueError("surface_mask contains no surface points")
    if pressure_weight < 0:
        raise ValueError("pressure_weight must be non-negative")

    velocity_error = np.linalg.norm(prediction[:, :3] - target[:, :3]) / (
        np.linalg.norm(target[:, :3]) + eps
    )
    pressure_error = np.linalg.norm(prediction[mask, 3] - target[mask, 3]) / (
        np.linalg.norm(target[mask, 3]) + eps
    )
    return CarFieldMetrics(
        velocity_rel_l2=float(velocity_error),
        pressure_rel_l2=float(pressure_error),
        field_error=float(velocity_error + pressure_weight * pressure_error),
    )


def reconstruct_car_fields(
    selected_indices,
    sampled_prediction,
    positions,
    surface_mask,
    *,
    prediction_is_normalized: bool = False,
    output_mean=None,
    output_std=None,
    k: int = 4,
):
    """Reconstruct velocity on the full cloud and pressure on the full surface."""
    positions = np.asarray(positions)
    mask = np.asarray(surface_mask, dtype=bool)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3 or mask.shape != (len(positions),):
        raise ValueError("positions must have shape (N,3) and surface_mask shape (N,)")
    if not mask.any():
        raise ValueError("surface_mask contains no surface points")
    if (
        selected.ndim != 1
        or len(selected) == 0
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(positions))
    ):
        raise ValueError("selected_indices must contain unique valid indices")

    prediction = np.asarray(sampled_prediction)
    if prediction.shape != (len(selected), 4):
        raise ValueError("sampled_prediction must have shape (len(selected_indices), 4)")
    if prediction_is_normalized:
        if output_mean is None or output_std is None:
            raise ValueError("normalized predictions require output_mean and output_std")
        prediction = decode_outputs(prediction, output_mean, output_std)
    elif output_mean is not None or output_std is not None:
        raise ValueError("output statistics are only valid when prediction_is_normalized=True")

    sampled_surface = mask[selected]
    if not sampled_surface.any():
        raise ValueError(
            "selected points contain no surface points; pressure reconstruction is undefined"
        )
    velocity = knn_idw(
        positions[selected], prediction[:, :3], positions, k=k
    )
    surface_indices = np.flatnonzero(mask)
    pressure = knn_idw(
        positions[selected][sampled_surface],
        prediction[sampled_surface, 3],
        positions[surface_indices],
        k=k,
    )
    return velocity, pressure, surface_indices


def car_dual_rel_l2(
    selected_indices,
    sampled_prediction_norm,
    target_phys,
    positions,
    surface_mask,
    output_mean,
    output_std,
    *,
    k: int = 4,
    pressure_weight: float = 0.5,
) -> CarFieldMetrics:
    velocity, pressure, surface_indices = reconstruct_car_fields(
        selected_indices,
        sampled_prediction_norm,
        positions,
        surface_mask,
        prediction_is_normalized=True,
        output_mean=output_mean,
        output_std=output_std,
        k=k,
    )
    reconstructed = np.empty_like(np.asarray(target_phys))
    reconstructed[:, :3] = velocity
    reconstructed[:, 3] = 0
    reconstructed[surface_indices, 3] = pressure
    return dual_field_metrics(
        reconstructed,
        target_phys,
        surface_mask,
        pressure_weight=pressure_weight,
    )
