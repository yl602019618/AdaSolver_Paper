from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .field import relative_l2


def full20_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """One physical space-time displacement norm per sample; frames are diagnostics."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim == 2 and pred.shape[-1] == 40:
        pred = pred.reshape(-1, 20, 2)
    if truth.ndim == 2 and truth.shape[-1] == 40:
        truth = truth.reshape(-1, 20, 2)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1:] != (20, 2):
        raise ValueError("full20 requires matching (points,20,2) or (points,40) arrays")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("full20 fields must be finite")
    error = np.square(pred - truth).sum(axis=(0, 2))
    energy = np.square(truth).sum(axis=(0, 2))
    valid = energy > 1e-24
    per_frame = [float(np.sqrt(e / d)) if ok else None
                 for e, d, ok in zip(error, energy, valid)]
    primary = relative_l2(torch.from_numpy(pred[None]), torch.from_numpy(truth[None]))
    return {
        "full_field_relative_l2": float(primary[0]),
        "squared_error_total": float(error.sum()), "target_energy_total": float(energy.sum()),
        "per_frame_squared_error": error.tolist(), "per_frame_target_energy": energy.tolist(),
        "per_frame_relative_l2": per_frame, "valid_frame_mask": valid.tolist(),
        "per_frame_rmse": np.sqrt(error / (len(pred) * 2)).tolist(),
        "terminal_relative_l2": per_frame[-1],
    }


@dataclass(frozen=True)
class TemporalFieldMetrics:
    per_frame_relative_l2: np.ndarray
    sequence_mean_relative_l2: float
    terminal_relative_l2: float


def temporal_relative_l2(prediction: np.ndarray, target: np.ndarray,
                         epsilon: float = 1e-12) -> TemporalFieldMetrics:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError(f"expected matching (points, frames, channels), got {pred.shape} and {truth.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("temporal fields must be finite")
    numerator = np.linalg.norm(pred - truth, axis=(0, 2))
    denominator = np.linalg.norm(truth, axis=(0, 2))
    if np.any(denominator <= epsilon):
        raise ValueError("relative L2 is undefined for a zero-norm frame")
    per_frame = numerator / denominator
    return TemporalFieldMetrics(per_frame, float(per_frame.mean()), float(per_frame[-1]))


def temporal_rauec(budgets: np.ndarray, method_errors: np.ndarray,
                   uniform_errors: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(budgets, dtype=np.float64)
    method = np.asarray(method_errors, dtype=np.float64)
    uniform = np.asarray(uniform_errors, dtype=np.float64)
    if method.shape != uniform.shape or method.ndim != 2 or method.shape[0] != len(x):
        raise ValueError("errors must have shape (budgets, frames) and match budgets")
    uniform_area = np.trapezoid(uniform, x=x, axis=0)
    if np.any(uniform_area <= 0):
        raise ValueError("uniform per-frame AUEC must be positive")
    per_frame = np.trapezoid(method, x=x, axis=0) / uniform_area
    sequence = float(np.trapezoid(method.mean(axis=1), x=x) /
                     np.trapezoid(uniform.mean(axis=1), x=x))
    return sequence, per_frame
