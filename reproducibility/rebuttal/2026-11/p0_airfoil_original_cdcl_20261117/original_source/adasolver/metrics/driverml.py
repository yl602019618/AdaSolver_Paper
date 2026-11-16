"""DriverML metrics for pressure reconstruction and Cd closure."""
from __future__ import annotations

import numpy as np
import torch

from adasolver.metrics.field import relative_l2


def pressure_relative_l2(prediction, target):
    return float(relative_l2(prediction.double().reshape(1, -1),
                             target.double().reshape(1, -1))[0])


def integrate_cd(normal, area, pressure, wss, a_ref, rho=1.0, velocity=38.889):
    """DrivAerML label convention: pressure +p*n*A, viscous -tau*A."""
    n = len(area)
    if normal.shape != (n, 3) or wss.shape != (n, 3) or pressure.numel() != n:
        raise ValueError("Cd inputs must share the full surface point order")
    if not np.isfinite([a_ref, rho, velocity]).all() or min(a_ref, rho, velocity) <= 0:
        raise ValueError("Cd reference area, density and velocity must be positive")
    weights = area.double().reshape(-1)
    den = 0.5 * rho * velocity ** 2 * a_ref
    pressure_drag = float((normal[:, 0].double() * pressure.double().reshape(-1) * weights).sum()) / den
    viscous_drag = -float((wss[:, 0].double() * weights).sum()) / den
    return {"cd": pressure_drag + viscous_drag, "cd_pressure": pressure_drag,
            "cd_viscous": viscous_drag}


def reconstruct_nearest(reference_coords: torch.Tensor, sampled_coords: torch.Tensor,
                        sampled_prediction: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor reconstruction from sampled points to a full mesh."""
    from scipy.spatial import cKDTree
    ref = np.asarray(reference_coords.detach().cpu(), dtype=np.float64)
    src = np.asarray(sampled_coords.detach().cpu(), dtype=np.float64)
    values = sampled_prediction.detach().cpu()
    if ref.ndim != 2 or src.ndim != 2 or ref.shape[1] != src.shape[1]:
        raise ValueError("reference and sampled coordinates must be compatible point clouds")
    if len(src) == 0 or len(values) != len(src):
        raise ValueError("sampled coordinates and predictions must be non-empty and aligned")
    _, indices = cKDTree(src).query(ref, k=1)
    return values[torch.from_numpy(np.asarray(indices, dtype=np.int64))]


def area_relative_l2(prediction: torch.Tensor, target: torch.Tensor,
                     area: torch.Tensor, eps: float = 1e-12) -> float:
    pred = prediction.double().reshape(-1)
    truth = target.double().reshape(-1)
    weights = area.double().reshape(-1)
    if pred.shape != truth.shape or pred.shape != weights.shape:
        raise ValueError("pressure and area must have matching point counts")
    value = (weights * (pred - truth).square()).sum() / (weights * truth.square()).sum().clamp_min(eps)
    return float(value.sqrt())


def relative_cd_error(predicted: float, target: float, eps: float = 1e-12) -> float:
    return abs(float(predicted) - float(target)) / (abs(float(target)) + eps)


def decode_pressure(normalized: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return normalized * float(std) + float(mean)


def decode_wss(normalized: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=normalized.dtype, device=normalized.device)
    std_t = torch.as_tensor(std, dtype=normalized.dtype, device=normalized.device)
    return normalized * std_t + mean_t
