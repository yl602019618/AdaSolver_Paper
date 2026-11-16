from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PressureForce:
    drag_pressure: float
    lift_pressure: float
    drag_coefficient: float
    lift_coefficient: float

    @property
    def ratio(self) -> float:
        return self.drag_coefficient / self.lift_coefficient


def pressure_force(surface_xy: np.ndarray, surface_pressure: np.ndarray,
                   dynamic_pressure: float = 0.448, chord: float = 1.0) -> PressureForce:
    xy = np.asarray(surface_xy, dtype=np.float64)
    p = np.asarray(surface_pressure, dtype=np.float64).reshape(-1)
    if xy.shape != (len(p), 2) or len(p) < 2:
        raise ValueError(f"expected surface coordinates (N,2) and pressure (N,), got {xy.shape}, {p.shape}")
    edge_p = 0.5 * (p[:-1] + p[1:])
    drag = float(np.sum((xy[:-1, 1] - xy[1:, 1]) * edge_p))
    lift = float(np.sum((xy[1:, 0] - xy[:-1, 0]) * edge_p))
    scale = float(dynamic_pressure * chord)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("dynamic_pressure * chord must be positive and finite")
    return PressureForce(drag, lift, drag / scale, lift / scale)


def airfoil_surface(coords: np.ndarray, shape: tuple[int, int] = (221, 51)) -> np.ndarray:
    values = np.asarray(coords)
    if values.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"expected flattened coordinates {(shape[0] * shape[1], 2)}, got {values.shape}")
    return values[np.arange(50, 171) * shape[1]]


def pressure_surface_force(coords: np.ndarray, pressure: np.ndarray,
                           shape: tuple[int, int] = (221, 51), **kwargs) -> PressureForce:
    values = np.asarray(pressure)
    if values.shape == shape:
        values = values.reshape(-1)
    if values.shape == (shape[0] * shape[1], 1):
        values = values[:, 0]
    surface_idx = np.arange(50, 171) * shape[1]
    return pressure_force(airfoil_surface(coords, shape), values[surface_idx], **kwargs)
