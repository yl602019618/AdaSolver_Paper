"""Cd-oriented coarse-prediction-guided sampling for ShapeNet-Car.

The acquisition uses one predicted scalar (coarse pressure) and fixed surface
geometry.  It does not use the target field or the target Cd.  The geometry
influence is the pressure-drag coefficient per surface point (cell area times
streamwise normal, with any constant normalization absorbed by the score).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference import adaptive
from adasolver.inference.base import SamplingResult


def _check_points(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape (N, 3)")
    return value


def map_surface_influence(
    reference_coords: np.ndarray,
    surface_coords: np.ndarray,
    surface_influence: np.ndarray,
    *,
    tolerance: float | None = None,
) -> np.ndarray:
    """Map fixed VTK surface influence to the reference cloud by nearest point.

    ``surface_influence`` must already be computed from the audited VTK mesh,
    e.g. ``cell_area * normal_z`` after the point-to-cell convention has been
    established.  A tolerance can reject an unverified coordinate mapping.
    """
    reference = _check_points("reference_coords", reference_coords)
    surface = _check_points("surface_coords", surface_coords)
    influence = np.asarray(surface_influence, dtype=np.float64).reshape(-1)
    if len(surface) != len(influence) or not np.isfinite(influence).all():
        raise ValueError("surface_coords and surface_influence must have equal finite length")
    if len(surface) == 0:
        raise ValueError("surface_coords cannot be empty")
    distance, nearest = cKDTree(surface).query(reference, k=1)
    if tolerance is not None and (tolerance < 0 or np.any(distance > tolerance)):
        raise ValueError("surface-to-reference mapping exceeds tolerance")
    return influence[nearest]


def cd_pressure_drag_score(reference_coords, coarse_indices, coarse_pressure,
                           surface_coords, surface_influence, *, k=8, surface_band=.1):
    from adasolver.inference.cpg.shapenet_car import surface_scalar_score
    reference = _check_points("reference_coords", reference_coords)
    distance = cKDTree(surface_coords).query(reference, k=1)[0]
    mask = distance <= 1e-6
    mapped = map_surface_influence(reference[mask], surface_coords, surface_influence,
                                   tolerance=1e-6)
    return surface_scalar_score(reference, mask, coarse_indices, coarse_pressure,
                                k=k, surface_band=surface_band, influence=mapped)


class ShapeNetCarCdCPG:
    """Cd wrapper around the same surface-pressure acquisition used by CA."""

    def __init__(self, reference_coords, *, surface_mask=None, patch_count=48,
                 coarse_fraction=.1, kappa=8., beta=.75, neighborhood=8,
                 score_power=1., score_floor=.1, surface_band=.1, patches=None):
        from adasolver.inference.cpg.shapenet_car import ShapeNetCarCPG
        self.reference_coords = _check_points("reference_coords", reference_coords)
        self.surface_mask = surface_mask
        self.sampler = ShapeNetCarCPG(reference_coords,
            np.ones(len(reference_coords), dtype=bool) if surface_mask is None else surface_mask,
            patch_count=patch_count, coarse_fraction=coarse_fraction, kappa=kappa,
            beta=beta, neighborhood=neighborhood, score_power=score_power,
            score_floor=score_floor, surface_band=surface_band, patches=patches)

    def coarse_count(self):
        return self.sampler.coarse_count()

    def refine(self, coarse_indices, coarse_pressure, final_count, rng, *,
               surface_coords, surface_influence, mapping_tolerance=1e-6):
        if len(coarse_indices) != self.coarse_count():
            raise ValueError("coarse set is smaller than configured coarse count or has an invalid size")
        if self.surface_mask is None:
            self.sampler.surface_mask = cKDTree(surface_coords).query(self.reference_coords, k=1)[0] <= 1e-6
        mapped = map_surface_influence(self.reference_coords[self.sampler.surface_mask],
            surface_coords, surface_influence, tolerance=mapping_tolerance)
        return self.sampler.refine(coarse_indices, coarse_pressure, final_count, rng, influence=mapped)
