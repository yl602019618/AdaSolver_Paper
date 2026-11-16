from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference.base import SamplingResult
from adasolver.utils.interpolation import index_grid


def airfoil_surface_indices(shape: tuple[int, int] = (221, 51)) -> np.ndarray:
    return np.arange(50, 171, dtype=np.int64) * shape[1]


def surface_distance(coords: np.ndarray, shape: tuple[int, int] = (221, 51),
                     surface_indices: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if points.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"expected coordinates {(shape[0] * shape[1], 2)}, got {points.shape}")
    surface = airfoil_surface_indices(shape) if surface_indices is None else np.asarray(surface_indices, dtype=np.int64)
    if len(surface) == 0 or np.any(surface < 0) or np.any(surface >= len(points)):
        raise ValueError("surface indices must be a non-empty subset of the point set")
    return cKDTree(points[surface]).query(points, k=1)[0]


def sample_airfoil_pi(coords: np.ndarray, count: int, rng: np.random.Generator,
                      amp: float = 8.0, ell: float = 0.35, beta: float = 0.5,
                      mandatory_indices: np.ndarray | None = None) -> SamplingResult:
    n = len(coords)
    mandatory = airfoil_surface_indices() if mandatory_indices is None else np.asarray(mandatory_indices, dtype=np.int64)
    if count < len(mandatory) or count > n or ell <= 0 or amp < 0:
        raise ValueError("invalid Airfoil PI parameters or budget")
    distance = surface_distance(coords)
    weight = 1.0 + amp * np.exp(-distance / ell)
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory, assume_unique=False)
    probabilities = weight[available] / weight[available].sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    density = weight[indices] / weight.sum()
    compensation = (1.0 / weight[indices]) ** beta
    result = SamplingResult(indices, density, compensation,
                            {"method": "pi", "amp": amp, "ell": ell, "beta": beta,
                             "mandatory_count": int(len(mandatory))})
    result.validate(n, count)
    return result


def voronoi_mass(indices: np.ndarray, shape: tuple[int, int] = (221, 51)) -> np.ndarray:
    """Return the number of index-grid points represented by each sample."""
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    reference = index_grid(shape)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError("Voronoi indices must be a unique one-dimensional subset")
    if np.any(selected < 0) or np.any(selected >= len(reference)):
        raise ValueError("Voronoi index outside reference grid")
    nearest = cKDTree(reference[selected]).query(reference, k=1)[1]
    return np.bincount(nearest, minlength=len(selected)).astype(np.float64)


def realized_density_from_mass(mass: np.ndarray) -> np.ndarray:
    """Convert represented mass into a density normalized to mean one."""
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Voronoi mass must be positive and finite")
    return values.mean() / values


def sample_airfoil_pi_voronoi(
    coords: np.ndarray,
    count: int,
    rng: np.random.Generator,
    amp: float = 20.0,
    ell: float = 0.3,
    beta: float = 1.0,
    mandatory_indices: np.ndarray | None = None,
    shape: tuple[int, int] = (221, 51),
) -> SamplingResult:
    """Geometry-only PI sampling with measured Voronoi compensation."""
    points = np.asarray(coords, dtype=np.float64)
    n = len(points)
    mandatory = airfoil_surface_indices(shape) if mandatory_indices is None else np.asarray(mandatory_indices, dtype=np.int64)
    if points.shape != (shape[0] * shape[1], 2):
        raise ValueError(f"expected coordinates {(shape[0] * shape[1], 2)}, got {points.shape}")
    if count < len(mandatory) or count > n or ell <= 0 or amp < 0 or beta < 0:
        raise ValueError("invalid Airfoil Voronoi PI parameters or budget")
    distance = surface_distance(points, shape, mandatory)
    target_density = 1.0 + amp * np.exp(-distance / ell)
    available = np.setdiff1d(np.arange(n, dtype=np.int64), mandatory, assume_unique=False)
    probabilities = target_density[available] / target_density[available].sum()
    extra = rng.choice(available, count - len(mandatory), replace=False, p=probabilities)
    indices = np.concatenate((mandatory, extra))
    mass = voronoi_mass(indices, shape)
    realized_density = realized_density_from_mass(mass)
    compensation = np.power(realized_density, -beta)
    result = SamplingResult(
        indices,
        target_density[indices] / target_density.sum(),
        compensation,
        {
            "method": "pi_voronoi", "amp": amp, "ell": ell, "beta": beta,
            "mandatory_count": int(len(mandatory)),
            "target_density_type": "physical_surface_distance_exponential",
            "realized_density_type": "index_grid_voronoi_mass_inverse",
            "compensation_type": "realized_density_inverse_power",
            "target_density_min": float(target_density[indices].min()),
            "target_density_max": float(target_density[indices].max()),
            "voronoi_mass_min": float(mass.min()),
            "voronoi_mass_max": float(mass.max()),
            "voronoi_mass_mean": float(mass.mean()),
            "realized_density_min": float(realized_density.min()),
            "realized_density_max": float(realized_density.max()),
            "realized_density_mean": float(realized_density.mean()),
        },
    )
    result.validate(n, count)
    return result
