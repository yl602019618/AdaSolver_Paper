"""ShapeNet-Car coarse-prediction-guided sampling.

The acquisition stage intentionally uses only the coarse predicted pressure by
 default.  It is therefore independent of the target field and can be used
for both pressure-focused and derived-metric experiments.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference import adaptive
from adasolver.inference.base import SamplingResult
from adasolver.inference.shapenet_car_contract import coarse_count, validate_indices


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite (N, 3) array")
    if len(points) < 2:
        raise ValueError(f"{name} must contain at least two points")
    return points


def _validate_pressure(pressure: np.ndarray, count: int) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if pressure.shape != (count,) or not np.all(np.isfinite(pressure)):
        raise ValueError("coarse pressure must be a finite vector matching coordinates")
    return pressure


def pressure_gradient_score(
    coarse_coords: np.ndarray,
    coarse_pressure: np.ndarray,
    reference_coords: np.ndarray | None = None,
    *,
    k: int = 8,
) -> np.ndarray:
    """Local pressure-gradient energy, optionally lifted to reference points."""
    points = _validate_points(coarse_coords, "coarse_coords")
    pressure = _validate_pressure(coarse_pressure, len(points))
    if k < 1:
        raise ValueError("k must be positive")
    kq = min(k + 1, len(points))
    distance, neighbor = cKDTree(points).query(points, k=kq)
    score = np.zeros(len(points), dtype=np.float64)
    for column in range(1, kq):
        delta = pressure - pressure[neighbor[:, column]]
        score += delta * delta / (distance[:, column] ** 2 + 1e-12)
    score /= max(kq - 1, 1)
    if reference_coords is not None:
        reference = _validate_points(reference_coords, "reference_coords")
        score = score[cKDTree(points).query(reference, k=1)[1]]
    return np.maximum(score, 0.0)


def pressure_curvature_score(
    coarse_coords: np.ndarray,
    coarse_pressure: np.ndarray,
    reference_coords: np.ndarray | None = None,
    *,
    k: int = 8,
) -> np.ndarray:
    """Squared residual from a distance-weighted local pressure mean."""
    points = _validate_points(coarse_coords, "coarse_coords")
    pressure = _validate_pressure(coarse_pressure, len(points))
    if k < 1:
        raise ValueError("k must be positive")
    kq = min(k + 1, len(points))
    distance, neighbor = cKDTree(points).query(points, k=kq)
    dist = distance[:, 1:]
    weights = 1.0 / (dist * dist + 1e-12)
    local_mean = (weights * pressure[neighbor[:, 1:]]).sum(axis=1) / np.maximum(
        weights.sum(axis=1), 1e-12
    )
    score = (pressure - local_mean) ** 2
    if reference_coords is not None:
        reference = _validate_points(reference_coords, "reference_coords")
        score = score[cKDTree(points).query(reference, k=1)[1]]
    return np.maximum(score, 0.0)


def normalize_cpg_score(
    score: np.ndarray,
    *,
    method: Literal["q95", "rank", "robust"] = "q95",
    floor: float = 0.1,
    power: float = 1.0,
) -> np.ndarray:
    """Return a positive, mean-scaled acquisition score."""
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("score must be finite and non-empty")
    if floor < 0 or power <= 0:
        raise ValueError("floor must be non-negative and power must be positive")
    values = np.maximum(values, 0.0)
    if method == "q95":
        scale = float(np.quantile(values, 0.95))
        normalized = values / max(scale, 1e-12)
    elif method == "rank":
        from scipy.stats import rankdata
        normalized = rankdata(values, method="average") / len(values)
    elif method == "robust":
        median = float(np.median(values))
        scale = float(np.median(np.abs(values - median)))
        normalized = np.abs(values - median) / max(1.4826 * scale, 1e-12)
    else:
        raise ValueError(f"unsupported score normalization: {method}")
    normalized = np.maximum(normalized, float(floor))
    return np.power(normalized, float(power))


def surface_scalar_score(reference, surface_mask, coarse_indices, coarse_values, *,
                         feature="gradient", k=8, surface_band=.1, influence=None):
    """Differentiate pressure only where it is supervised, then lift its score.

    Volume pressure predictions are never inspected. The geometric distance
    envelope is measured from the full known surface, not the sparse subset.
    """
    reference = _validate_points(reference, "reference")
    mask = np.asarray(surface_mask, dtype=bool)
    coarse = validate_indices(coarse_indices, len(reference))
    values = np.asarray(coarse_values).reshape(-1)
    if mask.shape != (len(reference),) or len(values) != len(coarse):
        raise ValueError("surface mask and coarse values must align")
    active = mask[coarse]
    if np.count_nonzero(active) < 2:
        raise ValueError("pressure acquisition requires two coarse surface points")
    if surface_band <= 0:
        raise ValueError("surface_band must be positive")
    functions = {"gradient": pressure_gradient_score, "curvature": pressure_curvature_score}
    if feature not in functions:
        raise ValueError(f"unknown scalar feature {feature}")
    points = reference[coarse[active]]
    local = functions[feature](points, values[active], k=k)
    surface = reference[mask]
    on_surface = local[cKDTree(points).query(surface, k=1)[1]]
    if influence is not None:
        influence = np.abs(np.asarray(influence, dtype=np.float64))
        if influence.shape != (len(surface),) or not np.isfinite(influence).all():
            raise ValueError("pressure influence must align with the full surface")
        on_surface *= np.maximum(influence / max(float(np.quantile(influence, .95)), 1e-12), .05)
    distance, nearest = cKDTree(surface).query(reference, k=1)
    return on_surface[nearest] * np.exp(-distance / surface_band)


def cd_scalar_score(reference, surface_mask, coarse_indices, coarse_pressure, influence,
                    *, feature="drag_load", k=8, surface_band=.1):
    """One pressure scalar projected onto the known Cd pressure integral.

    Load uses mean-centered pressure (gauge invariant); the derivative variants
    use one local pressure variation amplitude. No velocity prediction or target
    enters this acquisition, and volume pressure placeholders are ignored.
    """
    reference = _validate_points(reference, "reference")
    mask = np.asarray(surface_mask, dtype=bool)
    coarse = validate_indices(coarse_indices, len(reference))
    values = np.asarray(coarse_pressure).reshape(-1)
    if mask.shape != (len(reference),) or len(values) != len(coarse):
        raise ValueError("pressure and mask must align")
    active = mask[coarse]
    points = reference[coarse[active]]
    pressure = _validate_pressure(values[active], len(points))
    if len(points) < 2 or surface_band <= 0:
        raise ValueError("Cd score needs two coarse surface points and a positive band")
    signed_influence = np.asarray(influence, dtype=np.float64)
    influence = np.abs(signed_influence)
    if influence.shape != (int(mask.sum()),) or not np.isfinite(influence).all():
        raise ValueError("Cd influence must align with the full surface")
    if feature in {"drag_positive", "drag_negative"}:
        local = pressure - pressure.mean()
    elif feature == "drag_load":
        local = np.abs(pressure - pressure.mean())
    elif feature == "drag_gradient":
        local = np.sqrt(pressure_gradient_score(points, pressure, k=k))
    elif feature == "drag_curvature":
        local = np.sqrt(pressure_curvature_score(points, pressure, k=k))
    else:
        raise ValueError(f"unknown single-scalar Cd score: {feature}")
    surface = reference[mask]
    surface_value = local[cKDTree(points).query(surface, k=1)[1]]
    if feature in {"drag_positive", "drag_negative"}:
        # Cd's pressure coefficient is proportional to minus normal*area.
        sign = 1. if feature == "drag_positive" else -1.
        surface_score = np.maximum(sign * (-signed_influence) * surface_value, 0.)
    else:
        surface_score = surface_value * influence
    distance, nearest = cKDTree(surface).query(reference, k=1)
    return surface_score[nearest] * np.exp(-distance / surface_band)


class ShapeNetCarCPG:
    """10% acquisition and shared patch refinement, also used by CA heads."""

    def __init__(
        self,
        reference_coords: np.ndarray,
        surface_mask: np.ndarray,
        *,
        coarse_fraction: float = 0.10,
        patch_count: int = 48,
        neighborhood: int = 8,
        feature: Literal["gradient", "curvature"] = "gradient",
        normalization: Literal["q95", "rank", "robust"] = "q95",
        score_floor: float = 0.1,
        score_power: float = 1.0,
        kappa: float = 8.0,
        beta: float = 0.75,
        placement: Literal["fps", "random"] = "fps",
        signal: str = "pressure",
        surface_band: float = .1,
        patches=None,
        fps_distance_backend: str = "numpy",
    ):
        self.reference_coords = _validate_points(reference_coords, "reference_coords")
        self.surface_mask = np.asarray(surface_mask, dtype=bool).reshape(-1)
        if self.surface_mask.shape != (len(self.reference_coords),):
            raise ValueError("surface_mask must match reference_coords")
        if not self.surface_mask.any():
            raise ValueError("surface_mask must contain a surface point")
        coarse_count(len(self.reference_coords), coarse_fraction)
        if patch_count < 1:
            raise ValueError("invalid coarse fraction or patch count")
        if neighborhood < 1 or score_floor < 0 or score_power <= 0:
            raise ValueError("invalid score parameters")
        if kappa < 1 or beta < 0 or placement != "fps":
            raise ValueError("invalid refinement parameters")
        if signal not in {"pressure", "speed"} or surface_band <= 0:
            raise ValueError("invalid acquisition signal or surface band")
        if feature not in {"gradient", "curvature", "drag_load", "drag_gradient", "drag_curvature",
                           "drag_positive", "drag_negative"}:
            raise ValueError(f"unsupported ShapeNet-Car CPG feature: {feature}")
        self.coarse_fraction = float(coarse_fraction)
        self.patch_count = min(int(patch_count), len(self.reference_coords))
        self.neighborhood = int(neighborhood)
        self.feature = feature
        self.normalization = normalization
        self.score_floor = float(score_floor)
        self.score_power = float(score_power)
        self.kappa = float(kappa)
        self.beta = float(beta)
        self.placement = placement
        self.signal, self.surface_band = signal, float(surface_band)
        if fps_distance_backend not in {"numpy", "scipy", "native"}:
            raise ValueError("invalid FPS distance backend")
        self.fps_distance_backend = fps_distance_backend
        self.patch_id, self.patch_mass = patches if patches is not None else adaptive.build_patches(
            self.reference_coords, self.patch_count, seed=0)

    def coarse_count(self) -> int:
        return coarse_count(len(self.reference_coords), self.coarse_fraction)

    def _score(self, coarse_indices: np.ndarray, coarse_pressure: np.ndarray,
               *, influence=None) -> np.ndarray:
        coarse_indices = validate_indices(coarse_indices, len(self.reference_coords), self.coarse_count())
        if self.feature.startswith("drag_"):
            if self.signal != "pressure" or influence is None:
                raise ValueError("Cd acquisition requires pressure and fixed geometric influence")
            raw = cd_scalar_score(self.reference_coords, self.surface_mask, coarse_indices,
                coarse_pressure, influence, feature=self.feature, k=self.neighborhood,
                surface_band=self.surface_band)
        elif self.signal == "pressure":
            raw = surface_scalar_score(self.reference_coords, self.surface_mask, coarse_indices,
                coarse_pressure, feature=self.feature, k=self.neighborhood,
                surface_band=self.surface_band, influence=influence)
        else:
            points = self.reference_coords[coarse_indices]
            score_fn = pressure_gradient_score if self.feature == "gradient" else pressure_curvature_score
            raw = score_fn(points, coarse_pressure, self.reference_coords, k=self.neighborhood)
        return normalize_cpg_score(raw, method=self.normalization,
                                   floor=self.score_floor, power=self.score_power)

    def refine(
        self,
        coarse_indices: np.ndarray,
        coarse_pressure: np.ndarray,
        final_count: int,
        rng: np.random.Generator,
        *, influence=None,
    ) -> SamplingResult:
        coarse_indices = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
        if final_count <= len(coarse_indices) or final_count > len(self.reference_coords):
            raise ValueError("final_count must exceed coarse count and fit reference cloud")
        if len(coarse_indices) != self.coarse_count():
            raise ValueError("coarse_indices does not match configured coarse count")
        score = self._score(coarse_indices, coarse_pressure, influence=influence)
        return self.refine_score(coarse_indices, score, final_count, rng)

    def refine_score(self, coarse_indices, score, final_count, rng) -> SamplingResult:
        coarse_indices = validate_indices(coarse_indices, len(self.reference_coords), self.coarse_count())
        if not len(coarse_indices) < final_count <= len(self.reference_coords):
            raise ValueError("final_count must exceed coarse count and fit reference cloud")
        refined, info = adaptive.refine_once(
            coarse_indices, score, self.reference_coords, self.patch_id,
            self.patch_mass, final_count, kappa=self.kappa,
            seed=int(rng.integers(1 << 30)),
            fps_distance_backend=self.fps_distance_backend,
        )
        compensation = adaptive.comp_weights_for_subset(
            refined, self.reference_coords, len(self.reference_coords), beta=self.beta
        )
        if self.beta == 0:
            density = np.ones(len(refined), dtype=np.float64)
        else:
            density = np.power(np.maximum(compensation.astype(np.float64), 1e-12),
                               -1.0 / self.beta)
        surface_coverage = float(np.count_nonzero(self.surface_mask[refined]) /
                                 max(np.count_nonzero(self.surface_mask), 1))
        metadata = {
            "method": "cpg",
            "feature": f"{self.signal}_{self.feature}",
            "score_domain": "coarse_surface" if self.signal == "pressure" else "coarse_cloud",
            "surface_band": self.surface_band,
            "density_estimator": "final_voronoi_mass",
            "fps_distance_backend": self.fps_distance_backend,
            "coarse_count": int(len(coarse_indices)),
            "coarse_fraction": self.coarse_fraction,
            "neighborhood": self.neighborhood,
            "normalization": self.normalization,
            "score_floor": self.score_floor,
            "score_power": self.score_power,
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "score_mean": float(score.mean()),
            "patch_count": self.patch_count,
            "kappa": self.kappa,
            "beta": self.beta,
            "placement": self.placement,
            "surface_coverage": surface_coverage,
            "truth_access": False,
            **info,
        }
        result = SamplingResult(refined, density, compensation, metadata)
        result.validate(len(self.reference_coords), final_count)
        return result
