from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from adasolver.inference import adaptive
from adasolver.inference.acquisition import grad_score
from adasolver.inference.base import SamplingResult


class DarcyCPG:
    """Coarse-prediction-guided refinement for the Darcy lattice."""

    def __init__(
        self,
        reference_coords: np.ndarray,
        patch_count: int = 64,
        coarse_fraction: float = 0.25,
        kappa: float = 4.0,
        beta: float = 0.75,
        acquisition: str = "grad",
        score_power: float = 1.0,
        clip_quantile: float | None = None,
    ):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        if self.reference_coords.ndim != 2 or self.reference_coords.shape[1] != 2:
            raise ValueError("reference_coords must have shape (N, 2)")
        if not np.isfinite(self.reference_coords).all():
            raise ValueError("reference_coords must be finite")
        if acquisition not in {"grad", "abs", "lap", "grad_lap"}:
            raise ValueError(f"unsupported Darcy CPG acquisition: {acquisition}")
        if coarse_fraction <= 0 or coarse_fraction >= 1 or patch_count < 1:
            raise ValueError("invalid Darcy CPG geometry parameters")
        if kappa < 1 or beta < 0:
            raise ValueError("invalid Darcy CPG density parameters")
        if score_power <= 0 or clip_quantile is not None and not 0 < clip_quantile <= 1:
            raise ValueError("invalid Darcy CPG score parameters")
        self.patch_count = patch_count
        self.coarse_fraction = coarse_fraction
        self.kappa = kappa
        self.beta = beta
        self.acquisition = acquisition
        self.score_power = score_power
        self.clip_quantile = clip_quantile
        self.patch_id, self.patch_mass = adaptive.build_patches(
            self.reference_coords, patch_count, seed=0,
        )

    def coarse_count(self) -> int:
        return max(8, int(round(len(self.reference_coords) * self.coarse_fraction)))

    def _score(self, coarse_indices: np.ndarray, coarse_prediction: np.ndarray) -> np.ndarray:
        indices = np.asarray(coarse_indices, dtype=np.int64)
        values = np.asarray(coarse_prediction, dtype=np.float64).reshape(len(indices), -1)
        grad = grad_score(self.reference_coords[indices], values, k=8)
        if self.acquisition == "grad":
            score = grad
        else:
            # Darcy uses a fixed native lattice, so interpolate the coarse field
            # before finite-difference acquisition for abs/laplacian variants.
            axis = np.unique(self.reference_coords[:, 0])
            ay = np.unique(self.reference_coords[:, 1])
            gx, gy = np.meshgrid(axis, ay, indexing="xy")
            coarse = griddata(self.reference_coords[indices], values[:, 0],
                              np.column_stack((gx.reshape(-1), gy.reshape(-1))),
                              method="nearest")
            field = coarse.reshape(len(ay), len(axis))
            abs_score = np.abs(field).reshape(-1)
            lap_score = np.abs(
                np.gradient(np.gradient(field, axis=0), axis=0)
                + np.gradient(np.gradient(field, axis=1), axis=1)
            ).reshape(-1)
            if self.acquisition == "abs":
                xi = np.searchsorted(axis, self.reference_coords[indices, 0])
                yi = np.searchsorted(ay, self.reference_coords[indices, 1])
                score = abs_score[yi * len(axis) + xi]
            elif self.acquisition == "lap":
                xi = np.searchsorted(axis, self.reference_coords[indices, 0])
                yi = np.searchsorted(ay, self.reference_coords[indices, 1])
                score = lap_score[yi * len(axis) + xi]
            else:
                xi = np.searchsorted(axis, self.reference_coords[indices, 0])
                yi = np.searchsorted(ay, self.reference_coords[indices, 1])
                score = grad + lap_score[yi * len(axis) + xi]
        score = np.maximum(np.asarray(score, dtype=np.float64), 0.0)
        if self.clip_quantile is not None:
            score = np.minimum(score, np.quantile(score, self.clip_quantile))
        score = np.power(score / max(float(score.mean()), 1e-12), self.score_power)
        return score

    def refine(self, coarse_indices: np.ndarray, coarse_prediction: np.ndarray,
               final_count: int, rng: np.random.Generator) -> SamplingResult:
        coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
        if final_count <= len(coarse_indices) or final_count > len(self.reference_coords):
            raise ValueError("Darcy CPG budget must exceed coarse count and fit lattice")
        score = self._score(coarse_indices, coarse_prediction)
        refined, info = adaptive.refine_once(
            coarse_indices, score, self.reference_coords, self.patch_id,
            self.patch_mass, final_count, kappa=self.kappa,
            seed=int(rng.integers(1 << 30)),
        )
        compensation = adaptive.comp_weights_for_subset(
            refined, self.reference_coords, len(self.reference_coords), beta=self.beta,
        )
        if self.beta == 0:
            density = np.ones(len(compensation), dtype=np.float64)
        else:
            density = np.power(compensation.astype(np.float64), -1.0 / self.beta)
        result = SamplingResult(refined, density, compensation, {
            "method": "cpg", "coarse_fraction": self.coarse_fraction,
            "coarse_count": self.coarse_count(), "patch_count": self.patch_count,
            "kappa": self.kappa, "beta": self.beta, "acquisition": self.acquisition,
            "score_power": self.score_power, "clip_quantile": self.clip_quantile,
            "placement": "fps", "truth_access": False, **info,
        })
        result.validate(len(self.reference_coords), final_count)
        return result
