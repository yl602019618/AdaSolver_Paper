from __future__ import annotations

import numpy as np

from adasolver.inference import adaptive
from adasolver.inference.acquisition import grad_score
from adasolver.inference.base import SamplingResult


class ElasticityCPG:
    """Coarse-prediction-guided refinement for one Elasticity point cloud."""

    def __init__(self, reference_coords: np.ndarray, coarse_fraction: float = 0.10,
                 patch_count: int = 48,
                 kappa_by_budget: dict[float, float] | None = None,
                 beta_by_budget: dict[float, float] | None = None, *, legacy: bool = False):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        if self.reference_coords.ndim != 2 or self.reference_coords.shape[1] != 2:
            raise ValueError("reference_coords must have shape (N, 2)")
        self.coarse_fraction = coarse_fraction
        self.legacy = legacy
        if not legacy:
            from adasolver.inference.elasticity_contract import coarse_count
            coarse_count(len(self.reference_coords), coarse_fraction)
            from adasolver.inference.elasticity_density import ElasticityDensity
            self.density_builder = ElasticityDensity(self.reference_coords)
        self.patch_count = patch_count
        self.kappa_by_budget = kappa_by_budget or {0.375: 8.0, 0.5: 4.0, 0.75: 4.0}
        self.beta_by_budget = beta_by_budget or {0.375: 0.5, 0.5: 1.0, 0.75: 1.0}
        self.patch_id, self.patch_mass = adaptive.build_patches(
            self.reference_coords, self.patch_count, seed=0,
        )

    def coarse_count(self) -> int:
        if self.legacy:
            return max(2, int(round(len(self.reference_coords) * self.coarse_fraction)))
        from adasolver.inference.elasticity_contract import coarse_count
        return coarse_count(len(self.reference_coords), self.coarse_fraction)

    def refine(self, coarse_indices: np.ndarray, coarse_prediction: np.ndarray,
               final_count: int, budget: float, rng: np.random.Generator) -> SamplingResult:
        score = grad_score(
            self.reference_coords[np.asarray(coarse_indices)], coarse_prediction, k=8,
        )
        kappa = self.kappa_by_budget[budget]
        beta = self.beta_by_budget[budget]
        if not self.legacy:
            from adasolver.inference.elasticity_density import DensityConfig
            config = DensityConfig(coarse_fraction=self.coarse_fraction, patches=self.patch_count,
                                   kappa=kappa, beta=beta, placement="fps")
            result = self.density_builder.refine(coarse_indices, score, final_count, config, rng)
            result.metadata.update(method="cpg", beta=beta, placement="fps", patch_count=self.patch_count)
            return result
        refined, info = adaptive.refine_once(
            np.asarray(coarse_indices), score, self.reference_coords,
            self.patch_id, self.patch_mass, final_count, kappa=kappa,
            seed=int(rng.integers(1 << 30)),
        )
        compensation = adaptive.comp_weights_for_subset(
            refined, self.reference_coords, len(self.reference_coords), beta=beta,
        )
        density = np.power(compensation.astype(np.float64), -1.0 / beta)
        result = SamplingResult(refined, density, compensation, {
            "method": "cpg",
            "coarse_fraction": self.coarse_fraction,
            "coarse_count": self.coarse_count(),
            "patch_count": self.patch_count,
            "kappa": kappa,
            "beta": beta,
            "placement": "fps",
            **info,
        })
        result.validate(len(self.reference_coords), final_count)
        return result
