from __future__ import annotations

import numpy as np

from adasolver.inference import adaptive, adaptive_fast
from adasolver.inference.base import SamplingResult


class PipeCA:
    def __init__(self, reference_coords: np.ndarray, coarse_fraction: float = 0.10,
                 patch_count: int = 24, kappa_by_budget: dict[float, float] | None = None,
                 beta_by_budget: dict[float, float] | None = None,
                 placement: str = "fps"):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        self.coarse_fraction = coarse_fraction
        self.patch_count = patch_count
        self.kappa_by_budget = kappa_by_budget or {0.375: 4.0, 0.5: 4.0, 0.75: 8.0}
        self.beta_by_budget = beta_by_budget or {0.375: 1.0, 0.5: 1.0, 0.75: 1.0}
        if placement not in ("random", "fps"):
            raise ValueError(f"unsupported CA placement: {placement}")
        self.placement = placement
        self.patch_id, self.patch_mass = adaptive.build_patches(
            self.reference_coords, self.patch_count, seed=0)
        self.patch_cache = adaptive_fast.build_patch_cache(
            self.reference_coords, self.patch_id, self.patch_count)

    def refine(self, coarse_indices: np.ndarray, learned_score: np.ndarray,
               final_count: int, budget: float, rng: np.random.Generator) -> SamplingResult:
        kappa = self.kappa_by_budget[budget]
        beta = self.beta_by_budget[budget]
        refined, info = adaptive_fast.refine_once(
            coarse_indices, learned_score, self.reference_coords, self.patch_id,
            self.patch_mass, final_count, kappa=kappa,
            seed=int(rng.integers(1 << 30)), patch_cache=self.patch_cache,
            placement=self.placement)
        compensation = adaptive.comp_weights_for_subset(
            refined, self.reference_coords, len(self.reference_coords), beta=beta)
        density = np.power(compensation.astype(np.float64), -1.0 / beta)
        result = SamplingResult(refined, density, compensation, {
            "method": "ca", "coarse_fraction": self.coarse_fraction,
            "patch_count": self.patch_count, "kappa": kappa, "beta": beta,
            "placement": self.placement, **info,
        })
        result.validate(len(self.reference_coords), final_count)
        return result
