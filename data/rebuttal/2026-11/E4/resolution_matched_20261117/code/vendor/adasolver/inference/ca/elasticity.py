from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adasolver.inference import adaptive, adaptive_fast
from adasolver.inference.base import SamplingResult
from adasolver.models.checkpoint import sha256_file

from .head import TransolverGradientHead


class ElasticityCAHead(nn.Module):
    target_mean = 6.94226598739624
    target_std = 0.9406828284263611

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.operator.eval()
        self.head = TransolverGradientHead()

    @torch.no_grad()
    def features(self, pos: torch.Tensor, fx: torch.Tensor | None) -> torch.Tensor:
        net = self.operator.net
        values = torch.cat((pos, fx), dim=-1) if fx is not None else pos
        hidden = net.preprocess(values) + net.placeholder[None, None, :]
        return net.blocks[0](hidden)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None) -> torch.Tensor:
        return self.head(self.features(pos, fx))

    @classmethod
    def to_gradient_score(cls, standardized: np.ndarray) -> np.ndarray:
        log_gradient = np.clip(
            standardized * cls.target_std + cls.target_mean, 0.0, None,
        )
        return np.expm1(log_gradient) ** 2


def load_elasticity_ca_head(operator, checkpoint: str | Path, manifest: str | Path,
                            device: torch.device | str = "cpu") -> ElasticityCAHead:
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    required = {"benchmark": "elasticity", "method": "ca", "architecture": "transolver",
                "blocks": 2, "parameter_count": 236049}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Elasticity CA manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    path = Path(checkpoint).expanduser().resolve()
    if sha256_file(path) != metadata.get("head_sha256"):
        raise ValueError(f"Elasticity CA head hash mismatch: {path}")
    model = ElasticityCAHead(operator).to(device).eval()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.head.load_state_dict(state, strict=True)
    return model


class ElasticityCA:
    def __init__(self, reference_coords: np.ndarray, coarse_fraction: float = 0.10,
                 patch_count: int = 48, kappa: float = 8.0,
                 beta: float = 0.75, placement: str = "random", *, legacy: bool = False):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        self.coarse_fraction = coarse_fraction
        self.legacy = legacy
        if not legacy:
            from adasolver.inference.elasticity_contract import coarse_count
            coarse_count(len(self.reference_coords), coarse_fraction)
            from adasolver.inference.elasticity_density import ElasticityDensity
            self.density_builder = ElasticityDensity(self.reference_coords)
        self.patch_count = patch_count
        self.kappa = kappa
        self.beta = beta
        self.placement = placement
        self.patch_id, self.patch_mass = adaptive.build_patches(
            self.reference_coords, self.patch_count, seed=0,
        )
        self.patch_cache = adaptive_fast.build_patch_cache(
            self.reference_coords, self.patch_id, self.patch_count,
        )

    def coarse_count(self) -> int:
        if self.legacy:
            return max(2, int(round(len(self.reference_coords) * self.coarse_fraction)))
        from adasolver.inference.elasticity_contract import coarse_count
        return coarse_count(len(self.reference_coords), self.coarse_fraction)

    def refine(self, coarse_indices: np.ndarray, learned_score: np.ndarray,
               final_count: int, rng: np.random.Generator) -> SamplingResult:
        if not self.legacy:
            from adasolver.inference.elasticity_density import DensityConfig
            config = DensityConfig(coarse_fraction=self.coarse_fraction, patches=self.patch_count,
                                   kappa=self.kappa, beta=self.beta, placement=self.placement)
            result = self.density_builder.refine(coarse_indices, learned_score, final_count, config, rng)
            result.metadata.update(method="ca", beta=self.beta, placement=self.placement, patch_count=self.patch_count)
            return result
        refined, info = adaptive_fast.refine_once(
            coarse_indices, learned_score, self.reference_coords, self.patch_id,
            self.patch_mass, final_count, kappa=self.kappa,
            seed=int(rng.integers(1 << 30)), patch_cache=self.patch_cache,
            placement=self.placement,
        )
        compensation = adaptive.comp_weights_for_subset(
            refined, self.reference_coords, len(self.reference_coords), beta=self.beta,
        )
        density = np.power(compensation.astype(np.float64), -1.0 / self.beta)
        result = SamplingResult(refined, density, compensation, {
            "method": "ca", "coarse_fraction": self.coarse_fraction,
            "coarse_count": self.coarse_count(), "patch_count": self.patch_count,
            "kappa": self.kappa, "beta": self.beta, "placement": self.placement,
            **info,
        })
        result.validate(len(self.reference_coords), final_count)
        return result
