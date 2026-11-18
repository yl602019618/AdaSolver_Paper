from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from scipy.interpolate import griddata
from torch import nn
from scipy.spatial import cKDTree

from adasolver.inference import adaptive
from adasolver.inference.acquisition import grad_score
from adasolver.inference.base import SamplingResult
from adasolver.models.checkpoint import sha256_file

from .head import TransolverGradientHead


class DarcyCAHead(nn.Module):
    """Learned coarse acquisition head over a frozen Darcy operator."""

    def __init__(self, operator: nn.Module, target_mean: float = 0.0,
                 target_std: float = 1.0, blocks: int = 2):
        super().__init__()
        self.operator = operator
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.operator.eval()
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.head = TransolverGradientHead(
            hidden_dim=128, heads=8, mlp_ratio=2, slices=64, blocks=blocks,
        )

    @torch.no_grad()
    def features(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        if pos.ndim != 3 or fx.ndim != 3 or pos.shape[:2] != fx.shape[:2]:
            raise ValueError("pos and fx must align on batch and point dimensions")
        net = self.operator.net
        hidden = net.preprocess(torch.cat((pos, fx), dim=-1))
        hidden = hidden + net.placeholder[None, None, :]
        return net.blocks[0](hidden)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pos, fx))

    def decode_score(self, standardized: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized, dtype=np.float64)
        log_score = values * self.target_std + self.target_mean
        return np.expm1(np.clip(log_score, 0.0, None))


def load_darcy_ca_head(
    operator: nn.Module, checkpoint: str | Path, manifest: str | Path,
    device: torch.device | str = "cpu",
) -> DarcyCAHead:
    """Load a CA head only after validating its operator and manifest contract."""
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    required = {"benchmark": "darcy", "method": "ca", "blocks": 2,
                "architecture": "transolver", "target": "zscore_log1p_grad_lap"}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Darcy CA manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    operator_path = Path(metadata["operator"]).expanduser().resolve()
    if metadata.get("operator_sha256") != sha256_file(operator_path):
        raise ValueError("Darcy CA operator hash mismatch")
    if metadata.get("operator_manifest_sha256"):
        manifest_path = Path(metadata["operator_manifest"]).expanduser().resolve()
        if metadata["operator_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError("Darcy CA operator manifest hash mismatch")
    path = Path(checkpoint).expanduser().resolve()
    if metadata.get("head_sha256") != sha256_file(path):
        raise ValueError("Darcy CA head hash mismatch")
    model = DarcyCAHead(
        operator, float(metadata["target_mean"]), float(metadata["target_std"]),
        blocks=int(metadata["blocks"]),
    ).to(device).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.head.load_state_dict(state, strict=True)
    return model


def _lift_coarse_score(coarse_indices: np.ndarray, score: np.ndarray,
                       reference_coords: np.ndarray) -> np.ndarray:
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    reference_coords = np.asarray(reference_coords, dtype=np.float64)
    if len(coarse_indices) != len(score):
        raise ValueError("coarse indices and score must align")
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("CA score must be finite and nonnegative")
    nearest = cKDTree(reference_coords[coarse_indices]).query(reference_coords, k=1)[1]
    return np.maximum(score[nearest], 1e-12)


class DarcyCA:
    """Coarse learned acquisition followed by CPG-style patch refinement."""

    def __init__(self, reference_coords: np.ndarray, head: DarcyCAHead,
                 patch_count: int = 32, coarse_fraction: float = 0.10,
                 kappa: float = 2.0, beta: float = 1.0):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        if self.reference_coords.ndim != 2 or self.reference_coords.shape[1] != 2:
            raise ValueError("reference_coords must have shape (N, 2)")
        if not 0 < coarse_fraction < 0.375 or patch_count < 1 or kappa < 1 or beta < 0:
            raise ValueError("invalid Darcy CA parameters")
        self.head = head.eval()
        self.patch_count = patch_count
        self.coarse_fraction = coarse_fraction
        self.kappa = kappa
        self.beta = beta
        self.patch_id, self.patch_mass = adaptive.build_patches(
            self.reference_coords, patch_count, seed=0,
        )

    def coarse_count(self) -> int:
        return max(8, int(round(len(self.reference_coords) * self.coarse_fraction)))

    @torch.no_grad()
    def acquire(self, coarse_indices: np.ndarray, coarse_pos: torch.Tensor,
                 coarse_fx: torch.Tensor) -> np.ndarray:
        standardized = self.head(coarse_pos, coarse_fx).detach().cpu().numpy().reshape(-1)
        return self.head.decode_score(standardized)

    def refine(self, coarse_indices: np.ndarray, coarse_pos: torch.Tensor,
               coarse_fx: torch.Tensor, final_count: int,
               rng: np.random.Generator) -> SamplingResult:
        coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
        if len(np.unique(coarse_indices)) != len(coarse_indices):
            raise ValueError("coarse indices must be unique")
        if final_count <= len(coarse_indices) or final_count > len(self.reference_coords):
            raise ValueError("Darcy CA budget must exceed coarse count and fit lattice")
        score = self.acquire(coarse_indices, coarse_pos, coarse_fx)
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
            "method": "ca", "architecture": "transolver_gradient_head",
            "coarse_fraction": self.coarse_fraction, "coarse_count": len(coarse_indices),
            "patch_count": self.patch_count, "kappa": self.kappa, "beta": self.beta,
            "acquisition": "learned_grad_lap", "placement": "fps",
            "truth_access": False, **info,
        })
        result.validate(len(self.reference_coords), final_count)
        return result
