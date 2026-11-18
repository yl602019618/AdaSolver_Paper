from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from adasolver.data import PlasticityLayout, unflatten_temporal_field
from adasolver.inference.base import SamplingResult
from adasolver.models.checkpoint import sha256_file
from adasolver.models.legacy_transolver import LegacyPlasticityTransolver

from .head import TransolverGradientHead


@dataclass(frozen=True)
class PlasticityCATargetStats:
    mean: float
    std: float
    target: str = "zscore_log1p_frame_mean_displacement_gradient"


def temporal_gradient_target(target: np.ndarray, shape: tuple[int, int] = (101, 31),
                             layout: PlasticityLayout = PlasticityLayout()) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    if values.ndim == 2 and values.shape[-1] == layout.out_dim:
        values = unflatten_temporal_field(values, layout)
    if values.shape != (shape[0] * shape[1], layout.predicted_frames, layout.displacement_channels):
        raise ValueError(f"unexpected temporal target shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("temporal target must be finite")
    field = values.reshape(shape[0], shape[1], layout.predicted_frames, layout.displacement_channels)
    score = np.zeros(shape, dtype=np.float64)
    for frame in range(layout.predicted_frames):
        frame_score = np.zeros(shape, dtype=np.float64)
        for channel in range(layout.displacement_channels):
            grad_i, grad_j = np.gradient(field[:, :, frame, channel])
            frame_score += grad_i ** 2 + grad_j ** 2
        score += np.sqrt(frame_score)
    return (score / layout.predicted_frames).reshape(-1)


def fit_ca_target_stats(scores: np.ndarray, epsilon: float = 1e-8) -> PlasticityCATargetStats:
    values = np.log1p(np.asarray(scores, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("CA target scores must be finite")
    return PlasticityCATargetStats(float(values.mean()), float(values.std() + epsilon))


def encode_ca_target(scores: np.ndarray, stats: PlasticityCATargetStats) -> np.ndarray:
    return (np.log1p(np.asarray(scores, dtype=np.float64)) - stats.mean) / stats.std


class PlasticityCAHead(nn.Module):
    """Plasticity-specific head over frozen legacy operator block-0 features."""

    def __init__(self, operator: LegacyPlasticityTransolver,
                 stats: PlasticityCATargetStats, blocks: int = 2):
        super().__init__()
        self.operator = operator
        self.stats = stats
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.operator.eval()
        self.head = TransolverGradientHead(hidden_dim=128, heads=8, mlp_ratio=1,
                                           slices=64, blocks=blocks)

    @torch.no_grad()
    def features(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        if pos.ndim != 3 or fx.ndim != 3 or pos.shape[:2] != fx.shape[:2]:
            raise ValueError("pos and fx must align on batch and point dimensions")
        hidden = self.operator.preprocess(torch.cat((pos, fx), dim=-1))
        hidden = hidden + self.operator.placeholder[None, None, :]
        return self.operator.blocks[0](hidden)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pos, fx))

    def decode_score(self, standardized: np.ndarray) -> np.ndarray:
        log_score = np.asarray(standardized, dtype=np.float64) * self.stats.std + self.stats.mean
        return np.expm1(np.clip(log_score, 0.0, None))


def lift_head_score(coarse_pos: np.ndarray, coarse_score: np.ndarray,
                    reference: np.ndarray) -> np.ndarray:
    coarse_pos = np.asarray(coarse_pos, dtype=np.float64)
    coarse_score = np.asarray(coarse_score, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64)
    if coarse_pos.shape != (len(coarse_score), 2) or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("CA coarse score coordinates have invalid shapes")
    if not np.isfinite(coarse_score).all() or np.any(coarse_score < 0):
        raise ValueError("CA coarse score must be finite and nonnegative")
    lifted = coarse_score[cKDTree(coarse_pos).query(reference, k=1)[1]]
    return np.maximum(lifted, 1e-12)


def sample_plasticity_ca(coarse_indices: np.ndarray, density: np.ndarray,
                         count: int, rng: np.random.Generator,
                         beta: float = 1.0) -> SamplingResult:
    density = np.asarray(density, dtype=np.float64)
    coarse_indices = np.unique(np.asarray(coarse_indices, dtype=np.int64))
    full_count = len(density)
    if count <= len(coarse_indices) or count > full_count:
        raise ValueError("CA count must exceed coarse count and fit the lattice")
    if not np.isfinite(density).all() or np.any(density <= 0) or beta < 0:
        raise ValueError("CA density must be positive and finite")
    available = np.setdiff1d(np.arange(full_count, dtype=np.int64), coarse_indices)
    probability = density[available] / density[available].sum()
    extra = rng.choice(available, count - len(coarse_indices), replace=False, p=probability)
    indices = np.concatenate((coarse_indices, extra))
    base_fraction = len(coarse_indices) / full_count
    inclusion = base_fraction + (count - len(coarse_indices)) * density / density[available].sum()
    inclusion = np.clip(inclusion, 1e-12, None)
    compensation = (1.0 / (inclusion[indices] / base_fraction)) ** beta
    result = SamplingResult(indices, inclusion[indices], compensation, {
        "method": "ca", "prior": "learned_temporal_gradient_head",
        "coarse_count": int(len(coarse_indices)), "beta": beta,
        "truth_access": False,
    })
    result.validate(full_count, count)
    return result


def load_plasticity_ca_head(operator: LegacyPlasticityTransolver,
                            checkpoint: str | Path, manifest: str | Path,
                            device: torch.device | str = "cpu") -> PlasticityCAHead:
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    required = {"benchmark": "plasticity2d", "method": "ca",
                "operator_variant": "legacy_all_timestep", "blocks": 2,
                "target": "zscore_log1p_frame_mean_displacement_gradient"}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Plasticity CA manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    path = Path(checkpoint).expanduser().resolve()
    if sha256_file(path) != metadata.get("head_sha256"):
        raise ValueError(f"Plasticity CA head hash mismatch: {path}")
    stats = PlasticityCATargetStats(float(metadata["target_mean"]), float(metadata["target_std"]))
    model = PlasticityCAHead(operator, stats, blocks=2).to(device).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.head.load_state_dict(state, strict=True)
    return model
