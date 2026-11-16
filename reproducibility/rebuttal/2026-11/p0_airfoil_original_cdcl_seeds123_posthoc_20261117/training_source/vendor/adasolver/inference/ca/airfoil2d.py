from __future__ import annotations

"""Learned coarse-to-adaptive sampling for the Airfoil2D pressure task.

The head is deliberately separate from the pressure operator checkpoint.  It predicts a
standardized ``log1p`` pressure-gradient magnitude from frozen block-0 features; no target
values are needed at inference time.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from adasolver.inference import adaptive, adaptive_fast
from adasolver.inference.base import SamplingResult
from adasolver.models.checkpoint import sha256_file
from adasolver.inference.airfoil2d_contract import coarse_count, validate_coarse_indices
from adasolver.layers.block0_cache import coarse_features

from .head import TransolverGradientHead


@dataclass(frozen=True)
class Airfoil2dCATargetStats:
    mean: float
    std: float
    target: str = "zscore_log1p_pressure_gradient"

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("Airfoil CA target statistics must be finite with positive std")

    def decode(self, standardized: np.ndarray) -> np.ndarray:
        value = np.asarray(standardized, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError("Airfoil CA head output must be finite")
        log_score = value * self.std + self.mean
        # The supervised target is already a gradient magnitude; undo only the
        # log1p transform, without applying an additional power.
        return np.maximum(np.expm1(np.maximum(log_score, 0.0)), 0.0)


class Airfoil2dCAHead(nn.Module):
    """CA head over frozen features of the official pressure operator."""

    def __init__(self, operator: nn.Module, stats: Airfoil2dCATargetStats,
                 blocks: int = 2, hidden_dim: int = 128, heads: int = 8,
                 mlp_ratio: int = 2, slices: int = 64):
        super().__init__()
        self.operator = operator
        self.stats = stats
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.operator.eval()
        self.head = TransolverGradientHead(hidden_dim=hidden_dim, heads=heads,
                                           mlp_ratio=mlp_ratio, slices=slices,
                                           blocks=blocks)

    @torch.no_grad()
    def features(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        if pos.ndim != 3:
            raise ValueError("pos must have shape (batch, points, 2)")
        if fx is None:
            fx = pos
        if fx.ndim != 3 or fx.shape[:2] != pos.shape[:2]:
            raise ValueError("fx must align with pos on batch and point dimensions")
        return coarse_features(self.operator, pos, fx)[0]

    def forward_with_cache(self, pos, fx=None, *, capture=False):
        features, cache = coarse_features(self.operator, pos, fx, capture=capture)
        return self.head(features), cache

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.features(pos, fx))

    def decode_score(self, standardized: np.ndarray) -> np.ndarray:
        return self.stats.decode(standardized)

    @classmethod
    def to_gradient_score(cls, standardized: np.ndarray, mean: float,
                          std: float) -> np.ndarray:
        return Airfoil2dCATargetStats(mean, std).decode(standardized)


class PressureMLP(nn.Module):
    """One pressure output, with an explicit small feature bottleneck."""

    def __init__(self, feature_dim=8, hidden_dim=128):
        super().__init__()
        if feature_dim not in (8, 16, 32, 128):
            raise ValueError("unsupported pressure head bottleneck")
        self.layers = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.GELU(),
                                    nn.Linear(feature_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.layers(features)


class Airfoil2dPressureHead(Airfoil2dCAHead):
    def __init__(self, operator, mean, std, feature_dim=8):
        nn.Module.__init__(self)
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("invalid pressure normalization")
        self.operator = operator.eval()
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.pressure_mean, self.pressure_std = float(mean), float(std)
        self.feature_dim = feature_dim
        self.head = PressureMLP(feature_dim)

    def decode_pressure(self, standardized):
        values = np.asarray(standardized, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("pressure head output must be finite")
        return values * self.pressure_std + self.pressure_mean

    def decode_score(self, standardized):
        raise ValueError("pressure head needs a pressure-derived acquisition, not gradient decoding")


def lift_head_score(coarse_pos: np.ndarray, coarse_score: np.ndarray,
                    reference: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lift from coarse scores to a sample's full coordinate grid."""
    coarse_pos = np.asarray(coarse_pos, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    score = np.asarray(coarse_score, dtype=np.float64).reshape(-1)
    if coarse_pos.shape != (len(score), 2) or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("Airfoil CA score coordinates have invalid shapes")
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("Airfoil CA score must be finite and nonnegative")
    return np.maximum(score[cKDTree(coarse_pos).query(reference, k=1)[1]], 1e-12)


def _airfoil_compensation(sel: np.ndarray, reference: np.ndarray,
                          beta: float) -> np.ndarray:
    """Compute Voronoi compensation while splitting mass at duplicate grid coordinates."""
    selected = np.asarray(sel, dtype=np.int64)
    tree = cKDTree(reference[selected])
    _, nearest = tree.query(reference, k=1)
    raw_mass = np.bincount(nearest, minlength=len(selected)).astype(np.float64)
    raw_mass /= raw_mass.sum()
    _, inverse, counts = np.unique(
        reference[selected], axis=0, return_inverse=True, return_counts=True
    )
    mass = raw_mass.copy()
    for group in np.flatnonzero(counts > 1):
        count = counts[group]
        members = np.flatnonzero(inverse == group)
        if count > 1:
            mass[members] = raw_mass[members].sum() / count
    mean_mass = mass.mean()
    compensation = np.power(mass / (mean_mass + 1e-12), beta)
    return compensation.astype(np.float32)


def load_airfoil2d_ca_head(operator: nn.Module, checkpoint: str | Path,
                           manifest: str | Path, device: torch.device | str = "cpu",
                           operator_checkpoint: str | Path | None = None) -> Airfoil2dCAHead:
    """Strictly load a pressure-compatible head and independently verify its manifest."""
    manifest_path = Path(manifest).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else checkpoint_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"benchmark": "airfoil2d", "method": "ca",
                "target_channel": 3, "nfull": 11271, "truth_access": False}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Airfoil CA manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    pressure_head = metadata.get("target") == "zscore_pressure"
    if metadata.get("target") not in ("zscore_pressure", "zscore_log1p_pressure_gradient",
                                      "zscore(log1p(local_lsq_abs_grad_pressure))"):
        raise ValueError("unsupported Airfoil CA target encoding")
    expected_architecture = "pressure_mlp" if pressure_head else "transolver"
    if metadata.get("architecture", "transolver").lower() != expected_architecture:
        raise ValueError("unsupported Airfoil CA head architecture")
    expected_head = metadata.get("head_sha256")
    if expected_head and sha256_file(checkpoint_path) != expected_head:
        raise ValueError(f"Airfoil CA head hash mismatch: {checkpoint_path}")
    if operator_checkpoint is not None and metadata.get("operator_sha256", metadata.get("operator_checkpoint_sha256")):
        if sha256_file(operator_checkpoint) != metadata.get("operator_sha256", metadata.get("operator_checkpoint_sha256")):
            raise ValueError("Airfoil CA operator checkpoint hash mismatch")
    if pressure_head:
        model = Airfoil2dPressureHead(operator, metadata["target_mean"], metadata["target_std"],
                                     int(metadata["feature_dim"])).to(device).eval()
    else:
        stats = Airfoil2dCATargetStats(float(metadata["target_mean"]), float(metadata["target_std"]))
        model = Airfoil2dCAHead(operator, stats, blocks=int(metadata.get("blocks", 2)),
                            hidden_dim=int(metadata.get("hidden_dim", 128)),
                            heads=int(metadata.get("heads", 8)),
                            mlp_ratio=int(metadata.get("mlp_ratio", 2)),
                            slices=int(metadata.get("slices", 64))).to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.head.load_state_dict(state, strict=True)
    return model


class Airfoil2dCA:
    def __init__(self, reference_coords: np.ndarray, mandatory_indices: np.ndarray | None = None,
                 coarse_fraction: float = .10, patch_count: int = 48,
                 kappa_by_budget: dict[float, float] | None = None,
                 beta_by_budget: dict[float, float] | None = None, placement: str = "fps"):
        self.reference_coords = np.asarray(reference_coords, dtype=np.float64)
        if self.reference_coords.ndim != 2 or self.reference_coords.shape[1] != 2:
            raise ValueError("reference_coords must have shape (N, 2)")
        if not 0 < coarse_fraction <= .10 or patch_count < 1:
            raise ValueError("invalid Airfoil CA sampling configuration")
        if placement not in ("random", "fps"):
            raise ValueError(f"unsupported CA placement: {placement}")
        self.mandatory_indices = np.array([], dtype=np.int64) if mandatory_indices is None else np.unique(np.asarray(mandatory_indices, dtype=np.int64))
        if np.any(self.mandatory_indices < 0) or np.any(self.mandatory_indices >= len(self.reference_coords)):
            raise ValueError("mandatory index outside reference grid")
        coarse_count(len(self.reference_coords), coarse_fraction, len(self.mandatory_indices))
        self.coarse_fraction, self.patch_count, self.placement = coarse_fraction, patch_count, placement
        self.kappa_by_budget = kappa_by_budget or {.375: 16., .5: 32., .75: 8.}
        self.beta_by_budget = beta_by_budget or {.375: 1., .5: 1., .75: 1.}
        self.patch_id, self.patch_mass = adaptive.build_patches(self.reference_coords, patch_count, seed=0)
        self.patch_cache = adaptive_fast.build_patch_cache(self.reference_coords, self.patch_id, patch_count)

    def coarse_count(self) -> int:
        return coarse_count(len(self.reference_coords), self.coarse_fraction, len(self.mandatory_indices))

    def refine(self, coarse_indices: np.ndarray, learned_score: np.ndarray, final_count: int,
               budget: float, rng: np.random.Generator) -> SamplingResult:
        coarse_input = np.asarray(coarse_indices, dtype=np.int64).reshape(-1)
        if len(np.unique(coarse_input)) != len(coarse_input):
            raise ValueError("coarse CA indices must not contain duplicates")
        if np.any(coarse_input < 0) or np.any(coarse_input >= len(self.reference_coords)):
            raise ValueError("coarse CA index outside reference grid")
        coarse = validate_coarse_indices(coarse_input, len(self.reference_coords), self.mandatory_indices)
        if not set(self.mandatory_indices).issubset(set(coarse)):
            raise ValueError("coarse CA set must contain all mandatory surface points")
        if final_count < len(coarse) or final_count > len(self.reference_coords):
            raise ValueError("invalid Airfoil CA final count")
        score = np.asarray(learned_score, dtype=np.float64).reshape(-1)
        if len(score) == len(coarse):
            # The head output is paired with the caller's index order.  Convert it
            # to a full-grid vector before the refinement kernel indexes by IDs.
            score_full = np.zeros(len(self.reference_coords), dtype=np.float64)
            score_full[coarse] = score
        elif len(score) == len(self.reference_coords):
            score_full = score
        else:
            raise ValueError("Airfoil CA score must align with coarse or full-grid points")
        if not np.isfinite(score_full).all() or np.any(score_full < 0):
            raise ValueError("Airfoil CA score must be finite and nonnegative")
        kappa, beta = self.kappa_by_budget[budget], self.beta_by_budget[budget]
        refined, info = adaptive_fast.refine_once(coarse, score_full, self.reference_coords,
            self.patch_id, self.patch_mass, final_count, kappa=kappa,
            seed=int(rng.integers(1 << 30)), patch_cache=self.patch_cache, placement=self.placement)
        compensation = _airfoil_compensation(refined, self.reference_coords, beta)
        density = (np.power(compensation.astype(np.float64), -1. / beta) if beta > 0
                   else np.ones(len(refined), dtype=np.float64))
        result = SamplingResult(refined, density, compensation, {"method": "ca",
            "benchmark": "airfoil2d", "truth_access": False, "coarse_count": len(coarse),
            "coarse_fraction": self.coarse_fraction, "patch_count": self.patch_count,
            "kappa": kappa, "beta": beta, "placement": self.placement, **info})
        result.validate(len(self.reference_coords), final_count)
        if not set(self.mandatory_indices).issubset(set(refined)):
            raise ValueError("CA refinement dropped mandatory surface points")
        return result
