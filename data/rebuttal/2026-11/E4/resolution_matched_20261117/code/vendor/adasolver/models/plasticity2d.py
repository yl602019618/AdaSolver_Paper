from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .adapter import Transolver3Adapter
from .checkpoint import sha256_file
from .legacy_transolver import LegacyPlasticityTransolver


@dataclass(frozen=True)
class PlasticityModelSpec:
    architecture: str = "Transolver3"
    space_dim: int = 2
    fun_dim: int = 1
    out_dim: int = 38
    hidden_dim: int = 128
    heads: int = 8
    layers: int = 8
    mlp_ratio: int = 1
    slices: int = 64
    dropout: float = 0.0
    activation: str = "gelu"
    ref: int = 8
    temporal_mode: str = "single_forward_all_frames"
    predicted_frames: int = 19
    displacement_channels: int = 2
    frame_start: int = 1
    target_layout: str = "frame_major_channel_minor"

    def build(self) -> Transolver3Adapter:
        return Transolver3Adapter(
            space_dim=self.space_dim, fun_dim=self.fun_dim, out_dim=self.out_dim,
            hidden_dim=self.hidden_dim, heads=self.heads, layers=self.layers,
            mlp_ratio=self.mlp_ratio, slices=self.slices, dropout=self.dropout,
            activation=self.activation, ref=self.ref,
        )


def load_legacy_plasticity2d_operator(checkpoint: str | Path, manifest: str | Path,
                                      device: torch.device | str = "cpu") -> LegacyPlasticityTransolver:
    """Load the historical Transolver allT checkpoint in a legacy-only namespace."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("benchmark") != "plasticity2d" or metadata.get("variant") != "all_timestep":
        raise ValueError("not a Plasticity all-timestep manifest")
    expected_hash = metadata.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"Plasticity checkpoint hash mismatch: {checkpoint_path}")
    model_values = metadata.get("model")
    if not isinstance(model_values, dict) or model_values.get("architecture") != "Transolver":
        raise ValueError("legacy manifest must declare architecture=Transolver")
    model = LegacyPlasticityTransolver(
        space_dim=model_values["space_dim"], fun_dim=model_values["fun_dim"],
        out_dim=model_values["out_dim"], hidden_dim=model_values["hidden_dim"],
        heads=model_values["heads"], layers=model_values["layers"],
        mlp_ratio=model_values["mlp_ratio"], slices=model_values["slices"],
        dropout=model_values.get("dropout", 0.0),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


_REQUIRED_MANIFEST = {
    "benchmark": "plasticity2d",
    "variant": "all_timestep",
    "nfull": 3131,
    "frame_start": 1,
    "predicted_frames": 19,
    "displacement_channels": 2,
    "target_layout": "frame_major_channel_minor",
}


def _validate_manifest(metadata: dict[str, Any], checkpoint_path: Path) -> PlasticityModelSpec:
    required = dict(_REQUIRED_MANIFEST)
    if metadata.get("task_version") == "full20_v1":
        required.update(variant="official_full20", frame_start=0, predicted_frames=20)
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Plasticity checkpoint manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    model_values = metadata.get("model")
    if not isinstance(model_values, dict):
        raise ValueError("Plasticity checkpoint manifest must contain a model object")
    spec = PlasticityModelSpec(**model_values)
    if spec.out_dim != spec.predicted_frames * spec.displacement_channels:
        raise ValueError("Plasticity model out_dim does not match temporal layout")
    for key in ("frame_start", "predicted_frames", "displacement_channels", "target_layout"):
        if getattr(spec, key) != required[key]:
            raise ValueError(f"model and manifest disagree on {key}")
    if metadata.get("task_version") == "full20_v1":
        normalizer = Path(metadata["input_normalizer"])
        if sha256_file(normalizer) != metadata.get("input_normalizer_sha256"):
            raise ValueError("Plasticity input normalizer hash mismatch")
    expected_hash = metadata.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"Plasticity checkpoint hash mismatch: {checkpoint_path}")
    return spec


def load_plasticity2d_operator(checkpoint: str | Path, manifest: str | Path,
                               device: torch.device | str = "cpu") -> Transolver3Adapter:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = _validate_manifest(metadata, checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("Plasticity checkpoint must be a model state_dict")
    model = spec.build()
    model.load_state_dict(state, strict=True)
    model.plasticity_manifest = metadata
    return model.to(device).eval()
