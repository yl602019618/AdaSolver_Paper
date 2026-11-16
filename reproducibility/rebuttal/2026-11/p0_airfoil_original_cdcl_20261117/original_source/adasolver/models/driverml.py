"""Local DriverML Transolver-3 wrapper, compatible with the frozen legacy weights."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from .transolver3 import Model


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DriverMLModel(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = Model(space_dim=6, fun_dim=0, out_dim=out_dim, n_hidden=256,
                         n_layers=8, n_head=8, slice_num=32, mlp_ratio=2,
                         dropout=0, act="gelu", unified_pos=False)
        # The public wrapper also uses the dense-equivalent global chunk path.
        # Import lazily because the compensation layer imports the backbone.
        from adasolver.layers.compensation import install_compensation
        install_compensation(self)

    def forward(self, features, chunk_size=0, use_checkpoint=False):
        if features.ndim != 3 or features.shape[-1] != 6:
            raise ValueError("DriverML features must have shape (batch, points, 6)")
        if chunk_size and features.shape[1] > chunk_size:
            chunks = list(features.split(chunk_size, dim=1))
            return torch.cat(self.net(chunks, use_checkpoint=use_checkpoint, input_list=True), dim=1)
        return self.net(features, use_checkpoint=False, input_list=False)


def build_driver_model(out_dim: int, device: str | torch.device = "cpu") -> torch.nn.Module:
    if out_dim not in (1, 3):
        raise ValueError("DriverML output dimension must be 1 (pressure) or 3 (WSS)")
    model = DriverMLModel(out_dim).to(device)
    model.eval()
    return model


def load_driver_checkpoint(checkpoint: str | Path, out_dim: int,
                           device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint)
    model = build_driver_model(out_dim, device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"DriverML checkpoint must contain a model mapping: {path}")
    state = {key.removeprefix("module."): value for key, value in payload["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    meta = {"checkpoint": str(path.resolve()), "sha256": file_sha256(path),
            "out_dim": out_dim, "architecture": {"dim": 256, "n_layers": 8,
            "heads": 8, "slice_num": 32, "mlp_ratio": 2, "space_dim": 6},
            "epoch": payload.get("epoch"), "val_rel_l2": payload.get("val_rel_l2"),
            "tensor_count": len(state), "training_args": payload.get("args", {}),
            "role": "reused_pretrained_operator", "operator_retraining": False,
            "source": "adasolver.models.driverml.DriverMLModel"}
    meta["training_args"] = {key: str(value) if isinstance(value, Path) else value
                             for key, value in meta["training_args"].items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, meta
