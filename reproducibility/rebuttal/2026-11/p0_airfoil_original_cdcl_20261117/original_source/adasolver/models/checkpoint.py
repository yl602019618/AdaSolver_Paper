from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .adapter import Transolver3Adapter, Transolver3ModelSpec
from .legacy_darcy import LegacyDarcyTransolver


ELASTICITY_LEGACY_MANIFEST = Path(__file__).resolve().parents[2] / "checkpoints/elasticity/legacy_t3_manifest.json"
DARCY_LEGACY_MANIFEST = Path(__file__).resolve().parents[2] / "checkpoints/darcy/legacy_manifest.json"

PIPE_LEGACY_SHA256 = "2e2dd73afc767e7bd1fe97e5e7bdb490b9032e8214b270ebfad774f6ecdf1b39"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_operator(checkpoint: str | Path, model_spec: Transolver3ModelSpec,
                  device: torch.device | str = "cpu",
                  expected_sha256: str | None = None) -> Transolver3Adapter:
    """Strictly load a Transolver3 operator with an explicit model contract."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint must contain a model state_dict")
    model = model_spec.build()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_elasticity_operator(checkpoint: str | Path, manifest: str | Path = ELASTICITY_LEGACY_MANIFEST,
                             device: torch.device | str = "cpu") -> Transolver3Adapter:
    """Strictly load an Elasticity Transolver3 operator from its manifest."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"benchmark": "elasticity", "nfull": 972}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Elasticity manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    model_values = metadata.get("model")
    if not isinstance(model_values, dict):
        raise ValueError("Elasticity manifest must contain a model object")
    spec = Transolver3ModelSpec(**model_values)
    if spec.fun_dim != 0 or spec.out_dim != 1:
        raise ValueError("Elasticity operator must use fun_dim=0 and out_dim=1")
    expected_hash = metadata.get("checkpoint_sha256")
    return load_operator(checkpoint_path, spec, device, expected_hash)


def load_airfoil2d_operator(checkpoint: str | Path, manifest: str | Path,
                             device: torch.device | str = "cpu") -> Transolver3Adapter:
    """Load an Airfoil pressure operator only after manifest validation."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"benchmark": "airfoil2d", "target_name": "pressure", "target_channel": 3,
                "nfull": 11271}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Airfoil checkpoint manifest {key}={metadata.get(key)!r}, expected {expected!r}")
    fraction = metadata.get("nmin_fraction")
    if fraction not in (0.4, 1.0) or metadata.get("nmin_count") != round(11271 * fraction):
        raise ValueError("Airfoil pressure checkpoint must declare consistent variable-point or full-mesh training counts")
    expected_hash = metadata.get("checkpoint_sha256")
    if expected_hash and sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"Airfoil checkpoint hash mismatch: {checkpoint_path}")
    model = Transolver3Adapter()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_shapenet_car_operator(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
    expected_sha256: str | None = None,
) -> Transolver3Adapter:
    """Strictly load the audited 3/7/4 ShapeNet-Car Transolver3 operator."""
    spec = Transolver3ModelSpec(
        space_dim=3,
        fun_dim=7,
        out_dim=4,
        hidden_dim=256,
        heads=8,
        layers=8,
        mlp_ratio=2,
        slices=32,
    )
    return load_operator(checkpoint, spec, device, expected_sha256)
def load_pipe_operator(checkpoint: str | Path, device: torch.device | str = "cpu",
                       verify_legacy_hash: bool = False) -> Transolver3Adapter:
    expected_hash = PIPE_LEGACY_SHA256 if verify_legacy_hash else None
    return load_operator(checkpoint, Transolver3ModelSpec(), device, expected_hash)


def load_darcy_operator(checkpoint: str | Path,
                        manifest: str | Path = DARCY_LEGACY_MANIFEST,
                        device: torch.device | str = "cpu") -> Transolver3Adapter:
    """Strictly load either a legacy or paper-matched Darcy operator."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("benchmark") != "darcy":
        raise ValueError(f"Darcy manifest benchmark={metadata.get('benchmark')!r}")
    model_values = metadata.get("model")
    if not isinstance(model_values, dict):
        raise ValueError("Darcy manifest must contain a model object")
    spec = Transolver3ModelSpec(**model_values)
    if spec.space_dim != 2 or spec.fun_dim != 1 or spec.out_dim != 1:
        raise ValueError("Darcy operator must use space_dim=2, fun_dim=1, out_dim=1")
    expected_hash = metadata.get("checkpoint_sha256")
    if expected_hash is not None and sha256_file(checkpoint_path) != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    role = metadata.get("role")
    if role == "migration_only":
        required = {"nfull": 7225, "nmin": 2168, "nmin_fraction": 0.3}
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Darcy manifest {key}={metadata.get(key)!r}, expected {expected!r}")
        model = LegacyDarcyTransolver(
            space_dim=spec.space_dim, fun_dim=spec.fun_dim, out_dim=spec.out_dim,
            n_hidden=spec.hidden_dim, n_heads=spec.heads, n_layers=spec.layers,
            mlp_ratio=spec.mlp_ratio, slice_num=spec.slices, dropout=spec.dropout,
            act=spec.activation,
        )
    elif metadata.get("status") == "official":
        required = {"nfull": 7225, "train_samples": 1000,
                    "optimization_samples": 1000, "test_used_for_selection": True}
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Darcy manifest {key}={metadata.get(key)!r}, expected {expected!r}")
        fraction = metadata.get("nmin_fraction")
        count = metadata.get("nmin_count")
        if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError(f"invalid Darcy nmin_fraction: {fraction!r}")
        expected_count = int(round(float(fraction) * 7225))
        if count != expected_count:
            raise ValueError(f"Darcy manifest nmin_count={count!r}, expected {expected_count!r}")
        model = spec.build()
    else:
        raise ValueError(f"unsupported Darcy manifest role/status: {role!r}/{metadata.get('status')!r}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint must contain a model state_dict")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
