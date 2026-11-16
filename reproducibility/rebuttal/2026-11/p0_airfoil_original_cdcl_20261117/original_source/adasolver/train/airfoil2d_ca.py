from __future__ import annotations

"""Train the Airfoil2D learned pressure-gradient acquisition head."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data.airfoil2d import Airfoil2DAdapter
from adasolver.inference.ca.head import TransolverGradientHead
from adasolver.models.checkpoint import load_airfoil2d_operator, sha256_file
from adasolver.utils import seed_everything
from adasolver.inference.airfoil2d_contract import coarse_count, validate_coarse_count
from adasolver.layers.block0_cache import coarse_features

DEFAULT_DATA = "/mnt/ai-material-data/shuaibinqu/NACA/data/naca"


def gradient_magnitude_lsq(pos: np.ndarray, values: np.ndarray, k: int = 16,
                           ridge_fraction: float = 1e-3,
                           clip_percentile: float | None = 99.5) -> np.ndarray:
    """Estimate pressure-gradient magnitude using local weighted least squares."""
    pos = np.asarray(pos, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if pos.ndim != 2 or values.ndim != 2 or len(pos) != len(values):
        raise ValueError("pos and values must be aligned 2-D arrays")
    if k < 1 or len(pos) < 2:
        raise ValueError("gradient target requires at least two points and k >= 1")
    kq = min(int(k) + 1, len(pos))
    dist, indices = cKDTree(pos).query(pos, k=kq)
    offsets = pos[indices[:, 1:]] - pos[:, None, :]
    differences = values[indices[:, 1:]] - values[:, None, :]
    radius = np.maximum(np.median(dist[:, 1:], axis=1, keepdims=True), 1e-9)
    scaled = offsets / radius[..., None]
    weights = np.exp(-(dist[:, 1:] / radius) ** 2)
    weighted_offsets = scaled * weights[..., None]
    matrix = np.einsum("nmd,nme->nde", weighted_offsets, scaled)
    rhs = np.einsum("nmd,nmc->ndc", weighted_offsets, differences)
    trace = np.einsum("nii->n", matrix) / pos.shape[1]
    matrix += (ridge_fraction * trace + 1e-12)[:, None, None] * np.eye(pos.shape[1])
    gradient = np.linalg.solve(matrix, rhs) / radius[..., None]
    magnitude = np.sqrt((gradient ** 2).sum(axis=1)).sum(axis=1)
    if clip_percentile is not None and clip_percentile < 100:
        magnitude = np.minimum(magnitude, np.percentile(magnitude, clip_percentile))
    return magnitude.astype(np.float32)


def build_gradient_targets(coords: torch.Tensor, pressure: torch.Tensor, *, k: int = 16,
                           ridge_fraction: float = 1e-3,
                           clip_percentile: float | None = 99.5) -> np.ndarray:
    x, y = coords.detach().cpu().numpy(), pressure.detach().cpu().numpy()
    if y.ndim == 2:
        y = y[..., None]
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("coords must be (S,N,2), pressure must be (S,N,C)")
    return np.stack([
        gradient_magnitude_lsq(x[s], y[s], k, ridge_fraction, clip_percentile)
        for s in range(x.shape[0])
    ])


def target_statistics(raw_targets: np.ndarray) -> tuple[float, float]:
    raw = np.asarray(raw_targets, dtype=np.float64)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError("gradient targets must be finite and nonnegative")
    transformed = np.log1p(raw)
    return float(transformed.mean()), float(max(transformed.std(), 1e-8))


def sample_head_indices(nfull: int, count: int, mandatory: np.ndarray,
                        generator: torch.Generator) -> torch.Tensor:
    """Draw a deployment-shaped subset while retaining the Airfoil surface."""
    required = np.asarray(mandatory, dtype=np.int64).reshape(-1)
    if len(np.unique(required)) != len(required):
        raise ValueError("mandatory head-training indices must be unique")
    if count < len(required) or count > nfull:
        raise ValueError("head subset count must contain mandatory points and fit the grid")
    validate_coarse_count(count, nfull)
    mask = torch.ones(nfull, dtype=torch.bool)
    mandatory_tensor = torch.as_tensor(required, dtype=torch.long)
    mask[mandatory_tensor] = False
    available = torch.arange(nfull, dtype=torch.long)[mask]
    extra = available[torch.randperm(len(available), generator=generator)[:count - len(required)]]
    return torch.cat((mandatory_tensor, extra))


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _features(operator, pos: torch.Tensor) -> torch.Tensor:
    return coarse_features(operator, pos, pos)[0]


def evaluate_validation_huber(head: torch.nn.Module, operator: torch.nn.Module,
                              loader: DataLoader, device: torch.device, *,
                              fraction: float = .10, mandatory=None, seed: int = 29000) -> float:
    """Return point-weighted held-out Huber loss without updating either model."""
    if len(loader.dataset) < 1:
        raise ValueError("validation loader must contain at least one sample")
    head.eval()
    total, elements = 0.0, 0
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for pos, target in loader:
            if mandatory is not None:
                count = coarse_count(pos.shape[1], fraction, len(mandatory))
                indices = sample_head_indices(pos.shape[1], count, mandatory, generator)
                pos, target = pos[:, indices], target[:, indices]
            pos, target = pos.to(device), target.to(device)
            prediction = head(_features(operator, pos))
            total += float(F.smooth_l1_loss(prediction, target, reduction="sum"))
            elements += target.numel()
    if elements < 1:
        raise ValueError("validation loader produced no target elements")
    return total / elements


def validation_checkpoint_decision(value: float, best: float, min_delta: float) -> bool:
    if not np.isfinite(value):
        raise ValueError("validation loss must be finite")
    return value < best - min_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--operator-manifest", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/airfoil2d"))
    parser.add_argument("--name", default="airfoil2d_ca_head_pressure")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--nmin-fraction", type=float, default=0.4,
                        help="operator training floor recorded in the manifest")
    parser.add_argument("--head-min-fraction", type=float, default=0.10)
    parser.add_argument("--head-max-fraction", type=float, default=0.10)
    parser.add_argument("--target-k", type=int, default=16)
    parser.add_argument("--target-clip-percentile", type=float, default=99.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-train-samples", type=int, default=900)
    parser.add_argument("--head-validation-samples", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.name = 1, args.name + "_smoke"
        args.head_train_samples = min(args.head_train_samples, 8)
        args.head_validation_samples = min(args.head_validation_samples, 4)
        args.patience = max(args.patience, 1)
    if args.epochs < 1 or args.patience < 1 or args.min_delta < 0:
        raise ValueError("epochs and patience must be positive and min_delta nonnegative")
    if not 0 < args.head_min_fraction <= args.head_max_fraction <= .10:
        raise ValueError("head subset fractions must satisfy 0 < min <= max <= 0.10")
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = Airfoil2DAdapter(args.data_path, ntest=0).load()
    nfull = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * nfull))
    head_min = coarse_count(nfull, args.head_min_fraction, len(data.mandatory_indices))
    head_max = coarse_count(nfull, args.head_max_fraction, len(data.mandatory_indices))
    if nfull != 11271 or nmin != 4508:
        raise ValueError(f"unexpected current Airfoil pressure contract: N={nfull}, nmin={nmin}")
    train_samples = min(int(args.head_train_samples), len(data.train.pos))
    validation_samples = min(
        int(args.head_validation_samples), max(len(data.train.pos) - train_samples, 0)
    )
    if train_samples < 1:
        raise ValueError("head_train_samples must select at least one sample")
    if validation_samples < 1:
        raise ValueError("head_validation_samples must select a nonempty held-out split")
    pressure = data.y_normalizer.decode(data.train.target[:train_samples])
    raw_target = build_gradient_targets(
        data.train.pos[:train_samples], pressure, k=args.target_k,
        clip_percentile=args.target_clip_percentile,
    )
    mean, std = target_statistics(raw_target)
    target = torch.from_numpy(
        ((np.log1p(raw_target) - mean) / std).astype(np.float32)
    ).unsqueeze(-1)
    loader = DataLoader(
        TensorDataset(data.train.pos[:train_samples], target),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_pos = data.train.pos[train_samples:train_samples + validation_samples]
    validation_pressure = data.y_normalizer.decode(
        data.train.target[train_samples:train_samples + validation_samples]
    )
    validation_raw = build_gradient_targets(
        validation_pos, validation_pressure, k=args.target_k,
        clip_percentile=args.target_clip_percentile,
    ) if validation_samples else None
    validation_target = torch.from_numpy(
        ((np.log1p(validation_raw) - mean) / std).astype(np.float32)
    ).unsqueeze(-1) if validation_raw is not None else None
    validation_loader = DataLoader(
        TensorDataset(validation_pos, validation_target), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    operator_hash_before = sha256_file(args.operator)
    operator = load_airfoil2d_operator(args.operator, args.operator_manifest, device=device)
    operator.eval()
    for parameter in operator.parameters():
        parameter.requires_grad_(False)
    head = TransolverGradientHead(blocks=2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(loader), pct_start=args.pct_start,
    )
    point_generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{args.name}_best.pt"
    last_path = args.output_dir / f"{args.name}_last.pt"
    manifest_path = args.output_dir / f"{args.name}_manifest.json"
    existing = [path for path in (best_path, last_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing CA head artifacts: {existing}")
    operator_state_before = {
        name: tensor.detach().cpu().clone() for name, tensor in operator.state_dict().items()
    }
    best_validation, best_epoch, stale_epochs = float("inf"), None, 0
    history: list[dict[str, float | int]] = []
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        head.train()
        total, elements = 0.0, 0
        for pos, target_batch in loader:
            count = int(torch.randint(head_min, head_max + 1, (1,), generator=point_generator).item())
            selected = sample_head_indices(nfull, count, data.mandatory_indices, point_generator)
            pos, target_batch = pos[:, selected].to(device), target_batch[:, selected].to(device)
            with torch.no_grad():
                features = _features(operator, pos)
            prediction = head(features)
            loss = F.smooth_l1_loss(prediction, target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total += float(F.smooth_l1_loss(prediction.detach(), target_batch, reduction="sum"))
            elements += target_batch.numel()
        train_huber = total / elements
        validation_huber = evaluate_validation_huber(
            head, operator, validation_loader, device, fraction=args.head_max_fraction,
            mandatory=data.mandatory_indices,
        )
        history.append({"epoch": epoch, "train_huber": train_huber,
                        "validation_huber": validation_huber,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        print(f"epoch={epoch} train_huber={train_huber:.8f} "
              f"validation_huber={validation_huber:.8f}", flush=True)
        if validation_checkpoint_decision(validation_huber, best_validation, args.min_delta):
            best_validation, best_epoch, stale_epochs = validation_huber, epoch, 0
            torch.save(head.state_dict(), best_path)
        else:
            stale_epochs += 1
            if not args.disable_early_stopping and stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
                break
    training_seconds = time.perf_counter() - training_started
    torch.save(head.state_dict(), last_path)
    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("training did not produce a finite validation-selected checkpoint")
    operator_unchanged = all(
        torch.equal(operator_state_before[name], tensor.detach().cpu())
        for name, tensor in operator.state_dict().items()
    )
    operator_hash_after = sha256_file(args.operator)
    if not operator_unchanged or operator_hash_after != operator_hash_before:
        raise RuntimeError("frozen pressure operator changed during CA-head training")
    manifest = {
        "benchmark": "airfoil2d", "method": "ca",
        "status": "smoke" if args.smoke else "trained_head",
        "target_name": "pressure_gradient_magnitude", "target_channel": 3,
        "target": "zscore(log1p(local_lsq_abs_grad_pressure))",
        "target_k": args.target_k, "target_clip_percentile": args.target_clip_percentile,
        "target_mean": mean, "target_std": std,
        "architecture": "transolver", "blocks": 2,
        "parameter_count": sum(p.numel() for p in head.parameters()),
        "nfull": nfull, "nmin_fraction": args.nmin_fraction, "nmin_count": nmin,
        "head_min_fraction": args.head_min_fraction, "head_max_fraction": args.head_max_fraction,
        "head_min_count": head_min, "head_max_count": head_max,
        "head_subset_sampling": "mandatory_surface_plus_uniform_without_replacement",
        "mandatory_surface_count": int(len(data.mandatory_indices)),
        "operator_checkpoint": str(Path(args.operator).resolve()),
        "operator_manifest": str(Path(args.operator_manifest).resolve()),
        "operator_sha256": operator_hash_before, "head_sha256": sha256_file(best_path),
        "operator_frozen": True, "operator_unchanged_after_training": operator_unchanged,
        "data_path": str(Path(args.data_path).resolve()),
        "data_split": {
            "head_train": [0, train_samples],
            "head_validation": [train_samples, train_samples + validation_samples],
            "test": "not_accessed",
        },
        "truth_access": False, "seed": args.seed,
        "epochs_requested": args.epochs, "epochs_completed": len(history),
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "patience": args.patience, "min_delta": args.min_delta,
        "early_stopping_enabled": not args.disable_early_stopping,
        "checkpoint_selection": "minimum held-out sparse validation Huber loss",
        "best_epoch": best_epoch, "best_validation_huber": best_validation,
        "last_train_huber": history[-1]["train_huber"],
        "last_validation_huber": history[-1]["validation_huber"],
        "training_history": history, "training_seconds": training_seconds,
        "git_revision": _git_revision(), "python": platform.python_version(),
        "torch": torch.__version__, "command": " ".join(__import__("sys").argv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
