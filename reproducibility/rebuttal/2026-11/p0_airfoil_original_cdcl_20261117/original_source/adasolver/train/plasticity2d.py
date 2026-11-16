from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import Plasticity2DAdapter, unflatten_temporal_field
from adasolver.models import Transolver3ModelSpec
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/plas/plas_N987_T20.mat"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_mean_loss(prediction: torch.Tensor, target: torch.Tensor,
                    frames: int = 19, channels: int = 2) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != frames * channels:
        raise ValueError("Plasticity allT prediction/target layout mismatch")
    pred = prediction.reshape(prediction.shape[0], prediction.shape[1], frames, channels)
    truth = target.reshape(target.shape[0], target.shape[1], frames, channels)
    numerator = torch.linalg.vector_norm(pred - truth, dim=(1, 3))
    denominator = torch.linalg.vector_norm(truth, dim=(1, 3)).clamp_min(1e-12)
    return (numerator / denominator).mean(dim=1).mean()


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[float]]:
    model.eval()
    values = []
    for pos, fx, target in loader:
        prediction = model(pos.to(device), fx.to(device))
        pred = prediction.reshape(prediction.shape[0], prediction.shape[1], 19, 2)
        truth = target.to(device).reshape(target.shape[0], target.shape[1], 19, 2)
        num = torch.linalg.vector_norm(pred - truth, dim=(1, 3))
        den = torch.linalg.vector_norm(truth, dim=(1, 3)).clamp_min(1e-12)
        values.append(num / den)
    per_frame = torch.cat(values, dim=0)
    return float(per_frame.mean()), per_frame.mean(dim=0).cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/plasticity2d"))
    parser.add_argument("--name", default="plasticity2d_transolver3_allt_nmin040")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--nmin-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.nmin_fraction <= 1:
        raise ValueError("epochs/batch-size/nmin-fraction are invalid")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("Plasticity official training requires CUDA")
    if args.smoke:
        args.epochs = 1
        args.name += "_smoke"

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    adapter = Plasticity2DAdapter(args.data_path)
    data = adapter.load()
    train_split = data.splits["train"]
    validation_split = data.splits["unused"]
    if len(validation_split.sample_ids) == 0:
        raise ValueError("validation split must not be empty")
    full_count = adapter.full_point_count()
    nmin = int(round(args.nmin_fraction * full_count))
    if args.nmin_fraction == 0.4 and nmin != 1252:
        raise RuntimeError(f"unexpected Plasticity nmin rounding: {nmin}")

    train_loader = DataLoader(TensorDataset(train_split.pos, train_split.fx, train_split.target),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(TensorDataset(validation_split.pos, validation_split.fx,
                                                 validation_split.target),
                                   batch_size=args.batch_size, shuffle=False)
    model_spec = Transolver3ModelSpec(space_dim=2, fun_dim=1, out_dim=38,
                                      hidden_dim=128, heads=8, layers=8,
                                      mlp_ratio=1, slices=64)
    model = model_spec.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/plasticity2d") / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_path = args.output_dir / f"{args.name}_best.pt"
    with log_path.open("w", encoding="utf-8") as log:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        emit(f"start name={args.name} epochs={args.epochs} train={len(train_split.sample_ids)} "
             f"validation={len(validation_split.sample_ids)} N=[{nmin},{full_count}] seed={args.seed}")
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for pos, fx, target in train_loader:
                count = int(torch.randint(nmin, full_count + 1, (1,), generator=generator).item())
                selected = torch.randperm(full_count, generator=generator)[:count]
                prediction = model(pos[:, selected].to(device), fx[:, selected].to(device))
                loss = frame_mean_loss(prediction, target[:, selected].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * pos.shape[0]
            validation, per_frame = validate(model, validation_loader, device)
            emit(f"epoch={epoch} train_frame_mean={total / len(train_split.sample_ids):.8f} "
                 f"validation_frame_mean={validation:.8f} per_frame_mean={json.dumps(per_frame)}" +
                 (" best" if validation < best else ""))
            if validation < best:
                best = validation
                torch.save(model.state_dict(), best_path)
    last_path = args.output_dir / f"{args.name}_last.pt"
    torch.save(model.state_dict(), last_path)
    normalizer_path = args.output_dir / f"{args.name}_input_norm.npz"
    np.savez(normalizer_path, mean=data.input_normalizer.mean.cpu().numpy(),
             std=data.input_normalizer.std.cpu().numpy())
    manifest = {
        "benchmark": "plasticity2d", "variant": "official_all_timestep",
        "status": "smoke" if args.smoke else "official",
        "temporal_mode": "single_forward_all_frames", "frame_start": 1,
        "predicted_frames": 19, "displacement_channels": 2,
        "target_layout": data.layout.order, "target_normalization": "none",
        "input_normalization": "train_global_channel_standardization",
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "data_split": {"train": [0, 900], "validation": [900, 907],
                       "test": "pending_user_freeze"},
        "nfull": full_count, "nmin_fraction": args.nmin_fraction, "nmin_count": nmin,
        "epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "scheduler": {"name": "OneCycleLR", "pct_start": args.pct_start},
        "max_grad_norm": args.max_grad_norm, "loss": "mean_per_frame_relative_l2",
        "model": model_spec.__dict__, "best_validation_frame_mean": best,
        "checkpoint": str(best_path.resolve()), "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()), "last_checkpoint_sha256": sha256_file(last_path),
        "input_normalizer": str(normalizer_path.resolve()),
        "input_normalizer_sha256": sha256_file(normalizer_path),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "visible_devices": args.gpu},
        "command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
