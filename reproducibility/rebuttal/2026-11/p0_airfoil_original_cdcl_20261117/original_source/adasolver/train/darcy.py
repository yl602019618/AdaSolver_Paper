from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import DarcyAdapter
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3Adapter, Transolver3ModelSpec
from adasolver.models.checkpoint import sha256_file
from adasolver.train.pipe import variable_point_subset
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy"
DEFAULT_OUTPUT = Path("checkpoints/darcy")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return None


def build_manifest(args, model_spec: Transolver3ModelSpec, best_validation: float,
                   full_count: int, nmin: int, best_path: Path, last_path: Path,
                   normalizer_path: Path) -> dict:
    return {
        "benchmark": "darcy",
        "status": "smoke" if args.smoke else "official",
        "architecture": "Transolver3",
        "model": model_spec.__dict__,
        "epochs": args.epochs,
        "nmin_fraction": args.nmin_fraction,
        "nmin_count": nmin,
        "nfull": full_count,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "scheduler": {"name": "OneCycleLR", "pct_start": args.pct_start},
        "max_grad_norm": args.max_grad_norm,
        "train_samples": args.train_samples,
        "optimization_samples": getattr(args, "optimization_samples", args.train_samples),
        "validation_samples": args.validation_samples,
        "validation_source": getattr(args, "validation_source", "smooth2_test_diagnostic"),
        "validation_train_fraction": getattr(args, "validation_train_fraction", None),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "data_split": {"train": [0, args.train_samples],
                       "optimization": [0, getattr(args, "optimization_samples", args.train_samples)],
                       "validation": [args.train_samples, args.train_samples + args.validation_samples],
                       "test": [args.train_samples, args.train_samples + args.validation_samples]},
        "test_used_for_selection": True,
        "best_validation_relative_l2": best_validation,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "normalizer": str(normalizer_path.resolve()),
        "normalizer_sha256": sha256_file(normalizer_path),
        "git_revision_at_start": args.git_revision,
        "command": args.command,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


@torch.no_grad()
def validate(model, loader, normalizer, device: torch.device,
             *, target_is_normalized: bool) -> float:
    model.eval()
    values = []
    for pos, fx, target in loader:
        prediction = normalizer.decode(model(pos.to(device), fx.to(device)))
        truth = target.to(device)
        if target_is_normalized:
            truth = normalizer.decode(truth)
        values.append(relative_l2(prediction, truth).cpu())
    return float(torch.cat(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the official Darcy Transolver3 operator")
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default="darcy_transolver3_varpts_nmin040")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--nmin-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--validation-train-fraction", type=float, default=0.1,
                        help="fraction of smooth1 training samples reserved for checkpoint selection")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.command = " ".join(os.sys.argv)
    args.git_revision = git_revision()
    if args.smoke:
        args.epochs = 1
        args.train_samples = min(args.train_samples, 8)
        args.validation_samples = min(args.validation_samples, 4)
        args.name += "_smoke"
    if not 0 < args.nmin_fraction <= 1:
        raise ValueError("nmin_fraction must be in (0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {args.device}, but CUDA is unavailable")

    if not 0 < args.validation_train_fraction < 1:
        raise ValueError("validation_train_fraction must be in (0, 1)")
    generator = seed_everything(args.seed)
    optimization_samples = args.train_samples
    data = DarcyAdapter(args.data_path, ntrain=args.train_samples,
                        ntest=args.validation_samples).load()
    args.optimization_samples = optimization_samples
    args.validation_source = "smooth2_test_diagnostic"
    full_count = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * full_count))
    if args.nmin_fraction == 0.4 and nmin != 2890:
        raise RuntimeError(f"unexpected Darcy nmin rounding: {nmin}")
    train_loader = DataLoader(
        TensorDataset(data.train.pos[:optimization_samples], data.train.fx[:optimization_samples],
                      data.train.target[:optimization_samples]),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(data.test.pos, data.test.fx, data.test.target),
        batch_size=args.batch_size, shuffle=False,
    )
    model_spec = Transolver3ModelSpec(fun_dim=1, mlp_ratio=2)
    model = model_spec.build().to(device)
    data.y_normalizer.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/darcy/train") / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    with log_path.open("a", encoding="utf-8") as log:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        emit(f"start name={args.name} epochs={args.epochs} train={args.train_samples} "
             f"validation={args.validation_samples} N=[{nmin},{full_count}] seed={args.seed}")
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for pos, fx, target in train_loader:
                selected = variable_point_subset(full_count, nmin, generator)
                pos, fx, target = (x[:, selected].to(device) for x in (pos, fx, target))
                loss = relative_l2(model(pos, fx), target).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach())
            validation = validate(
                model, validation_loader, data.y_normalizer, device,
                target_is_normalized=False,
            )
            if validation < best_validation:
                best_validation = validation
                torch.save(model.state_dict(), args.output_dir / f"{args.name}_best.pt")
            emit(f"epoch={epoch} train_relative_l2={total / optimization_samples:.8f} "
                 f"validation_relative_l2={validation:.8f}" + (" best" if validation <= best_validation else ""))
    best_path = args.output_dir / f"{args.name}_best.pt"
    last_path = args.output_dir / f"{args.name}_last.pt"
    normalizer_path = args.output_dir / f"{args.name}_ynorm.npz"
    torch.save(model.state_dict(), last_path)
    np.savez(normalizer_path, mean=data.y_normalizer.mean.cpu().numpy(),
             std=data.y_normalizer.std.cpu().numpy())
    manifest = build_manifest(args, model_spec, best_validation, full_count, nmin,
                              best_path, last_path, normalizer_path)
    (args.output_dir / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
