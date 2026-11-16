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

from adasolver.data import PipeAdapter
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3Adapter
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def variable_point_subset(full_count: int, minimum: int,
                          generator: torch.Generator) -> torch.Tensor:
    count = int(torch.randint(minimum, full_count + 1, (1,), generator=generator).item())
    return torch.randperm(full_count, generator=generator)[:count]


def summed_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return relative_l2(prediction, target).sum()


@torch.no_grad()
def validate(model, loader, normalizer, device: torch.device) -> float:
    model.eval()
    values = []
    for pos, target in loader:
        pos = pos.to(device)
        target = target.to(device)
        prediction = normalizer.decode(model(pos, pos))
        values.append(relative_l2(prediction, target).cpu())
    return float(torch.cat(values).mean())


def build_manifest(args, best_validation: float, nmin: int, full_count: int) -> dict:
    return {
        "benchmark": "pipe",
        "status": "official" if not args.smoke else "smoke",
        "architecture": "Transolver3",
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
        "validation_samples": args.validation_samples,
        "data_path": str(Path(args.data_path).resolve()),
        "data_split": {"train": [0, 1000], "test": [1000, 1200]},
        "best_validation_relative_l2": best_validation,
        "git_revision_at_start": args.git_revision,
        "command": args.command,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pipe"))
    parser.add_argument("--name", default="pipe_transolver3_varpts_nmin040")
    parser.add_argument("--gpu", default="1,2")
    parser.add_argument("--num-gpus", type=int, default=2)
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.command = " ".join(os.sys.argv)
    args.git_revision = git_revision()

    if args.smoke:
        args.epochs = 1
        args.train_samples = min(args.train_samples, 8)
        args.validation_samples = min(args.validation_samples, 4)
        args.name += "_smoke"
    if not 0 < args.nmin_fraction <= 1:
        raise ValueError("nmin_fraction must be in (0, 1]")

    generator = seed_everything(args.seed)
    device = torch.device("cuda:0")
    data = PipeAdapter(args.data_path).load()
    full_count = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * full_count))
    if nmin != 6656 and args.nmin_fraction == 0.4:
        raise RuntimeError(f"unexpected Pipe nmin rounding: {nmin}")

    train_dataset = TensorDataset(data.train.pos[:args.train_samples],
                                  data.train.target[:args.train_samples])
    validation_dataset = TensorDataset(data.test.pos[:args.validation_samples],
                                       data.test.target[:args.validation_samples])
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              generator=loader_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = Transolver3Adapter().to(device)
    train_model = model
    if args.num_gpus > 1:
        if torch.cuda.device_count() < args.num_gpus:
            raise RuntimeError(f"requested {args.num_gpus} GPUs, found {torch.cuda.device_count()}")
        train_model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    data.y_normalizer.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/pipe/train") / f"{args.name}.log"
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
            train_model.train()
            total = 0.0
            for pos, target in train_loader:
                selected = variable_point_subset(full_count, nmin, generator)
                pos = pos[:, selected].to(device)
                target = target[:, selected].to(device)
                prediction = train_model(pos, pos)
                loss = summed_relative_l2(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach())
            validation = validate(model, validation_loader, data.y_normalizer, device)
            is_best = validation < best_validation
            if is_best:
                best_validation = validation
                torch.save(model.state_dict(), args.output_dir / f"{args.name}_best.pt")
            emit(f"epoch={epoch} train_relative_l2={total / args.train_samples:.8f} "
                 f"validation_relative_l2={validation:.8f}" + (" best" if is_best else ""))

    torch.save(model.state_dict(), args.output_dir / f"{args.name}_last.pt")
    np.savez(args.output_dir / f"{args.name}_ynorm.npz",
             mean=data.y_normalizer.mean.cpu().numpy(), std=data.y_normalizer.std.cpu().numpy())
    manifest = build_manifest(args, best_validation, nmin, full_count)
    (args.output_dir / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
