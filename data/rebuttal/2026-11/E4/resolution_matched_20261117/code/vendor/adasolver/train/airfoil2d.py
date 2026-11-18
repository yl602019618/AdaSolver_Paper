from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import Airfoil2DAdapter
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3Adapter
from adasolver.train.pipe import variable_point_subset
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/shuaibinqu/NACA/data/naca"


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def validate(model, loader, normalizer, device: torch.device) -> float:
    model.eval()
    values = []
    for pos, target in loader:
        pos, target = pos.to(device), target.to(device)
        prediction = normalizer.decode(model(pos, pos))
        values.append(relative_l2(prediction, target).cpu())
    return float(torch.cat(values).mean())


def build_manifest(args, best_validation: float, nmin: int, full_count: int,
                   checkpoint_sha256: str) -> dict:
    return {
        "benchmark": "airfoil2d", "status": "official" if not args.smoke else "smoke",
        "architecture": "Transolver3", "target_name": "pressure", "target_channel": 3,
        "epochs": args.epochs, "nmin_fraction": args.nmin_fraction, "nmin_count": nmin,
        "nfull": full_count, "seed": args.seed, "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate, "weight_decay": args.weight_decay},
        "scheduler": {"name": "OneCycleLR", "pct_start": args.pct_start},
        "max_grad_norm": args.max_grad_norm, "train_samples": args.train_samples,
        "validation_samples": args.validation_samples, "data_path": str(Path(args.data_path).resolve()),
        "data_split": {"train": [0, 1000], "test": [1000, 1200]},
        "shape": [221, 51], "mandatory_surface": {"i_start": 50, "i_stop": 171, "j": 0},
        "best_validation_relative_l2": best_validation, "checkpoint_sha256": checkpoint_sha256,
        "git_revision_at_start": args.git_revision, "command": args.command,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/airfoil2d"))
    parser.add_argument("--name", default="airfoil2d_transolver3_pressure_nmin040")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
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
    args.command, args.git_revision = " ".join(os.sys.argv), git_revision()
    if args.smoke:
        args.epochs, args.train_samples, args.validation_samples = 1, min(args.train_samples, 8), min(args.validation_samples, 4)
        args.name += "_smoke"
    generator = seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Airfoil training requires CUDA; use the project GPU environment")
    device = torch.device("cuda:0")
    data = Airfoil2DAdapter(args.data_path, ntrain=args.train_samples,
                            ntest=args.validation_samples).load()
    full_count = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * full_count))
    if args.nmin_fraction == 0.4 and nmin != 4508:
        raise RuntimeError(f"unexpected Airfoil nmin rounding: {nmin}")
    train_loader = DataLoader(TensorDataset(data.train.pos, data.train.target),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(TensorDataset(data.test.pos, data.test.target), batch_size=args.batch_size)
    model = Transolver3Adapter().to(device)
    data.y_normalizer.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                     epochs=args.epochs, steps_per_epoch=len(train_loader),
                                                     pct_start=args.pct_start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best, best_path = float("inf"), args.output_dir / f"{args.name}_best.pt"
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for pos, target in train_loader:
            selected = variable_point_subset(full_count, nmin, generator)
            prediction = model(pos[:, selected].to(device), pos[:, selected].to(device))
            loss = relative_l2(prediction, target[:, selected].to(device)).sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); total += float(loss.detach())
        validation = validate(model, validation_loader, data.y_normalizer, device)
        print(f"epoch={epoch} train_relative_l2={total / len(data.train.sample_ids):.8f} validation_relative_l2={validation:.8f}", flush=True)
        if validation < best:
            best = validation; torch.save(model.state_dict(), best_path)
    last_path = args.output_dir / f"{args.name}_last.pt"
    torch.save(model.state_dict(), last_path)
    np.savez(args.output_dir / f"{args.name}_ynorm.npz", mean=data.y_normalizer.mean.cpu().numpy(), std=data.y_normalizer.std.cpu().numpy())
    manifest = build_manifest(args, best, nmin, full_count, sha256_file(best_path))
    (args.output_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
