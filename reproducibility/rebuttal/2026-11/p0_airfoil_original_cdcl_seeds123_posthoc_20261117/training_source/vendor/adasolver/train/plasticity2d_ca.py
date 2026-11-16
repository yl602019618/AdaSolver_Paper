from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import Plasticity2DAdapter, unflatten_temporal_field
from adasolver.inference.ca.plasticity2d import (
    PlasticityCATargetStats,
    PlasticityCAHead,
    fit_ca_target_stats,
    temporal_gradient_target,
)
from adasolver.models import load_legacy_plasticity2d_operator
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/plas/plas_N987_T20.mat"
DEFAULT_OPERATOR = "/mnt/ai-material-data/zhijun/CarD/Adasolver_official/checkpoints/plas_allT_Transolver_varpts_best.pt"
DEFAULT_MANIFEST = "checkpoints/plasticity2d/legacy_allt_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--operator-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/plasticity2d"))
    parser.add_argument("--name", default="legacy_allt_ca_head")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.name += "_smoke"
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    adapter = Plasticity2DAdapter(args.data_path)
    data = adapter.load()
    operator = load_legacy_plasticity2d_operator(args.operator, args.operator_manifest, device=device)
    train_split = data.splits["train"]
    calibration_split = data.splits["unused"]
    train_targets = unflatten_temporal_field(train_split.target, data.layout).numpy()
    calibration_targets = unflatten_temporal_field(calibration_split.target, data.layout).numpy()
    train_score = np.stack([temporal_gradient_target(value, data.shape, data.layout)
                             for value in train_targets])
    calibration_score = np.stack([temporal_gradient_target(value, data.shape, data.layout)
                                  for value in calibration_targets])
    stats = fit_ca_target_stats(train_score)
    train_y = torch.from_numpy(((np.log1p(train_score) - stats.mean) / stats.std)[..., None]).float()
    calibration_y = torch.from_numpy(((np.log1p(calibration_score) - stats.mean) / stats.std)[..., None]).float()

    head_model = PlasticityCAHead(operator, stats).to(device)
    train_loader = DataLoader(TensorDataset(train_split.pos, train_split.fx, train_y),
                              batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    calibration_loader = DataLoader(TensorDataset(calibration_split.pos, calibration_split.fx, calibration_y),
                                    batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(head_model.head.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3)
    nfull = adapter.full_point_count()
    nmin = int(round(0.1 * nfull))
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
        emit(f"start {args.name} epochs={args.epochs} train={len(train_split.sample_ids)} "
             f"calibration={len(calibration_split.sample_ids)} N=[{nmin},{nfull}] "
             f"target_mean={stats.mean:.9f} target_std={stats.std:.9f}")
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(args.epochs):
            head_model.train()
            total = 0.0
            for pos, fx, target in train_loader:
                count = int(torch.randint(nmin, nfull + 1, (1,), generator=generator).item())
                selected = torch.randperm(nfull, generator=generator)[:count]
                prediction = head_model(pos[:, selected].to(device), fx[:, selected].to(device))
                loss = F.smooth_l1_loss(prediction, target[:, selected].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_model.head.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * pos.shape[0]
            head_model.eval()
            values = []
            with torch.no_grad():
                for pos, fx, target in calibration_loader:
                    prediction = head_model(pos.to(device), fx.to(device))
                    values.append(F.smooth_l1_loss(prediction, target.to(device), reduction="mean"))
            validation = float(torch.stack(values).mean())
            emit(f"epoch={epoch} train_loss={total / len(train_split.sample_ids):.8f} "
                 f"calibration_huber={validation:.8f}" + (" best" if validation < best else ""))
            if validation < best:
                best = validation
                torch.save(head_model.head.state_dict(), best_path)
    last_path = args.output_dir / f"{args.name}_last.pt"
    torch.save(head_model.head.state_dict(), last_path)
    manifest = {
        "benchmark": "plasticity2d", "method": "ca",
        "operator_variant": "legacy_all_timestep", "status": "smoke" if args.smoke else "legacy_diagnostic",
        "head_checkpoint": str(best_path.resolve()), "head_sha256": file_sha256(best_path),
        "operator_checkpoint": str(Path(args.operator).expanduser().resolve()),
        "operator_manifest": str(Path(args.operator_manifest).expanduser().resolve()),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "train_split": [0, 900], "calibration_split": [900, 907],
        "test_split": "frozen_unused", "truth_access": False,
        "epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
        "nfull": nfull, "nmin_fraction": 0.1, "nmin_count": nmin,
        "architecture": "transolver", "blocks": 2,
        "target": stats.target, "target_mean": stats.mean, "target_std": stats.std,
        "best_calibration_huber": best,
        "temporal_mode": "single_forward_all_frames",
        "predicted_frames": data.layout.predicted_frames,
        "displacement_channels": data.layout.displacement_channels,
        "target_layout": data.layout.order,
        "command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
