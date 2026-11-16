from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import ElasticityAdapter
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3ModelSpec
from adasolver.models.checkpoint import sha256_file
from adasolver.train.pipe import git_revision, summed_relative_l2, variable_point_subset
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/elasticity"


@torch.no_grad()
def validate(model, loader, normalizer, device: torch.device) -> float:
    model.eval()
    values = []
    for pos, target in loader:
        pos = pos.to(device)
        target = target.to(device)
        prediction = normalizer.decode(model(pos, None))
        values.append(relative_l2(prediction, target).cpu())
    return float(torch.cat(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/elasticity"))
    parser.add_argument("--name", default="elasticity_transolver3_varpts_nmin040")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--nmin-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = " ".join(os.sys.argv)
    revision = git_revision()
    if args.smoke:
        args.epochs = 1
        args.train_samples = min(args.train_samples, 8)
        args.validation_samples = min(args.validation_samples, 4)
        args.name += "_smoke"
    if not 0 < args.nmin_fraction <= 1:
        raise ValueError("nmin_fraction must be in (0, 1]")

    generator = seed_everything(args.seed)
    device = torch.device("cuda:0")
    data = ElasticityAdapter(args.data_path).load()
    full_count = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * full_count))
    if args.nmin_fraction == 0.4 and nmin != 389:
        raise RuntimeError(f"unexpected Elasticity nmin rounding: {nmin}")
    train_dataset = TensorDataset(
        data.train.pos[:args.train_samples], data.train.target[:args.train_samples],
    )
    validation_dataset = TensorDataset(
        data.test.pos[:args.validation_samples], data.test.target[:args.validation_samples],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    model_spec = Transolver3ModelSpec(fun_dim=0, mlp_ratio=2)
    model = model_spec.build().to(device)
    data.y_normalizer.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/elasticity/train") / f"{args.name}.log"
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
            for pos, target in train_loader:
                selected = variable_point_subset(full_count, nmin, generator)
                pos = pos[:, selected].to(device)
                target = target[:, selected].to(device)
                prediction = model(pos, None)
                loss = summed_relative_l2(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                total += float(loss.detach())
            scheduler.step()
            validation = validate(model, validation_loader, data.y_normalizer, device)
            is_best = validation < best_validation
            if is_best:
                best_validation = validation
                torch.save(model.state_dict(), args.output_dir / f"{args.name}_best.pt")
            emit(f"epoch={epoch} train_relative_l2={total / args.train_samples:.8f} "
                 f"validation_relative_l2={validation:.8f}" + (" best" if is_best else ""))

    torch.save(model.state_dict(), args.output_dir / f"{args.name}_last.pt")
    np.savez(
        args.output_dir / f"{args.name}_ynorm.npz",
        mean=data.y_normalizer.mean.cpu().numpy(), std=data.y_normalizer.std.cpu().numpy(),
    )
    best_path = args.output_dir / f"{args.name}_best.pt"
    last_path = args.output_dir / f"{args.name}_last.pt"
    normalizer_path = args.output_dir / f"{args.name}_ynorm.npz"
    manifest = {
        "benchmark": "elasticity", "status": "smoke" if args.smoke else "official",
        "architecture": "Transolver3", "epochs": args.epochs,
        "nmin_fraction": args.nmin_fraction, "nmin_count": nmin, "nfull": full_count,
        "seed": args.seed, "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
        "max_grad_norm": args.max_grad_norm,
        "train_samples": args.train_samples, "validation_samples": args.validation_samples,
        "data_path": str(Path(args.data_path).resolve()),
        "data_split": {"train": [0, 1000], "test": [1800, 2000]},
        "model": model_spec.__dict__,
        "best_validation_relative_l2": best_validation,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "normalizer": str(normalizer_path.resolve()),
        "normalizer_sha256": sha256_file(normalizer_path),
        "git_revision_at_start": revision, "command": command,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "visible_devices": args.gpu},
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
