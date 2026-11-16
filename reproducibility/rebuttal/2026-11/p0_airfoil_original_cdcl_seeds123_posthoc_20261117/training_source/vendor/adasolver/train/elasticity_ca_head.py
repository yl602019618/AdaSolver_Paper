from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import ElasticityAdapter
from adasolver.inference.ca import ElasticityCAHead
from adasolver.models import load_elasticity_operator
from adasolver.models.checkpoint import sha256_file
from adasolver.train.pipe import variable_point_subset
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/elasticity"
DEFAULT_TARGETS = "/mnt/ai-material-data/zhijun/Adasolver_3_official/grad_nn/cache"


@torch.no_grad()
def validate(model, loader, mean: float, std: float, device: torch.device) -> float:
    model.eval()
    values = []
    for pos, target in loader:
        prediction = model(pos.to(device), None)
        pred_log = prediction * std + mean
        target_log = target.to(device) * std + mean
        diff = torch.linalg.vector_norm((pred_log - target_log).flatten(1), dim=1)
        denom = torch.linalg.vector_norm(target_log.flatten(1), dim=1).clamp_min(1e-12)
        values.append((diff / denom).cpu())
    return float(torch.cat(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--operator-manifest", required=True)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/elasticity"))
    parser.add_argument("--name", default="elasticity_ca_head_nmin040")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--nmin-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.name += "_smoke"
    seed_everything(args.seed)
    device = torch.device(args.device)
    data = ElasticityAdapter(args.data_path).load()
    operator = load_elasticity_operator(args.operator, args.operator_manifest, device=device)
    model = ElasticityCAHead(operator).to(device)

    target_dir = Path(args.targets)
    grad_train = np.load(target_dir / "elasticity_gradmag_train.npy")
    grad_test = np.load(target_dir / "elasticity_gradmag_test.npy")
    if grad_train.shape != (1000, 972) or grad_test.shape != (200, 972):
        raise ValueError(f"unexpected Elasticity CA target shapes: {grad_train.shape}, {grad_test.shape}")
    log_train = np.log1p(grad_train)
    log_test = np.log1p(grad_test)
    mean = float(log_train.mean())
    std = float(log_train.std() + 1e-8)
    train_target = torch.from_numpy((log_train - mean) / std).float().unsqueeze(-1)
    test_target = torch.from_numpy((log_test - mean) / std).float().unsqueeze(-1)
    train_count = 8 if args.smoke else 1000
    test_count = 4 if args.smoke else 200
    train_loader = DataLoader(
        TensorDataset(data.train.pos[:train_count], train_target[:train_count]),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        TensorDataset(data.test.pos[:test_count], test_target[:test_count]),
        batch_size=args.batch_size, shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start,
    )
    generator = torch.Generator().manual_seed(args.seed)
    nfull = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * nfull))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/elasticity/train") / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with log_path.open("a", encoding="utf-8") as log:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        emit(f"start {args.name} epochs={args.epochs} N=[{nmin},{nfull}] "
             f"target_mean={mean:.9f} target_std={std:.9f}")
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for pos, target in train_loader:
                selected = variable_point_subset(nfull, nmin, generator)
                prediction = model(pos[:, selected].to(device), None)
                loss = F.smooth_l1_loss(prediction, target[:, selected].to(device), beta=1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * pos.shape[0]
            validation = validate(model, test_loader, mean, std, device)
            if validation < best:
                best = validation
                torch.save(model.head.state_dict(), args.output_dir / f"{args.name}_best.pt")
            emit(f"epoch={epoch} train_loss={total / train_count:.8f} "
                 f"validation_rel_log={validation:.8f}" + (" best" if validation == best else ""))

    torch.save(model.head.state_dict(), args.output_dir / f"{args.name}_last.pt")
    best_path = args.output_dir / f"{args.name}_best.pt"
    last_path = args.output_dir / f"{args.name}_last.pt"
    manifest = {
        "benchmark": "elasticity", "method": "ca",
        "status": "smoke" if args.smoke else "official_head",
        "operator": str(Path(args.operator).resolve()),
        "operator_sha256": sha256_file(args.operator),
        "head": str(best_path.resolve()), "head_sha256": sha256_file(best_path),
        "last_head": str(last_path.resolve()), "last_head_sha256": sha256_file(last_path),
        "architecture": "transolver", "blocks": 2, "hidden_dim": 128,
        "heads": 8, "mlp_ratio": 2, "slices": 64,
        "parameter_count": sum(p.numel() for p in model.head.parameters()),
        "target": "zscore(log1p(abs_grad_stress))",
        "target_mean": mean, "target_std": std,
        "epochs": args.epochs, "nmin_fraction": args.nmin_fraction,
        "nmin_count": nmin, "nfull": nfull, "batch_size": args.batch_size,
        "loss": "huber", "best_validation_relative_log": best,
        "seed": args.seed, "command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
