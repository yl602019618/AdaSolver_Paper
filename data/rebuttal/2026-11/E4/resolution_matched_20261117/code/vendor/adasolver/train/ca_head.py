from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import PipeAdapter
from adasolver.inference.ca.head import PipeCAHead
from adasolver.models import load_pipe_operator
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"
DEFAULT_OPERATOR = "checkpoints/pipe/pipe_transolver3_varpts_nmin040_best.pt"
DEFAULT_TARGETS = "/mnt/ai-material-data/zhijun/Adasolver_3_official/grad_nn/cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pipe"))
    parser.add_argument("--name", default="pipe_ca_head_nmin040")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--nmin-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.name += "_smoke"
    seed_everything(args.seed)
    device = torch.device(args.device)

    data = PipeAdapter(args.data_path).load()
    operator = load_pipe_operator(args.operator, device=device)
    head_model = PipeCAHead(operator).to(device)
    for parameter in head_model.operator.parameters():
        parameter.requires_grad_(False)
    head_model.operator.eval()

    target_dir = Path(args.targets)
    target_train = np.load(target_dir / "pipe_gradmag_train.npy")
    target_test = np.load(target_dir / "pipe_gradmag_test.npy")
    log_train = np.log1p(target_train)
    log_test = np.log1p(target_test)
    mean = float(log_train.mean())
    std = float(log_train.std() + 1e-8)
    train_target = torch.from_numpy((log_train - mean) / std).float().unsqueeze(-1)
    test_target = torch.from_numpy((log_test - mean) / std).float().unsqueeze(-1)

    train_ds = TensorDataset(data.train.pos, data.train.fx, train_target)
    test_ds = TensorDataset(data.test.pos, data.test.fx, test_target)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(head_model.head.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start)
    generator = torch.Generator().manual_seed(args.seed)
    nfull = data.train.pos.shape[1]
    nmin = int(round(args.nmin_fraction * nfull))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/pipe/train") / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with log_path.open("a") as log:
        def emit(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        emit(f"start {args.name} epochs={args.epochs} N=[{nmin},{nfull}] target_mean={mean:.9f} target_std={std:.9f}")
        for epoch in range(args.epochs):
            head_model.train()
            total = 0.0
            for pos, fx, target in train_loader:
                count = int(torch.randint(nmin, nfull + 1, (1,), generator=generator).item())
                selected = torch.randperm(nfull, generator=generator)[:count]
                prediction = head_model(pos[:, selected].to(device), fx[:, selected].to(device))
                loss = F.smooth_l1_loss(prediction, target[:, selected].to(device), beta=1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_model.head.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * pos.shape[0]
            head_model.eval()
            values = []
            with torch.no_grad():
                for pos, fx, target in test_loader:
                    pred = head_model(pos.to(device), fx.to(device))
                    pred_log = pred * std + mean
                    target_log = target.to(device) * std + mean
                    values.append(torch.linalg.vector_norm(pred_log.reshape(pred.shape[0], -1) - target_log.reshape(pred.shape[0], -1), dim=1) /
                                  torch.linalg.vector_norm(target_log.reshape(pred.shape[0], -1), dim=1).clamp_min(1e-12))
            validation = float(torch.cat(values).mean())
            if validation < best:
                best = validation
                torch.save(head_model.head.state_dict(), args.output_dir / f"{args.name}_best.pt")
            emit(f"epoch={epoch} train_loss={total / len(data.train.sample_ids):.8f} validation_rel_log={validation:.8f}" + (" best" if validation == best else ""))
    torch.save(head_model.head.state_dict(), args.output_dir / f"{args.name}_last.pt")
    manifest = {
        "benchmark": "pipe", "method": "ca", "status": "official_head" if not args.smoke else "smoke",
        "operator": str(Path(args.operator).resolve()), "epochs": args.epochs,
        "nmin_fraction": args.nmin_fraction, "nmin_count": nmin, "nfull": nfull,
        "seed": args.seed, "batch_size": args.batch_size, "architecture": "transolver",
        "blocks": 2, "head_parameters": sum(p.numel() for p in head_model.head.parameters()),
        "loss": "huber", "target": "zscore(log1p|grad u|)",
        "target_mean": mean, "target_std": std, "best_validation_relative_log": best,
        "data_path": str(Path(args.data_path).resolve()), "command": " ".join(__import__('sys').argv),
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
