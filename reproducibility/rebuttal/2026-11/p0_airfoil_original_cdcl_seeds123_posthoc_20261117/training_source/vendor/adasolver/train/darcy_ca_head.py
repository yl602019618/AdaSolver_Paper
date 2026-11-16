from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import DarcyAdapter
from adasolver.inference.acquisition import grad_score
from adasolver.inference.ca import DarcyCAHead
from adasolver.models import load_darcy_operator
from adasolver.models.checkpoint import sha256_file
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy"
DEFAULT_OPERATOR = "checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_best.pt"
DEFAULT_OPERATOR_MANIFEST = "checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_manifest.json"


def coarse_grad_lap_score(reference: np.ndarray, coarse_indices: np.ndarray,
                          prediction: np.ndarray) -> np.ndarray:
    """Compute the CPG grad-plus-laplacian target on coarse points."""
    reference = np.asarray(reference, dtype=np.float64)
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
    coords = reference[coarse_indices]
    values = np.asarray(prediction, dtype=np.float64).reshape(len(coords), -1)
    gradient = grad_score(coords, values, k=8)
    axis = np.unique(reference[:, 0])
    ay = np.unique(reference[:, 1])
    gx, gy = np.meshgrid(axis, ay, indexing="xy")
    from scipy.interpolate import griddata
    coarse = griddata(coords, values[:, 0], np.column_stack((gx.ravel(), gy.ravel())), method="nearest")
    field = coarse.reshape(len(ay), len(axis))
    lap = np.abs(np.gradient(np.gradient(field, axis=0), axis=0) +
                 np.gradient(np.gradient(field, axis=1), axis=1)).reshape(-1)
    xi = np.searchsorted(axis, coords[:, 0])
    yi = np.searchsorted(ay, coords[:, 1])
    return np.maximum(gradient + lap[yi * len(axis) + xi], 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--operator-manifest", default=DEFAULT_OPERATOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/darcy_ca"))
    parser.add_argument("--name", default="darcy_ca_head_grad_lap_cf010")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--pct-start", type=float, default=0.3)
    parser.add_argument("--coarse-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.name += "_smoke"
    seed_everything(args.seed)
    device = torch.device(args.device)
    data = DarcyAdapter(args.data_path).load()
    operator = load_darcy_operator(args.operator, args.operator_manifest, device=device)
    model = DarcyCAHead(operator).to(device)
    reference = DarcyAdapter(args.data_path).reference_coords().numpy()
    nfull = len(reference)
    coarse_count = max(8, int(round(nfull * args.coarse_fraction)))
    if coarse_count >= 2709:
        raise ValueError("coarse subset must be below the smallest final Darcy budget")

    train_count = 8 if args.smoke else 1000
    holdout_start = max(0, train_count - (2 if args.smoke else 100))
    generator = np.random.default_rng(args.seed)
    positions, features, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for sample in range(train_count):
            indices = generator.choice(nfull, coarse_count, replace=False)
            pos = data.train.pos[sample:sample + 1, indices].to(device)
            fx = data.train.fx[sample:sample + 1, indices].to(device)
            prediction = operator(pos, fx)[0].detach().cpu().numpy()
            target = coarse_grad_lap_score(reference, indices, prediction)
            positions.append(data.train.pos[sample, indices])
            features.append(data.train.fx[sample, indices])
            targets.append(target)
    log_targets = np.log1p(np.asarray(targets))
    mean = float(log_targets[:holdout_start].mean())
    std = float(log_targets[:holdout_start].std() + 1e-8)
    target_tensor = torch.from_numpy(((log_targets - mean) / std)[..., None]).float()
    train_ds = TensorDataset(torch.stack(positions), torch.stack(features), target_tensor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=args.pct_start,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    log_path = Path("logs/darcy/train") / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for pos, fx, target in train_loader:
                pred = model(pos.to(device), fx.to(device))
                loss = F.smooth_l1_loss(pred, target.to(device), beta=1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step()
                total += float(loss.detach()) * pos.shape[0]
            model.eval()
            with torch.no_grad():
                pos = torch.stack(positions[holdout_start:]).to(device)
                fx = torch.stack(features[holdout_start:]).to(device)
                target = target_tensor[holdout_start:].to(device)
                validation = float(torch.linalg.vector_norm(
                    model(pos, fx).flatten(1) - target.flatten(1), dim=1
                ).mean())
            message = f"epoch={epoch} train_loss={total / train_count:.8f} validation={validation:.8f}"
            print(message, flush=True); log.write(message + "\n"); log.flush()
            if validation < best:
                best = validation
                torch.save(model.head.state_dict(), args.output_dir / f"{args.name}_best.pt")
    last_path = args.output_dir / f"{args.name}_last.pt"
    torch.save(model.head.state_dict(), last_path)
    best_path = args.output_dir / f"{args.name}_best.pt"
    operator_manifest = Path(args.operator_manifest).resolve()
    manifest = {
        "benchmark": "darcy", "method": "ca",
        "status": "smoke" if args.smoke else "official_head",
        "operator": str(Path(args.operator).resolve()), "operator_sha256": sha256_file(args.operator),
        "operator_manifest": str(operator_manifest), "operator_manifest_sha256": sha256_file(operator_manifest),
        "head": str(best_path.resolve()), "head_sha256": sha256_file(best_path),
        "last_head": str(last_path.resolve()), "last_head_sha256": sha256_file(last_path),
        "architecture": "transolver", "blocks": 2, "hidden_dim": 128, "heads": 8,
        "mlp_ratio": 2, "slices": 64, "parameter_count": sum(p.numel() for p in model.head.parameters()),
        "target": "zscore_log1p_grad_lap", "target_version": "coarse_prediction_grad_score_plus_laplacian_v1",
        "acquisition": "grad_lap", "coarse_fraction": args.coarse_fraction,
        "coarse_count": coarse_count, "nfull": nfull, "train_samples": 1000,
        "optimization_samples": 1000, "validation_source": "smooth1_holdout",
        "validation_ids": list(range(holdout_start, train_count)), "epochs": args.epochs,
        "batch_size": args.batch_size, "seed": args.seed, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "loss": "huber", "target_mean": mean,
        "target_std": std, "best_validation": best, "command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
