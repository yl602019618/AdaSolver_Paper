"""Fit a one-output pressure MLP on deployment-matched cached features."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from adasolver.inference.ca.airfoil2d import PressureMLP
from adasolver.inference.airfoil2d_contract import validate_coarse_count
from adasolver.models.checkpoint import sha256_file


def load_tensors(root, manifest, split, device, surface_balance=False):
    features, targets, weights = [], [], []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        validate_coarse_count(shard["coarse_count"], manifest["nfull"])
        feature = np.load(root / shard["features"], mmap_mode="r")
        target = np.load(root / shard["target"], mmap_mode="r")
        if feature.shape[:2] != target.shape[:2] or feature.shape[1:] != (shard["coarse_count"], 128):
            raise ValueError("cache feature/target shapes do not match their manifest")
        features.append(torch.tensor(feature.reshape(-1, 128), device=device))
        targets.append(torch.tensor(target.reshape(-1, 1), device=device))
        if surface_balance:
            indices = np.load(root / shard["indices"], mmap_mode="r")
            surface = np.isin(indices, np.arange(50,171)*51)
            w = np.where(surface, .5 * indices.shape[1] / 121,
                         .5 * indices.shape[1] / (indices.shape[1] - 121)).astype(np.float32)
            weights.append(torch.tensor(w.reshape(-1, 1), device=device))
    if not features:
        raise ValueError(f"no {split} cache shards")
    return torch.cat(features), torch.cat(targets), torch.cat(weights) if weights else None


def train(dataset, output_dir, name, *, feature_dim=8, device="cuda:0", epochs=200,
          patience=30, batch_points=65536, seed=0, surface_balance=False):
    root, output_dir = Path(dataset), Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("feature_tap") != "frozen_block0_on_S0":
        raise ValueError("head training requires a complete sparse block-0 dataset")
    if set(manifest["train_ids"]) & set(manifest["validation_ids"]):
        raise ValueError("head dataset train/validation IDs overlap")
    if epochs < 1 or patience < 1 or batch_points < 1:
        raise ValueError("epochs, patience and batch points must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path, manifest_path = [output_dir / f"{name}_{suffix}" for suffix in
                                         ("best.pt", "last.pt", "manifest.json")]
    if any(path.exists() for path in (best_path, last_path, manifest_path)):
        raise FileExistsError(f"head artifacts already exist: {name}")
    torch.manual_seed(seed)
    x, y, w = load_tensors(root, manifest, "train", device, surface_balance)
    xv, yv, wv = load_tensors(root, manifest, "validation", device, surface_balance)
    head = PressureMLP(feature_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, stale, best_epoch = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(x), device=device)
        train_sum = torch.zeros((), device=device)
        for indices in order.split(batch_points):
            loss = F.smooth_l1_loss(head(x[indices]), y[indices], reduction="none")
            if w is not None:
                loss = loss * w[indices]
            value = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.detach().sum()
        scheduler.step()
        head.eval()
        validation_sum = torch.zeros((), device=device)
        with torch.no_grad():
            for start in range(0, len(xv), batch_points):
                loss = F.smooth_l1_loss(head(xv[start:start+batch_points]), yv[start:start+batch_points], reduction="none")
                if wv is not None:
                    loss = loss * wv[start:start+batch_points]
                validation_sum += loss.sum()
        training, validation = float(train_sum / len(x)), float(validation_sum / len(xv))
        if not np.isfinite([training, validation]).all():
            raise RuntimeError("nonfinite pressure head loss")
        history.append({"epoch": epoch, "train_huber": training, "validation_huber": validation})
        print(f"epoch={epoch} train_huber={training:.8f} sparse_validation_huber={validation:.8f}", flush=True)
        if validation < best - 1e-7:
            best, best_epoch, stale = validation, epoch, 0
            torch.save(head.state_dict(), best_path)
        else:
            stale += 1
        if stale >= patience:
            break
    torch.save(head.state_dict(), last_path)
    result = {"benchmark": "airfoil2d", "method": "ca", "status": "trained_head",
              "architecture": "pressure_mlp", "target": "zscore_pressure", "target_name": "pressure",
              "target_channel": 3, "truth_access": False, "nfull": 11271,
              "feature_dim": feature_dim, "hidden_dim": 128, "feature_tap": manifest["feature_tap"],
              "parameter_count": sum(p.numel() for p in head.parameters()),
              "operator_checkpoint": manifest["operator_checkpoint"], "operator_manifest": manifest["operator_manifest"],
              "operator_sha256": manifest["operator_sha256"], "normalizer_sha256": manifest["normalizer_sha256"],
              "target_mean": manifest["target_mean"], "target_std": manifest["target_std"],
              "head_sha256": sha256_file(best_path), "operator_frozen": True,
              "dataset_manifest": str((root/"manifest.json").resolve()), "fractions": manifest["fractions"],
              "coarse_cap_count": 1127, "data_split": {"head_train": manifest["train_ids"], "head_validation": manifest["validation_ids"]},
              "epochs_requested": epochs, "epochs_completed": len(history), "best_epoch": best_epoch,
              "patience": patience, "seed": seed, "best_validation_huber": best,
              "surface_balance": surface_balance, "batch_points": batch_points,
              "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-5},
              "scheduler": {"name": "CosineAnnealingLR", "T_max": epochs},
              "max_grad_norm": 1.0, "loss": "smooth_l1_beta1",
              "architecture_layers": [128, feature_dim, 32, 1], "activation": "GELU",
              "checkpoint_selection": "minimum held-out sparse validation Huber",
              "training_history": history, "training_seconds": time.perf_counter()-started}
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="checkpoints/airfoil2d")
    parser.add_argument("--name", default="airfoil2d_ca_pressure_budget10_d8")
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surface-balance", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
