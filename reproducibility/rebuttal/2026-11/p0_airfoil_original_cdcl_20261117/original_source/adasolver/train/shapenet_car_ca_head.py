"""Train a scalar pressure MLP; only the cached head parameters are optimized."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from adasolver.inference.ca.shapenet_car import CarPressureHead
from adasolver.inference.shapenet_car_contract import coarse_count
from adasolver.models.checkpoint import sha256_file


def load_split(root, metadata, split, device):
    xs, ys, weights = [], [], []
    for entry in metadata["entries"]:
        if entry["split"] != split:
            continue
        with np.load(root / entry["path"]) as data:
            if entry["coarse_count"] != coarse_count(entry["nfull"]):
                raise ValueError("head dataset contains non-10% features")
            x, y, w = data["features"], data["target"], data["weights"]
            if x.shape != (len(y), metadata["hidden_dim"]) or w.shape != y.shape:
                raise ValueError("head feature/target/weight shape mismatch")
            xs.append(torch.as_tensor(x, device=device))
            ys.append(torch.as_tensor(y, device=device))
            weights.append(torch.as_tensor(w, device=device))
    x, y, w = torch.cat(xs), torch.cat(ys), torch.cat(weights)
    # Each geometry/mask contributes equal total mass, irrespective of how
    # many surface points the coarse subset happens to contain.
    w *= len(w) / w.sum()
    return x, y, w


def train(dataset, output_dir, name="ca_pressure_budget10_v1", *, device="cuda:0",
          width=64, epochs=200, batch_points=65536, seed=0, smoke=False):
    root, output_dir = Path(dataset), Path(output_dir)
    meta = json.loads((root / "manifest.json").read_text())
    if meta["status"] != "complete" and not (smoke and meta["status"] == "smoke"):
        raise ValueError("head training requires a completed dataset")
    if set(meta["train_ids"]) & set(meta["validation_ids"]) or any(
        sid.startswith("param0/") for sid in meta["train_ids"] + meta["validation_ids"]):
        raise ValueError("head train/validation overlap or fold0 leakage")
    if meta["feature_tap"] != "frozen_block0_on_actual_S0" or meta["target"] != "surface_pressure":
        raise ValueError("incompatible head feature/target dataset")
    if epochs < 1 or batch_points < 1:
        raise ValueError("epochs/batch_points must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: output_dir / f"{name}_{k}" for k in ("best.pt", "last.pt", "manifest.json")}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError(f"head output already exists: {name}")
    torch.manual_seed(seed)
    x, y, w = load_split(root, meta, "train", device)
    xv, yv, wv = load_split(root, meta, "validation", device)
    head = CarPressureHead(meta["hidden_dim"], width).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_epoch, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(x), device=device)
        total = torch.zeros((), device=device)
        for idx in order.split(batch_points):
            loss = ((head(x[idx])[:, 0] - y[idx]).square() * w[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            opt.step()
            total += loss.detach() * len(idx)
        scheduler.step()
        head.eval()
        with torch.no_grad():
            validation = sum(((head(xx)[:, 0]-yy).square()*ww).sum()
                             for xx, yy, ww in zip(xv.split(batch_points), yv.split(batch_points), wv.split(batch_points))) / len(xv)
        training, validation = float(total / len(x)), float(validation)
        if not np.isfinite([training, validation]).all():
            raise RuntimeError("nonfinite pressure head loss")
        history.append({"epoch": epoch, "train_mse": training, "validation_mse": validation})
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save(head.state_dict(), paths["best.pt"])
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"epoch={epoch} train_mse={training:.8f} validation_mse={validation:.8f}", flush=True)
    torch.save(head.state_dict(), paths["last.pt"])
    result = {k: meta[k] for k in ("benchmark", "protocol_version", "operator_sha256", "normalizer_sha256",
                                  "operator_manifest", "target", "target_mean", "target_std", "coarse_fraction", "split_hash")}
    result.update(status="smoke" if smoke else "trained_head", input_dim=meta["hidden_dim"], width=width,
                  parameter_count=sum(p.numel() for p in head.parameters()), output_scalars=1,
                  checkpoint=str(paths["best.pt"].resolve()), head_sha256=sha256_file(paths["best.pt"]),
                  dataset_manifest=str((root/"manifest.json").resolve()), dataset_manifest_sha256=sha256_file(root/"manifest.json"),
                  epochs_requested=epochs, epochs_completed=len(history), best_epoch=best_epoch, seed=seed,
                  optimizer="AdamW", learning_rate=1e-3, weight_decay=1e-5, scheduler="CosineAnnealingLR",
                  batch_points=batch_points, loss="masked pressure MSE, equal geometry/mask total weight",
                  train_ids=meta["train_ids"], validation_ids=meta["validation_ids"],
                  operator_frozen=True, best_validation_mse=best, training_history=history,
                  training_seconds=time.perf_counter()-started, source_sha256=sha256_file(__file__))
    paths["manifest.json"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="checkpoints/shapenet_car")
    p.add_argument("--name", default="ca_pressure_budget10_v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-points", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    train(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
