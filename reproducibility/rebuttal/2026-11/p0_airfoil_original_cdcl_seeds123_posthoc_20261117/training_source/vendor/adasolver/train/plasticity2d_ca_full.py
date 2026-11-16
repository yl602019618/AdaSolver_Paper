"""Train small field heads; masks, rather than points, are the loss samples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from adasolver.inference.ca.plasticity2d_full import Full20FieldHead
from adasolver.inference.plasticity2d_contract import write_json
from adasolver.metrics.field import relative_l2
from adasolver.models.checkpoint import sha256_file
from adasolver.train.plasticity2d_full import atomic_torch_save


def load_dataset(root, device):
    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("status") != "complete" or metadata.get("feature_tap") != "actual_S0_block0" or metadata.get("coarse_count") != 313:
        raise ValueError("head needs a complete deployment-matched block0 dataset")
    tensors, ids = {}, {}
    for shard in metadata["shards"]:
        name = shard["split"]
        ids[name] = {r["sample_id"] for r in shard["rows"]}
        arrays = []
        for key in ("features", "targets"):
            path = Path(shard["files"][key])
            if sha256_file(path) != shard["sha256"][key]:
                raise ValueError("CA dataset shard hash mismatch")
            arrays.append(torch.from_numpy(np.load(path)).to(device))
        if arrays[0].shape[:2] != arrays[1].shape[:2] or arrays[0].shape[1] != 313 or arrays[1].shape[-1] != metadata.get("out_dim",40):
            raise ValueError("invalid head feature/target shape")
        tensors[name] = tuple(arrays)
    if ids["train"] & ids["validation"]:
        raise ValueError("CA geometries leak between train and validation")
    return metadata, tensors


def train_head(dataset, output, feature_dim=8, *, device="cuda:0", epochs=200, patience=30,
               batch_masks=32, seed=0, loaded=None):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata, tensors = load_dataset(dataset, device) if loaded is None else loaded
    torch.manual_seed(seed)
    out_dim = metadata.get("out_dim",40)
    head = Full20FieldHead(feature_dim, metadata["hidden_dim"],out_dim).to(device)
    field_target = metadata["target"] == "raw_full20_displacement"
    def loss_values(prediction,target):
        return relative_l2(prediction,target) if field_target else F.smooth_l1_loss(prediction,target,reduction="none").flatten(1).mean(1)
    metric_name = "full_field_relative_l2" if field_target else "gradient_huber"
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, stale, history = float("inf"), 0, []
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(epochs):
        head.train()
        x, y = tensors["train"]
        total = torch.zeros((), device=device)
        for rows in torch.randperm(len(x), device=device).split(batch_masks):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_values(head(x[rows]), y[rows]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite full20 head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            total += loss.detach()*len(rows)
        scheduler.step()
        head.eval()
        vx, vy = tensors["validation"]
        with torch.no_grad():
            validation = torch.cat([loss_values(head(a), b) for a, b in zip(vx.split(batch_masks), vy.split(batch_masks))]).mean().item()
        if validation < best - 1e-7:
            best, best_epoch, stale = validation, epoch, 0
            atomic_torch_save(head.state_dict(), output / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, f"train_{metric_name}": float(total/len(x)),
               f"validation_{metric_name}": validation}
        history.append(row)
        print(json.dumps(dict(feature_dim=feature_dim, **row)), flush=True)
        if stale >= patience:
            break
    atomic_torch_save(head.state_dict(), output / "last.pt")
    result = {
        "task_version": "full20_v1", "method": "ca", "status": "trained_head",
        "feature_dim": feature_dim, "feature_tap": metadata["feature_tap"],
        "coarse_count": 313, "target": metadata["target"], "target_normalization": metadata["target_normalization"], "out_dim": out_dim,
        "operator_sha256": metadata["operator_sha256"], "input_normalizer_sha256": metadata["input_normalizer_sha256"],
        "operator_manifest": metadata["operator_manifest"], "dataset_manifest": str(Path(dataset).resolve()/"manifest.json"),
        "dataset_manifest_sha256": sha256_file(Path(dataset)/"manifest.json"),
        "checkpoint": str(output/"best.pt"), "head_sha256": sha256_file(output/"best.pt"),
        "parameter_count": sum(p.numel() for p in head.parameters()), "architecture": [metadata["hidden_dim"], feature_dim, 32, out_dim],
        "seed": seed, "epochs_requested": epochs, "epochs_completed": len(history), "best_epoch": best_epoch,
        f"best_validation_{metric_name}": best, "patience": patience, "batch_masks": batch_masks,
        "loss": "sample_mask_full20_relative_l2" if field_target else "sample_mask_standardized_gradient_huber", "operator_frozen": True,
        "training_seconds": time.perf_counter()-started, "source_smoke": metadata.get("source_smoke", False),
        "optimizer": {"name":"AdamW","learning_rate":1e-3,"weight_decay":1e-5},
        "scheduler": {"name":"CosineAnnealingLR","T_max":epochs}, "max_grad_norm":1.,
        "training_source_sha256": sha256_file(Path(__file__)),
        "environment": {"torch":torch.__version__,"cuda":torch.version.cuda,"device":str(device)},
    }
    if not field_target:
        result.update(target_mean=metadata["target_mean"],target_std=metadata["target_std"])
    write_json(output/"history.json", history)
    write_json(output/"manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-dims", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--patience", type=int, default=30)
    a = p.parse_args()
    torch.set_num_threads(4)
    loaded = load_dataset(a.dataset, a.device)
    for dim in a.feature_dims:
        train_head(a.dataset, a.output/f"d{dim}", dim, device=a.device, epochs=a.epochs,
                   patience=a.patience, loaded=loaded)


if __name__ == "__main__":
    main()
