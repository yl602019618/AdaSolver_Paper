"""Cache pressure-head supervision from real 10% block-0 passes."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from adasolver.data.shapenet_car_runtime import CarRuntime
from adasolver.inference.shapenet_car_contract import (
    DEFAULT_MANIFEST, DEFAULT_SPLITS, PROTOCOL, coarse_count, rng_for,
)
from adasolver.layers.block0_cache import coarse_features
from adasolver.models.checkpoint import sha256_file


@torch.no_grad()
def build_dataset(output, *, manifest=DEFAULT_MANIFEST, splits_path=DEFAULT_SPLITS,
                  data_path=None, masks=2, device="cuda:0", smoke=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if masks < 1:
        raise ValueError("masks must be positive")
    runtime = CarRuntime(manifest, data_path)
    splits = runtime.splits(splits_path)
    train_ids = splits["head_train"][:2] if smoke else splits["head_train"]
    val_ids = splits["head_validation"][:2] if smoke else splits["head_validation"]
    total, total_sq, count = 0., 0., 0
    for sid in train_ids:
        # Only head-training surface labels set the target normalization.
        raw = runtime.data_path / "preprocessed_data" / sid
        mask = np.load(raw / "surf.npy").astype(bool)
        pressure = np.asarray(np.load(raw / "y.npy")[mask, 3], dtype=np.float64)
        total += pressure.sum(); total_sq += np.square(pressure).sum(); count += len(pressure)
    mean = float(total / count)
    std = float(np.sqrt(max(total_sq / count - mean ** 2, 1e-16)))
    operator = runtime.load_operator(device)
    output.mkdir(parents=True)
    entries = []
    start = time.perf_counter()
    for role, ids in (("train", train_ids), ("validation", val_ids)):
        for i, sid in enumerate(ids):
            item = runtime.sample(sid)
            features, targets, weights, groups, selected = [], [], [], [], []
            n0 = coarse_count(len(item))
            for mask_seed in range(masks):
                indices = rng_for(sid, mask_seed, "head_mask").choice(len(item), n0, replace=False)
                active = item.surface_mask[indices].numpy()
                if np.count_nonzero(active) < 2:
                    raise ValueError(f"head mask lacks surface supervision: {sid}")
                pos, fx = item.pos[indices][None].to(device), item.x_norm[indices][None].to(device)
                hidden, _ = coarse_features(operator, pos, fx, nfull=len(item))
                # The full 10% block pass is required; only supervised rows need
                # disk storage for this pointwise pressure head.
                features.append(hidden[0, active].cpu().numpy())
                target = item.target_phys[indices][active, 3].numpy()
                targets.append(((target - mean) / std).astype(np.float32))
                weights.append(np.full(active.sum(), 1. / active.sum(), dtype=np.float32))
                groups.append(np.full(active.sum(), mask_seed, dtype=np.int64))
                selected.append(indices)
            name = f"{role}_{sid.replace('/', '_')}.npz"
            np.savez_compressed(output / name, features=np.concatenate(features), target=np.concatenate(targets),
                                weights=np.concatenate(weights), group=np.concatenate(groups),
                                coarse_indices=np.stack(selected), nfull=len(item))
            entries.append({"split": role, "sample_id": sid, "path": name,
                            "masks": masks, "nfull": len(item), "coarse_count": n0,
                            "rows": int(sum(len(x) for x in targets))})
            if (i + 1) % 50 == 0 or i + 1 == len(ids):
                print(f"{role}: {i+1}/{len(ids)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    result = {**runtime.binding(), "benchmark": "shapenet_car", "protocol_version": PROTOCOL,
              "status": "smoke" if smoke else "complete", "target": "surface_pressure",
              "target_mean": mean, "target_std": std, "target_statistics_samples": train_ids,
              "feature_tap": "frozen_block0_on_actual_S0", "hidden_dim": runtime.spec.hidden_dim,
              "stored_rows": "surface rows only after full 10% block pass", "coarse_fraction": .1,
              "coarse_rounding": "floor", "split_hash": splits["split_hash"],
              "train_ids": train_ids, "validation_ids": val_ids, "masks_per_geometry": masks,
              "entries": entries, "generation_seconds": time.perf_counter()-start,
              "source_sha256": sha256_file(__file__)}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--splits-path", default=str(DEFAULT_SPLITS))
    p.add_argument("--data-path")
    p.add_argument("--masks", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    build_dataset(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
