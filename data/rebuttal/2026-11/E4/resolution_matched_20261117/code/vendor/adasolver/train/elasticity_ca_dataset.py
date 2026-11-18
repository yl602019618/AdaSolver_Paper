"""Build real sparse first-block feature banks and scalar stress supervision."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from adasolver.evaluation.elasticity_budget10 import ElasticityRuntime
from adasolver.inference.elasticity_contract import coarse_count, split_ids, validate_coarse_indices
from adasolver.layers.first_block import first_block_features


@torch.no_grad()
def build(output, *, device="cuda:0", fractions=(.05, .075, .1), train_masks=4, validation_masks=2,
          batch_size=64, train_ids=None, validation_ids=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"feature bank is not empty: {output}")
    train_ids = split_ids("head_train") if train_ids is None else list(train_ids)
    validation_ids = split_ids("head_validation") if validation_ids is None else list(validation_ids)
    if (set(train_ids) & set(validation_ids) or any(i not in range(1000) for i in train_ids)
            or any(i not in range(1000, 1100) for i in validation_ids)):
        raise ValueError("invalid head train/validation IDs")
    counts = [coarse_count(972, f) for f in fractions]
    if train_masks < 1 or validation_masks < 1:
        raise ValueError("mask counts must be positive")
    runtime = ElasticityRuntime(device, batch_size=batch_size)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    for split, ids, masks in (("train", train_ids, train_masks), ("validation", validation_ids, validation_masks)):
        for fraction, count in zip(fractions, counts):
            name = f"{split}_n{count}"
            entries = [(sid, mask) for sid in ids for mask in range(masks)]
            arrays = {key: np.lib.format.open_memmap(output/f"{name}_{key}.npy", mode="w+", dtype=dtype,
                                                    shape=(len(entries), count, *shape))
                      for key, dtype, shape in (("features", np.float32, (128,)), ("target", np.float32, (1,)),
                                                ("coords", np.float32, (2,)), ("indices", np.int64, ()))}
            seeds = []
            for start in range(0, len(entries), batch_size):
                group = entries[start:start+batch_size]
                indices = []
                for sid, mask in group:
                    # Separate train/validation mask streams, never joined online.
                    seed = [30_000_000 if split == "train" else 40_000_000, sid, mask, count]
                    chosen = np.random.default_rng(np.random.SeedSequence(seed)).choice(972, count, replace=False)
                    indices.append(validate_coarse_indices(chosen))
                    seeds.append(seed)
                coords = np.stack([runtime.sample(sid)[0][ix] for (sid, _), ix in zip(group, indices)])
                target = np.stack([runtime.sample(sid)[1][ix] for (sid, _), ix in zip(group, indices)])
                features, _ = first_block_features(runtime.model, torch.as_tensor(coords, device=device))
                stop = start + len(group)
                arrays["features"][start:stop] = features.cpu().numpy()
                arrays["coords"][start:stop] = coords
                arrays["target"][start:stop] = (target-runtime.mean)/runtime.std
                arrays["indices"][start:stop] = np.stack(indices)
            for arr in arrays.values():
                arr.flush()
            shards.append({"name": name, "split": split, "coarse_fraction": fraction, "coarse_count": count,
                           "entries": [{"sample_id": sid, "mask": mask, "seed": s} for (sid, mask), s in zip(entries, seeds)],
                           **{key: f"{name}_{key}.npy" for key in arrays}})
            print(f"built {name}: {len(entries)} sparse inputs, {count} points each", flush=True)
    manifest = {"benchmark": "elasticity", "status": "complete", "target": "zscore_stress",
                "target_mean": float(runtime.mean), "target_std": float(runtime.std),
                "feature_tap": "frozen_block0_on_S0", "feature_dim": 128, "physical_output_dim": 1,
                "coarse_cap_count": 97, "operator_sha256": runtime.operator_hash,
                "normalizer_sha256": runtime.normalizer_hash, "operator_checkpoint": runtime.operator_path,
                "train_ids": train_ids, "validation_ids": validation_ids, "shards": shards}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    torch.set_num_threads(1)
    build(**vars(args))


if __name__ == "__main__":
    main()
