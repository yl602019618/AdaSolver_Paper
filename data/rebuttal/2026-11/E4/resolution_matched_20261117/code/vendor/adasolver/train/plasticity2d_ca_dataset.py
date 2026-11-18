"""Create deployment-matched 313-point block-0 datasets for full20 heads."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from adasolver.evaluation.plasticity2d_full import Full20Evaluator
from adasolver.inference.plasticity2d_contract import N_COARSE, N_FULL, rng_for, write_json
from adasolver.layers.first_block import first_block_features
from adasolver.models.checkpoint import sha256_file


def full20_gradient_target(target):
    values = np.asarray(target, np.float64).reshape(101,31,20,2)
    gx, gy = np.gradient(values, .5, .5, axis=(0,1), edge_order=2)
    return np.sqrt(np.square(gx).sum(axis=(2,3)) + np.square(gy).sum(axis=(2,3))).reshape(-1)


def build_gradient_dataset(field_dataset, data, output):
    """Reuse identical sparse h0 features, supervised by full20 training-only gradient energy."""
    output = Path(output).resolve()
    if (output/"manifest.json").exists():
        return json.loads((output/"manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(json.loads((Path(field_dataset)/"manifest.json").read_text()))
    scores = {}
    for name in ("train", "validation"):
        split = data.splits[name]
        scores[name] = np.stack([full20_gradient_target(t.numpy()) for t in split.target])
    logged = np.log1p(scores["train"])
    mean, std = float(logged.mean()), float(logged.std()+1e-8)
    for shard in metadata["shards"]:
        name = shard["split"]
        lookup = {sid:i for i,sid in enumerate(data.splits[name].sample_ids)}
        indices = np.load(shard["files"]["indices"], mmap_mode="r")
        values = np.stack([scores[name][lookup[row["sample_id"]], selected]
                           for row,selected in zip(shard["rows"],indices)])
        targets = ((np.log1p(values)-mean)/std)[...,None].astype(np.float32)
        path = output/f"{name}_targets.npy"
        np.save(path,targets)
        shard["files"]["targets"] = str(path)
        shard["sha256"]["targets"] = sha256_file(path)
    metadata.update(target="zscore_log1p_full20_gradient", out_dim=1,
                    target_mean=mean, target_std=std, target_normalization="train_zscore_log1p",
                    reference_feature_dataset=str(Path(field_dataset).resolve()/"manifest.json"),
                    gradient_spacing=[.5,.5], gradient_frames=list(range(20)))
    write_json(output/"manifest.json",metadata)
    return metadata


@torch.no_grad()
def build_dataset(manifest, output, device="cuda:0", *, fit_masks=4, validation_masks=2, allow_smoke=False):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    evaluator = Full20Evaluator(manifest, device, allow_smoke=allow_smoke)
    hidden_dim = evaluator.metadata["model"]["hidden_dim"]
    shards = []
    for split_name, masks in (("train", fit_masks), ("validation", validation_masks)):
        split = evaluator.data.splits[split_name]
        shape = len(split.sample_ids)*masks
        paths = {key: output / f"{split_name}_{key}.npy" for key in ("features", "targets", "indices")}
        x = np.lib.format.open_memmap(paths["features"], mode="w+", dtype="float32", shape=(shape, N_COARSE, hidden_dim))
        y = np.lib.format.open_memmap(paths["targets"], mode="w+", dtype="float32", shape=(shape, N_COARSE, 40))
        indices = np.lib.format.open_memmap(paths["indices"], mode="w+", dtype="int64", shape=(shape, N_COARSE))
        rows = []
        for i, sid in enumerate(split.sample_ids):
            for mask in range(masks):
                row = i*masks+mask
                selected = rng_for(mask, sid, 40).choice(N_FULL, N_COARSE, replace=False)
                inputs = torch.cat((split.pos[i:i+1, selected], split.fx[i:i+1, selected]), -1).to(device)
                features, _ = first_block_features(evaluator.operator, inputs)
                x[row] = features[0].cpu().numpy()
                y[row] = split.target[i, selected].numpy()
                indices[row] = selected
                rows.append({"sample_id": sid, "mask_seed": mask})
        x.flush(); y.flush(); indices.flush()
        shards.append({"split": split_name, "rows": rows, "masks_per_geometry": masks,
                       "files": {k: str(v) for k, v in paths.items()},
                       "sha256": {k: sha256_file(v) for k, v in paths.items()}})
        print(f"dataset split={split_name} masks={shape} points_per_mask={N_COARSE}", flush=True)
    metadata = {"status": "complete", "task_version": "full20_v1", "target": "raw_full20_displacement",
                "feature_tap": "actual_S0_block0", "hidden_dim": hidden_dim, "coarse_count": N_COARSE,
                "operator_manifest": str(Path(manifest).resolve()),
                "operator_sha256": evaluator.metadata["checkpoint_sha256"],
                "input_normalizer_sha256": evaluator.metadata["input_normalizer_sha256"],
                "shards": shards, "target_normalization": "none", "source_smoke": allow_smoke}
    write_json(output / "manifest.json", metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fit-masks", type=int, default=4)
    p.add_argument("--validation-masks", type=int, default=2)
    p.add_argument("--allow-smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    build_dataset(a.manifest, a.output, a.device, fit_masks=a.fit_masks,
                  validation_masks=a.validation_masks, allow_smoke=a.allow_smoke)


if __name__ == "__main__":
    main()
