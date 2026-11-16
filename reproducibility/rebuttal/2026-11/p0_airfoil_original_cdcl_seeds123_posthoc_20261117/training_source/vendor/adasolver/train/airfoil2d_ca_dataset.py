"""Generate frozen block-0 features on actual <=10% Airfoil subsets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.inference.airfoil2d_contract import coarse_count, validate_coarse_indices
from adasolver.inference.uniform.airfoil2d import sample_airfoil_uniform
from adasolver.layers.block0_cache import coarse_features
from adasolver.models.checkpoint import load_airfoil2d_operator, sha256_file

DEFAULT_OPERATOR = "checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_best.pt"
DEFAULT_MANIFEST = "checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_manifest.json"
DEFAULT_DATA = "/mnt/ai-material-data/shuaibinqu/NACA/data/naca"


@torch.no_grad()
def build_dataset(output, *, operator=DEFAULT_OPERATOR, operator_manifest=DEFAULT_MANIFEST,
                  data_path=DEFAULT_DATA, fractions=(.05, .10), masks=2, device="cuda:0",
                  train_ids=range(900), validation_ids=range(900, 1000)):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dataset output is not empty: {output}")
    train_ids, validation_ids = list(train_ids), list(validation_ids)
    if set(train_ids) & set(validation_ids) or any(i < 0 or i >= 1000 for i in train_ids + validation_ids):
        raise ValueError("head train/validation must be disjoint subsets of [0,1000)")
    if not train_ids or not validation_ids or masks < 1:
        raise ValueError("nonempty train/validation and positive mask count required")
    counts = [coarse_count(11271, f, 121) for f in fractions]
    data = AirfoilRuntimeData(data_path, operator_manifest)
    model = load_airfoil2d_operator(operator, operator_manifest, device=device)
    model.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        for fraction, count in zip(fractions, counts):
            name = f"{split}_cf{fraction:g}"
            records = len(ids) * masks
            arrays = {
                "features": np.lib.format.open_memmap(output / f"{name}_features.npy", mode="w+", dtype=np.float32, shape=(records, count, 128)),
                "target": np.lib.format.open_memmap(output / f"{name}_target.npy", mode="w+", dtype=np.float32, shape=(records, count, 1)),
                "indices": np.lib.format.open_memmap(output / f"{name}_indices.npy", mode="w+", dtype=np.int64, shape=(records, count)),
            }
            entries = []
            for local, sid in enumerate(ids):
                coords, target = data.coords(sid), data.target(sid)
                for mask in range(masks):
                    mask_seed = (10_000_000 if split == "train" else 20_000_000) + 1000 * sid + mask
                    chosen = sample_airfoil_uniform(11271, count, data.mandatory, np.random.default_rng(mask_seed)).indices
                    validate_coarse_indices(chosen, 11271, data.mandatory)
                    pos = torch.as_tensor(coords[chosen], device=device)[None]
                    feature = coarse_features(model, pos, pos)[0][0].cpu().numpy()
                    row = local * masks + mask
                    arrays["features"][row] = feature
                    arrays["target"][row] = (target[chosen] - np.float32(data.mean)) / np.float32(data.std)
                    arrays["indices"][row] = chosen
                    entries.append({"sample_id": sid, "mask_seed": mask_seed})
                if (local + 1) % 100 == 0:
                    print(f"{name}: {local+1}/{len(ids)} geometries", flush=True)
            for arr in arrays.values():
                arr.flush()
            shards.append({"name": name, "split": split, "fraction": fraction, "coarse_count": count,
                           "records": records, "entries": entries,
                           **{key: f"{name}_{key}.npy" for key in arrays}})
    manifest = {"benchmark": "airfoil2d", "version": "budget10_v1", "status": "complete",
                "operator_checkpoint": str(Path(operator).resolve()), "operator_manifest": str(Path(operator_manifest).resolve()),
                "operator_sha256": sha256_file(operator), "normalizer_sha256": data.normalizer_sha256,
                "data_path": str(Path(data_path).resolve()),
                "target": "zscore_pressure", "target_mean": data.mean, "target_std": data.std,
                "feature_tap": "frozen_block0_on_S0", "hidden_dim": 128, "nfull": 11271,
                "train_ids": train_ids, "validation_ids": validation_ids, "masks_per_geometry_fraction": masks,
                "fractions": list(fractions), "coarse_cap_count": 1127,
                "mandatory_count": 121, "shards": shards}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operator", default=DEFAULT_OPERATOR)
    p.add_argument("--operator-manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--data-path", default=DEFAULT_DATA)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fractions", type=float, nargs="+", default=[.05, .10])
    p.add_argument("--masks", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    kwargs = dict(train_ids=range(2), validation_ids=range(900, 902)) if a.smoke else {}
    result = build_dataset(a.output, operator=a.operator, operator_manifest=a.operator_manifest,
                          data_path=a.data_path, fractions=a.fractions, masks=a.masks, device=a.device, **kwargs)
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
