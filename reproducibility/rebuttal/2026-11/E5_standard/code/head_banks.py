"""Cache fixed sparse first-block features for the actual Elasticity/Darcy heads."""
from common import ROOT, configuration, read, write
from learning import first_features, load_split
import numpy as np
import torch


@torch.no_grad()
def build(benchmark, seed, model, model_manifest, device):
    directory = ROOT / "records" / benchmark / f"seed{seed}" / "head_bank"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        metadata = read(directory / "manifest.json")
        assert metadata["operator_sha256"] == model_manifest["best_sha256"]
        return directory, metadata
    cfg = configuration(benchmark)
    shards = []
    for split in ("train", "validation"):
        data = load_split(benchmark, split, device)
        ids = cfg["split"][split + "_ids"]
        masks = (cfg["head"]["train_masks"] if split == "train" else cfg["head"]["validation_masks"]) if benchmark == "elasticity" else 1
        counts = [48, 72, 97] if benchmark == "elasticity" else [722]
        for count in counts:
            entries = [(i, mask) for i in range(len(ids)) for mask in range(masks)]
            name = f"{split}_n{count}"
            feature_path, target_path = name + "_features.npy", name + "_target.npy"
            x = np.lib.format.open_memmap(directory / feature_path, mode="w+", dtype=np.float32,
                                          shape=(len(entries), count, 128))
            dtype = np.float32 if benchmark == "elasticity" else np.float64
            y = np.lib.format.open_memmap(directory / target_path, mode="w+", dtype=dtype,
                                          shape=(len(entries), count, 1))
            for start in range(0, len(entries), 32):
                group = entries[start:start + 32]
                indexes = []
                for local, mask in group:
                    stream = [30_000_000 if split == "train" else 40_000_000, ids[local], mask, count]
                    if benchmark == "darcy":
                        stream.append(seed)
                    indexes.append(np.random.default_rng(np.random.SeedSequence(stream)).choice(
                        cfg["nfull"], count, replace=False))
                batch = torch.as_tensor([i for i, _ in group], device=device)[:, None]
                points = torch.as_tensor(np.stack(indexes), device=device)
                pos = data["pos"][batch, points]
                fx = data["fx"][batch, points] if data["fx"] is not None else None
                features = first_features(model, pos, fx)
                x[start:start + len(group)] = features.cpu().numpy()
                if benchmark == "elasticity":
                    y[start:start + len(group)] = data["encoded"][batch, points].cpu().numpy()
                else:
                    from adasolver.train.darcy_ca_head import coarse_grad_lap_score
                    prediction = model(pos, fx).cpu().numpy()
                    reference = np.load(ROOT / "data/darcy/reference.npy")
                    for j, indices in enumerate(indexes):
                        y[start + j, :, 0] = coarse_grad_lap_score(reference, indices, prediction[j])
            x.flush()
            y.flush()
            shards.append({"split": split, "count": count, "views": len(entries),
                           "features": feature_path, "target": target_path})
            print("HEAD BANK", benchmark, seed, name, len(entries), flush=True)
        del data
    metadata = {"benchmark": benchmark, "operator_sha256": model_manifest["best_sha256"],
                "training_seed": seed, "shards": shards, "status": "completed"}
    if benchmark == "darcy":
        values = np.log1p(np.load(directory / "train_n722_target.npy"))
        metadata.update(target_mean=float(values.mean()), target_std=float(values.std() + 1e-8))
    write(directory / "manifest.json", metadata)
    return directory, metadata


def tensors(directory, metadata, split, device):
    features, target, weights = [], [], []
    for shard in metadata["shards"]:
        if shard["split"] != split:
            continue
        x = np.load(directory / shard["features"])
        y = np.load(directory / shard["target"])
        if metadata["benchmark"] == "darcy":
            y = ((np.log1p(y) - metadata["target_mean"]) / metadata["target_std"]).astype(np.float32)
        else:
            x, y = x.reshape(-1, 128), y.reshape(-1, 1)
        features.append(torch.as_tensor(x, device=device))
        target.append(torch.as_tensor(y, device=device))
        if metadata["benchmark"] == "elasticity":
            weights.append(torch.full((len(x), 1), 1. / shard["count"], device=device))
    return torch.cat(features), torch.cat(target), torch.cat(weights) if weights else None
