"""Train small heads; save fixed epochs for downstream rAUEC selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from adasolver.inference.ca.elasticity_stress import ElasticityStressHead
from adasolver.models.checkpoint import sha256_file


def tensors(root, manifest, split, device):
    xx, yy, ww = [], [], []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        x = np.load(root/shard["features"])
        y = np.load(root/shard["target"])
        if x.shape[:-1] != y.shape[:-1] or x.shape[-1] != 128:
            raise ValueError("feature bank shape mismatch")
        # Each geometry-mask gets equal total loss across different n0 values.
        xx.append(torch.as_tensor(x.reshape(-1, 128), device=device))
        yy.append(torch.as_tensor(y.reshape(-1, 1), device=device))
        ww.append(torch.full((x.shape[0]*x.shape[1], 1), 1./x.shape[1], device=device))
    return torch.cat(xx), torch.cat(yy), torch.cat(ww)


def train(dataset, output, *, device="cuda:0", dims=(0, 1, 2, 4, 8), losses=("mse",),
          epochs=200, seed=0, batch_points=65536):
    root, output = Path(dataset), Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["feature_tap"] != "frozen_block0_on_S0":
        raise ValueError("training requires a complete sparse bank")
    if set(manifest["train_ids"]) & set(manifest["validation_ids"]):
        raise ValueError("head split overlap")
    x, y, w = tensors(root, manifest, "train", device)
    xv, yv, wv = tensors(root, manifest, "validation", device)
    weight_sum = w.sum()
    mean = (x*w).sum(0)/weight_sum
    std = (((x-mean)**2*w).sum(0)/weight_sum).sqrt().clamp_min(1e-5)
    candidates = []
    for loss_name in losses:
        if loss_name not in {"mse", "huber"}:
            raise ValueError("unsupported head loss")
        for dim in dims:
            torch.manual_seed(seed)
            head = ElasticityStressHead(dim).to(device)
            head.feature_mean.copy_(mean)
            head.feature_std.copy_(std)
            destination = output/f"d{dim}_{loss_name}_seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            if (destination/"history.json").exists():
                raise FileExistsError(destination)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            history, began = [], time.perf_counter()
            for epoch in range(1, epochs+1):
                head.train()
                order = torch.randperm(len(x), device=device)
                total = 0.
                for indices in order.split(batch_points):
                    prediction = head(x[indices])
                    element = ((prediction-y[indices])**2 if loss_name == "mse" else
                               F.smooth_l1_loss(prediction, y[indices], reduction="none"))
                    loss = (element*w[indices]).sum()/w[indices].sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    optimizer.step()
                    total += float((element*w[indices]).sum().detach())
                scheduler.step()
                head.eval()
                with torch.no_grad():
                    val_sum = sum(float((((head(xv[i:i+batch_points])-yv[i:i+batch_points])**2)
                                         *wv[i:i+batch_points]).sum())
                                  for i in range(0, len(xv), batch_points))
                row = {"epoch": epoch, "train_loss": total/float(weight_sum), "validation_mse": val_sum/float(wv.sum())}
                history.append(row)
                if epoch % 25 == 0 or epoch == epochs:
                    print(f"d={dim} loss={loss_name} seed={seed} {row}", flush=True)
                if epoch in {25, 50, 100, 150, 200, epochs}:
                    path = destination/f"epoch{epoch}.pt"
                    torch.save(head.state_dict(), path)
                    metadata = {k: manifest[k] for k in ("benchmark", "target", "target_mean", "target_std",
                                "feature_tap", "feature_dim", "physical_output_dim", "coarse_cap_count",
                                "operator_sha256", "normalizer_sha256", "train_ids", "validation_ids")}
                    metadata.update(status="candidate_requires_rauec_selection", bottleneck_dim=dim,
                                    architecture="linear" if dim == 0 else "128_d_1_gelu",
                                    parameter_count=sum(p.numel() for p in head.parameters()),
                                    epoch=epoch, epochs_requested=epochs, seed=seed, loss=loss_name,
                                    feature_normalization="train_weighted_mean_std", operator_frozen=True,
                                    head_sha256=sha256_file(path), dataset_manifest=str(root/"manifest.json"),
                                    checkpoint_selection="fixed epochs; final selection by downstream validation/development rAUEC")
                    mpath = path.with_suffix(".json")
                    mpath.write_text(json.dumps(metadata, indent=2)+"\n")
                    candidates.append({"head_checkpoint": str(path), "head_manifest": str(mpath), "dim": dim,
                                       "epoch": epoch, "loss": loss_name, "seed": seed})
            (destination/"history.json").write_text(json.dumps({"history": history, "training_seconds": time.perf_counter()-began}, indent=2)+"\n")
    index = output/f"candidates_seed{seed}_{'_'.join(losses)}.json"
    index.write_text(json.dumps(candidates, indent=2)+"\n")
    return candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dims", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    p.add_argument("--losses", nargs="+", default=["mse"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    train(**vars(args))


if __name__ == "__main__":
    main()
