"""Load each run's own complete backbone and its fixed data split."""
from common import ROOT, configuration, digest, read
import numpy as np
import torch
from adasolver.models import Transolver3ModelSpec


def load_backbone(benchmark, seed, device):
    cfg = configuration(benchmark)
    directory = ROOT / "checkpoints" / benchmark / f"seed{seed}" / "backbone"
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "completed"
    assert manifest["training_seed"] == seed
    assert manifest["epochs_completed"] == cfg["training"]["epochs"]
    assert digest(directory / "best.pt") == manifest["best_sha256"]
    model = Transolver3ModelSpec(**cfg["model"]).build().to(device)
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=True))
    return model.requires_grad_(False).eval(), manifest


def load_split(benchmark, split, device):
    directory = ROOT / "data" / benchmark
    values = {key: torch.as_tensor(np.load(directory / f"{split}_{key}.npy"), device=device)
              for key in ("pos", "target", "encoded")}
    if values["pos"].ndim == 2:
        values["pos"] = values["pos"][None].expand(len(values["target"]), -1, -1)
    values["fx"] = (torch.as_tensor(np.load(directory / f"{split}_fx.npy"), device=device)
                    if benchmark == "darcy" else values["pos"] if benchmark == "pipe" else None)
    return values


@torch.no_grad()
def first_features(model, pos, fx=None):
    inputs = pos if fx is None else torch.cat((pos, fx), dim=-1)
    hidden = model.net.preprocess(inputs) + model.net.placeholder[None, None, :]
    return model.net.blocks[0](hidden)
