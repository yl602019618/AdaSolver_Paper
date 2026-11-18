from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adasolver.data import PipeAdapter
from adasolver.inference.acquisition import grad_score
from adasolver.inference.base import SamplingResult
from adasolver.inference.adaptive import build_patches, comp_weights_for_subset
from adasolver.inference.adaptive_fast import build_patch_cache, refine_once as refine_fast
from adasolver.inference.ca import PipeCAHead
from adasolver.inference.pi import pipe_wall_distance, sample_pipe_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.models import load_pipe_operator
from adasolver.utils.budget import budget_count
from adasolver.utils.interpolation import index_grid, reconstruct_field

DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"
OPERATOR = "checkpoints/pipe/pipe_transolver3_varpts_nmin040_best.pt"
HEAD = "checkpoints/pipe/pipe_ca_head_nmin040_best.pt"
SAMPLES = (1149, 1174, 1040)
BUDGETS = (0.375, 0.5, 0.75)


def save_field(path: Path, field: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(field.reshape(129, 129).T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("streamwise index")
    ax.set_ylabel("wall-normal index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_density(path: Path, reference: np.ndarray, indices: np.ndarray,
                 density: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    points = ax.scatter(reference[indices, 0], reference[indices, 1], c=density,
                        s=3, cmap="magma", rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("streamwise index")
    ax.set_ylabel("wall-normal index")
    ax.set_aspect("equal")
    fig.colorbar(points, ax=ax, label="relative sampling density")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/pipe/figures"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = PipeAdapter(DATA).load()
    model = load_pipe_operator(OPERATOR, device=device)
    context, _ = install_compensation(model)
    data.y_normalizer.to(device)
    head = PipeCAHead(model).to(device).eval()
    head.head.load_state_dict(torch.load(HEAD, map_location="cpu", weights_only=False), strict=True)
    reference = index_grid(data.shape)
    patch_id, patch_mass = build_patches(reference, 24, seed=0)
    patch_cache = build_patch_cache(reference, patch_id, 24)

    manifest = {"samples": list(SAMPLES), "sample_selection": "nearest to fullmesh error quartiles",
                "seed": 0, "budgets": list(BUDGETS), "methods": {}}
    for sample_id in SAMPLES:
        local = sample_id - 1000
        sample_dir = args.output_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target = data.test.target[local].numpy().reshape(-1)
        with torch.no_grad():
            pred = data.y_normalizer.decode(model(data.test.pos[local:local + 1].to(device),
                                                   data.test.pos[local:local + 1].to(device)))[0].cpu().numpy().reshape(-1)
        save_field(sample_dir / "target.png", target, f"Pipe sample {sample_id}: GT")
        save_field(sample_dir / "fullmesh_prediction.png", pred, f"Pipe sample {sample_id}: fullmesh prediction")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(reference[:, 0], reference[:, 1], c=data.test.pos[local, :, 0].numpy(),
                   s=2, cmap="viridis", rasterized=True)
        ax.set_title(f"Pipe sample {sample_id}: input X geometry")
        ax.set_xlabel("streamwise index")
        ax.set_ylabel("wall-normal index")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(sample_dir / "input_geometry.png", dpi=140)
        plt.close(fig)
        method_records = {}
        wall = pipe_wall_distance(data.test.pos[local].numpy(), data.shape)
        for budget in BUDGETS:
            keep = budget_count(len(reference), budget)
            for method in ("uniform", "pi", "cpg", "ca"):
                rng = np.random.default_rng(sample_id - 1000)
                if method == "uniform":
                    sampling = sample_uniform(len(reference), keep, rng)
                elif method == "pi":
                    sampling = sample_pipe_pi(wall, keep, rng, 0.05, 0.25, 0.75)
                else:
                    coarse_count = budget_count(len(reference), 0.10)
                    coarse = rng.choice(len(reference), coarse_count, replace=False)
                    with torch.no_grad():
                        pos0 = data.test.pos[local:local + 1, coarse].to(device)
                        if method == "cpg":
                            score = grad_score(reference[coarse], data.y_normalizer.decode(model(pos0, pos0))[0].cpu().numpy(), k=8)
                        else:
                            score = PipeCAHead.to_gradient_score(head(pos0, pos0)[0].cpu().numpy().reshape(-1))
                    refine_rng = np.random.default_rng(7_000_000 + sample_id - 1000)
                    selected, info = refine_fast(coarse, score, reference, patch_id, patch_mass, keep,
                                                  kappa=4, seed=int(refine_rng.integers(1 << 30)),
                                                  patch_cache=patch_cache, placement="fps")
                    compensation = comp_weights_for_subset(selected, reference, len(reference), beta=1.0)
                    sampling = SamplingResult(selected, compensation ** -1, compensation, {"info": info})
                indices = sampling.indices
                pos = data.test.pos[local:local + 1, indices].to(device)
                context.set(None if method == "uniform" else torch.as_tensor(sampling.compensation, device=device))
                with torch.no_grad():
                    output = data.y_normalizer.decode(model(pos, pos))[0].cpu().numpy().reshape(-1)
                context.clear()
                reconstructed, fallback = reconstruct_field(indices, output, reference)
                reconstructed = reconstructed.reshape(-1)
                save_field(sample_dir / f"{method}_{int(budget * 1000):03d}_prediction.png", reconstructed,
                           f"{method} {budget:.1%} prediction")
                save_field(sample_dir / f"{method}_{int(budget * 1000):03d}_error.png", np.abs(reconstructed - target),
                           f"{method} {budget:.1%} absolute error")
                save_density(sample_dir / f"{method}_{int(budget * 1000):03d}_density.png",
                             reference, indices, sampling.sampling_density,
                             f"{method} {budget:.1%} sampling density")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(reference[:, 0], reference[:, 1], s=1, alpha=0.08, color="0.6")
                ax.scatter(reference[indices, 0], reference[indices, 1], s=2, alpha=0.8)
                ax.set_title(f"{method} {budget:.1%} sampled points")
                ax.set_aspect("equal")
                fig.tight_layout()
                fig.savefig(sample_dir / f"{method}_{int(budget * 1000):03d}_sampling.png", dpi=140)
                plt.close(fig)
                method_records[f"{method}:{budget:.3f}"] = {"count": int(len(indices)), "fallback": fallback}
        manifest["methods"][str(sample_id)] = method_records
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
