from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adasolver.data import ElasticityAdapter
from adasolver.inference.ca import ElasticityCA, load_elasticity_ca_head
from adasolver.inference.cpg import ElasticityCPG
from adasolver.inference.pi import sample_elasticity_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.models import load_elasticity_operator
from adasolver.utils import budget_count
from adasolver.utils.interpolation import reconstruct_field

from adasolver.evaluation.elasticity import (
    DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_MANIFEST, DEFAULT_NORMALIZER,
    _check_normalizer,
)
from adasolver.evaluation.elasticity_ca_curve import DEFAULT_HEAD, DEFAULT_HEAD_MANIFEST
from adasolver.evaluation.elasticity_curve import BUDGETS

METHOD_COLORS = {"uniform": "#2a78d6", "pi": "#1baf7a", "cpg": "#eda100", "ca": "#4a3aa7"}


def scatter_field(path: Path, coords: np.ndarray, values: np.ndarray, title: str,
                  colorbar: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    points = ax.scatter(coords[:, 0], coords[:, 1], c=values.reshape(-1), s=9,
                        cmap="viridis", linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(points, ax=ax, label=colorbar)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def sample_points(path: Path, coords: np.ndarray, indices: np.ndarray,
                  density: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(coords[:, 0], coords[:, 1], s=5, color="#b8b8b8", alpha=0.35,
               linewidths=0, rasterized=True)
    points = ax.scatter(coords[indices, 0], coords[indices, 1], c=density, s=12,
                        cmap="magma", linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(points, ax=ax, label="sampling density")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def select_representatives(errors: list[float]) -> list[int]:
    order = np.argsort(np.asarray(errors))
    return [int(order[0]), int(order[len(order) // 2]), int(order[-1])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/elasticity/figures"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--head-checkpoint", default=DEFAULT_HEAD)
    parser.add_argument("--head-manifest", type=Path, default=DEFAULT_HEAD_MANIFEST)
    args = parser.parse_args()

    device = torch.device(args.device)
    data = ElasticityAdapter(args.data_path).load()
    _check_normalizer(data, args.normalizer)
    model = load_elasticity_operator(args.checkpoint, args.manifest, device=device)
    head = load_elasticity_ca_head(model, args.head_checkpoint, args.head_manifest, device)
    context, _ = install_compensation(model)
    data.y_normalizer.to(device)
    split = data.test
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fullmesh errors are used only to choose fixed representative samples.
    full_errors = []
    full_predictions = []
    for local_index in range(len(split.sample_ids)):
        with torch.no_grad():
            prediction = data.y_normalizer.decode(
                model(split.pos[local_index:local_index + 1].to(device), None),
            )[0].cpu().numpy()
        full_predictions.append(prediction)
        target = split.target[local_index].numpy()
        full_errors.append(float(np.linalg.norm(prediction - target) /
                                max(np.linalg.norm(target), 1e-12)))
    representatives = select_representatives(full_errors)
    manifest = {
        "benchmark": "elasticity", "operator": str(Path(args.checkpoint).resolve()),
        "selection": "lowest/median/highest official fullmesh relative L2",
        "sample_ids": [split.sample_ids[i] for i in representatives],
        "local_indices": representatives, "seed": 0, "budgets": list(BUDGETS),
        "methods": {},
    }

    for local_index in representatives:
        sample_id = split.sample_ids[local_index]
        sample_dir = args.output_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        coords = split.pos[local_index].numpy()
        target = split.target[local_index].numpy()
        scatter_field(sample_dir / "gt_stress.png", coords, target,
                      f"Elasticity sample {sample_id}: ground truth stress", "von-Mises stress")
        scatter_field(sample_dir / "fullmesh_prediction.png", coords, full_predictions[local_index],
                      f"Elasticity sample {sample_id}: fullmesh prediction", "von-Mises stress")
        scatter_field(sample_dir / "fullmesh_error.png", coords,
                      np.abs(full_predictions[local_index] - target),
                      f"Elasticity sample {sample_id}: fullmesh absolute error", "absolute error")
        method_records = {}
        for budget in BUDGETS:
            keep = budget_count(len(coords), budget)
            for method_name in ("uniform", "pi", "cpg", "ca"):
                rng = np.random.default_rng(local_index)
                if method_name == "uniform":
                    sampling = sample_uniform(len(coords), keep, rng)
                elif method_name == "pi":
                    sampling = sample_elasticity_pi(coords, keep, rng)
                else:
                    coarse_method = ElasticityCPG(coords) if method_name == "cpg" else ElasticityCA(coords)
                    coarse = rng.choice(len(coords), coarse_method.coarse_count(), replace=False)
                    coarse_pos = split.pos[local_index:local_index + 1, coarse].to(device)
                    with torch.no_grad():
                        if method_name == "cpg":
                            coarse_prediction = data.y_normalizer.decode(model(coarse_pos, None))[0].cpu().numpy()
                            score = coarse_method.refine(coarse, coarse_prediction, keep, budget, rng)
                        else:
                            learned = head(coarse_pos, None)[0].cpu().numpy().reshape(-1)
                            refine_rng = np.random.default_rng(7_000_000 + local_index)
                            score = coarse_method.refine(coarse, head.to_gradient_score(learned), keep, refine_rng)
                    sampling = score
                indices = sampling.indices
                context.set(None if method_name == "uniform" else
                            torch.as_tensor(sampling.compensation, dtype=torch.float32, device=device))
                with torch.no_grad():
                    prediction = data.y_normalizer.decode(
                        model(split.pos[local_index:local_index + 1, indices].to(device), None),
                    )[0].cpu().numpy()
                context.clear()
                reconstructed, fallback = reconstruct_field(indices, prediction, coords)
                error = np.abs(reconstructed - target)
                tag = f"{method_name}_{int(round(budget * 1000)):03d}"
                sample_points(sample_dir / f"{tag}_sampling.png", coords, indices,
                              sampling.sampling_density, f"{method_name} {budget:.1%}: sampled points")
                scatter_field(sample_dir / f"{tag}_prediction.png", coords, reconstructed,
                              f"{method_name} {budget:.1%}: reconstructed stress", "von-Mises stress")
                scatter_field(sample_dir / f"{tag}_error.png", coords, error,
                              f"{method_name} {budget:.1%}: absolute error", "absolute error")
                method_records[f"{method_name}:{budget:.3f}"] = {
                    "count": int(len(indices)), "fallback": int(fallback),
                    "relative_l2": float(np.linalg.norm(reconstructed - target) /
                                         max(np.linalg.norm(target), 1e-12)),
                }
        manifest["methods"][str(sample_id)] = method_records
    (args.output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )

    comparison = json.loads(Path("results/elasticity/summary/official_comparison.json").read_text())
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method, values in comparison["methods"].items():
        ax.plot([b * 100 for b in BUDGETS], values["curve"], marker="o", linewidth=2,
                label=f"{method} (rAUEC={values['rAUEC_vs_uniform']:.3f})",
                color=METHOD_COLORS[method])
    ax.set_xlabel("final budget (%)")
    ax.set_ylabel("full-field relative L2")
    ax.set_title("Elasticity official accuracy versus final point budget")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "official_budget_curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
