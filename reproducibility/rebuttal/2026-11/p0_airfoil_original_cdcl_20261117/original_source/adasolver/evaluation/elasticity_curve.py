from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import ElasticityAdapter
from adasolver.inference.pi import sample_elasticity_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.models import load_elasticity_operator
from adasolver.utils import budget_count
from adasolver.utils.interpolation import reconstruct_field

from .elasticity import DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_MANIFEST, DEFAULT_NORMALIZER, _check_normalizer

BUDGETS = (0.375, 0.5, 0.75)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _relative_l2_record(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    numerator = float(np.linalg.norm(prediction - target))
    denominator = float(np.linalg.norm(target))
    return numerator / max(denominator, 1e-12), numerator, denominator


@torch.no_grad()
def evaluate_elasticity_method(
    output: Path,
    method: str,
    sampler,
    checkpoint: str = DEFAULT_CHECKPOINT,
    manifest: str | Path = DEFAULT_MANIFEST,
    normalizer: str | Path = DEFAULT_NORMALIZER,
    data_path: str = DEFAULT_DATA,
    device: str = "cuda:0",
    samples: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    torch_device = torch.device(device)
    data = ElasticityAdapter(data_path).load()
    _check_normalizer(data, normalizer)
    model = load_elasticity_operator(checkpoint, manifest, device=torch_device)
    context, patched = install_compensation(model)
    data.y_normalizer.to(torch_device)
    split = data.test
    count = min(samples, len(split.sample_ids))
    full_count = split.pos.shape[1]
    raw: list[dict] = []

    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in seeds:
            for local_index in range(count):
                sample_id = split.sample_ids[local_index]
                reference = split.reference_coords(local_index).numpy()
                rng = np.random.default_rng(1000 * seed + local_index)
                sample_started = time.perf_counter()
                sampling = sampler(reference, keep, rng)
                sampling_seconds = time.perf_counter() - sample_started

                indices = sampling.indices
                pos = split.pos[local_index:local_index + 1, indices].to(torch_device)
                weights = torch.as_tensor(
                    sampling.compensation, dtype=torch.float32, device=torch_device,
                )
                context.set(None if method == "uniform" else weights)
                _synchronize(torch_device)
                inference_started = time.perf_counter()
                prediction = data.y_normalizer.decode(model(pos, None))
                _synchronize(torch_device)
                inference_seconds = time.perf_counter() - inference_started
                context.clear()

                sparse = prediction[0].cpu().numpy()
                reconstruction_started = time.perf_counter()
                reconstructed, fallback_count = reconstruct_field(indices, sparse, reference)
                reconstruction_seconds = time.perf_counter() - reconstruction_started
                target = split.target[local_index].numpy()
                relative_l2, numerator, denominator = _relative_l2_record(reconstructed, target)
                raw.append({
                    "benchmark": "elasticity",
                    "method": method,
                    "sample_id": sample_id,
                    "seed": seed,
                    "budget_fraction": budget,
                    "budget_count": keep,
                    "relative_l2": relative_l2,
                    "relative_l2_numerator": numerator,
                    "relative_l2_denominator": denominator,
                    "interpolation_fallback_count": fallback_count,
                    "sampling_seconds": sampling_seconds,
                    "acquisition_seconds": 0.0,
                    "inference_seconds": inference_seconds,
                    "reconstruction_seconds": reconstruction_seconds,
                    "sampling_metadata": sampling.metadata,
                })

    summary = {
        "benchmark": "elasticity",
        "method": method,
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": count,
        "seeds": list(seeds),
        "patched_attention_layers": patched,
        "budgets": {},
    }
    for budget in BUDGETS:
        records = [row for row in raw if row["budget_fraction"] == budget]
        values = np.asarray([row["relative_l2"] for row in records], dtype=np.float64)
        summary["budgets"][f"{budget:.3f}"] = {
            "count": budget_count(full_count, budget),
            "mean_relative_l2": float(values.mean()),
            "std_relative_l2_over_records": float(values.std()),
            "per_seed_mean": {
                str(seed): float(np.mean([
                    row["relative_l2"] for row in records if row["seed"] == seed
                ]))
                for seed in seeds
            },
            "mean_interpolation_fallback_count": float(np.mean([
                row["interpolation_fallback_count"] for row in records
            ])),
            "records": len(records),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.stem + "_raw.jsonl")
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_elasticity_uniform(output: Path, checkpoint: str = DEFAULT_CHECKPOINT,
                                  manifest: str | Path = DEFAULT_MANIFEST,
                                  normalizer: str | Path = DEFAULT_NORMALIZER,
                                  data_path: str = DEFAULT_DATA, device: str = "cuda:0",
                                  samples: int = 200,
                                  seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    return evaluate_elasticity_method(
        output, "uniform", lambda coords, count, rng: sample_uniform(len(coords), count, rng),
        checkpoint, manifest, normalizer, data_path, device, samples, seeds,
    )


def evaluate_elasticity_pi(output: Path, checkpoint: str = DEFAULT_CHECKPOINT,
                            manifest: str | Path = DEFAULT_MANIFEST,
                            normalizer: str | Path = DEFAULT_NORMALIZER,
                            data_path: str = DEFAULT_DATA, device: str = "cuda:0",
                            samples: int = 200,
                            seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    return evaluate_elasticity_method(
        output, "pi", sample_elasticity_pi, checkpoint, manifest, normalizer,
        data_path, device, samples, seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("uniform", "pi"), required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = evaluate_elasticity_uniform if args.method == "uniform" else evaluate_elasticity_pi
    summary = evaluator(
        args.output, args.checkpoint, args.manifest, args.normalizer,
        args.data_path, args.device, args.samples, tuple(args.seeds),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
