from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import ElasticityAdapter
from adasolver.inference.cpg import ElasticityCPG
from adasolver.layers import install_compensation
from adasolver.models import load_elasticity_operator
from adasolver.utils import budget_count
from adasolver.utils.interpolation import reconstruct_field

from .elasticity import DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_MANIFEST, DEFAULT_NORMALIZER, _check_normalizer
from .elasticity_curve import BUDGETS, _relative_l2_record, _synchronize


@torch.no_grad()
def evaluate_elasticity_cpg(output: Path, checkpoint: str = DEFAULT_CHECKPOINT,
                              manifest: str | Path = DEFAULT_MANIFEST,
                              normalizer: str | Path = DEFAULT_NORMALIZER,
                              data_path: str = DEFAULT_DATA, device: str = "cuda:0",
                              samples: int = 200,
                              seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    if samples < 1 or not seeds:
        raise ValueError("samples and seeds must be non-empty")
    torch_device = torch.device(device)
    data = ElasticityAdapter(data_path).load()
    _check_normalizer(data, normalizer)
    model = load_elasticity_operator(checkpoint, manifest, device=torch_device)
    context, patched = install_compensation(model)
    data.y_normalizer.to(torch_device)
    split = data.test
    count = min(samples, len(split.sample_ids))
    full_count = split.pos.shape[1]
    references = [split.reference_coords(i).numpy() for i in range(count)]
    methods = [ElasticityCPG(reference) for reference in references]
    raw: list[dict] = []

    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in seeds:
            for local_index in range(count):
                reference = references[local_index]
                method = methods[local_index]
                rng = np.random.default_rng(1000 * seed + local_index)
                started = time.perf_counter()
                coarse_indices = rng.choice(full_count, method.coarse_count(), replace=False)
                coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(torch_device)
                context.clear()
                _synchronize(torch_device)
                acquisition_started = time.perf_counter()
                coarse_prediction = data.y_normalizer.decode(model(coarse_pos, None))
                _synchronize(torch_device)
                acquisition_seconds = time.perf_counter() - acquisition_started
                sampling = method.refine(
                    coarse_indices, coarse_prediction[0].cpu().numpy(), keep, budget, rng,
                )
                sampling_seconds = time.perf_counter() - started
                indices = sampling.indices
                pos = split.pos[local_index:local_index + 1, indices].to(torch_device)
                weights = torch.as_tensor(sampling.compensation, dtype=torch.float32, device=torch_device)
                context.set(weights)
                _synchronize(torch_device)
                inference_started = time.perf_counter()
                prediction = data.y_normalizer.decode(model(pos, None))
                _synchronize(torch_device)
                inference_seconds = time.perf_counter() - inference_started
                context.clear()
                reconstruction_started = time.perf_counter()
                reconstructed, fallback_count = reconstruct_field(
                    indices, prediction[0].cpu().numpy(), reference,
                )
                reconstruction_seconds = time.perf_counter() - reconstruction_started
                relative_l2, numerator, denominator = _relative_l2_record(
                    reconstructed, split.target[local_index].numpy(),
                )
                raw.append({
                    "benchmark": "elasticity", "method": "cpg",
                    "sample_id": split.sample_ids[local_index], "seed": seed,
                    "budget_fraction": budget, "budget_count": keep,
                    "relative_l2": relative_l2,
                    "relative_l2_numerator": numerator,
                    "relative_l2_denominator": denominator,
                    "interpolation_fallback_count": fallback_count,
                    "sampling_seconds": sampling_seconds,
                    "acquisition_seconds": acquisition_seconds,
                    "inference_seconds": inference_seconds,
                    "reconstruction_seconds": reconstruction_seconds,
                    "sampling_metadata": sampling.metadata,
                })

    summary = {"benchmark": "elasticity", "method": "cpg",
               "checkpoint": str(Path(checkpoint).resolve()), "samples": count,
               "seeds": list(seeds), "patched_attention_layers": patched, "budgets": {}}
    for budget in BUDGETS:
        records = [r for r in raw if r["budget_fraction"] == budget]
        values = np.asarray([r["relative_l2"] for r in records], dtype=np.float64)
        summary["budgets"][f"{budget:.3f}"] = {
            "count": budget_count(full_count, budget),
            "mean_relative_l2": float(values.mean()),
            "std_relative_l2_over_records": float(values.std()),
            "per_seed_mean": {str(seed): float(np.mean([
                r["relative_l2"] for r in records if r["seed"] == seed
            ])) for seed in seeds},
            "records": len(records),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8",
    )
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_elasticity_cpg(
        args.output, args.checkpoint, args.manifest, args.normalizer,
        args.data_path, args.device, args.samples, tuple(args.seeds),
    ), indent=2))


if __name__ == "__main__":
    main()
