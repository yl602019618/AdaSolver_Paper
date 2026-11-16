from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import ElasticityAdapter
from adasolver.inference.ca import ElasticityCA, load_elasticity_ca_head
from adasolver.layers import install_compensation
from adasolver.models import load_elasticity_operator
from adasolver.utils import budget_count
from adasolver.utils.interpolation import reconstruct_field

from .elasticity import DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_MANIFEST, DEFAULT_NORMALIZER, _check_normalizer
from .elasticity_curve import BUDGETS, _relative_l2_record, _synchronize

DEFAULT_HEAD = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/elasticity_gradhead_tblk2_best.pt"
DEFAULT_HEAD_MANIFEST = Path(__file__).resolve().parents[2] / "checkpoints/elasticity/ca_head_legacy_manifest.json"


@torch.no_grad()
def evaluate_elasticity_ca(output: Path, checkpoint: str = DEFAULT_CHECKPOINT,
                             manifest: str | Path = DEFAULT_MANIFEST,
                             head_checkpoint: str = DEFAULT_HEAD,
                             head_manifest: str | Path = DEFAULT_HEAD_MANIFEST,
                             normalizer: str | Path = DEFAULT_NORMALIZER,
                             data_path: str = DEFAULT_DATA, device: str = "cuda:0",
                             samples: int = 200,
                             seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    torch_device = torch.device(device)
    data = ElasticityAdapter(data_path).load()
    _check_normalizer(data, normalizer)
    model = load_elasticity_operator(checkpoint, manifest, device=torch_device)
    head = load_elasticity_ca_head(model, head_checkpoint, head_manifest, torch_device)
    context, patched = install_compensation(model)
    data.y_normalizer.to(torch_device)
    split = data.test
    count = min(samples, len(split.sample_ids))
    references = [split.reference_coords(i).numpy() for i in range(count)]
    methods = [ElasticityCA(reference) for reference in references]
    full_count = split.pos.shape[1]
    score_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    raw: list[dict] = []

    for seed in seeds:
        for local_index in range(count):
            method = methods[local_index]
            rng = np.random.default_rng(1000 * seed + local_index)
            coarse_indices = rng.choice(full_count, method.coarse_count(), replace=False)
            coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(torch_device)
            _synchronize(torch_device)
            acquisition_started = time.perf_counter()
            standardized = head(coarse_pos, None)[0].cpu().numpy().reshape(-1)
            learned_score = head.to_gradient_score(standardized)
            _synchronize(torch_device)
            score_cache[(seed, local_index)] = (
                coarse_indices, learned_score, time.perf_counter() - acquisition_started,
            )

    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in seeds:
            for local_index in range(count):
                method = methods[local_index]
                reference = references[local_index]
                coarse_indices, learned_score, acquisition_seconds = score_cache[(seed, local_index)]
                refine_rng = np.random.default_rng(7_000_000 + 1000 * seed + local_index)
                sampling_started = time.perf_counter()
                sampling = method.refine(coarse_indices, learned_score, keep, refine_rng)
                sampling_seconds = time.perf_counter() - sampling_started
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
                    "benchmark": "elasticity", "method": "ca",
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

    summary = {"benchmark": "elasticity", "method": "ca",
               "checkpoint": str(Path(checkpoint).resolve()),
               "head_checkpoint": str(Path(head_checkpoint).resolve()),
               "samples": count, "seeds": list(seeds),
               "patched_attention_layers": patched, "budgets": {}}
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
    parser.add_argument("--head-checkpoint", default=DEFAULT_HEAD)
    parser.add_argument("--head-manifest", type=Path, default=DEFAULT_HEAD_MANIFEST)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_elasticity_ca(
        args.output, args.checkpoint, args.manifest, args.head_checkpoint,
        args.head_manifest, args.normalizer, args.data_path, args.device,
        args.samples, tuple(args.seeds),
    ), indent=2))


if __name__ == "__main__":
    main()
