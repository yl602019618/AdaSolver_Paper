from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import PipeAdapter
from adasolver.inference.ca import PipeCA, PipeCAHead
from adasolver.inference.cpg import PipeCPG
from adasolver.inference.pi import pipe_wall_distance, sample_pipe_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.metrics import relative_area_under_error_curve
from adasolver.models import load_pipe_operator
from adasolver.utils import budget_count
from adasolver.utils.interpolation import index_grid, reconstruct_field

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"
DEFAULT_CKPT = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/pipe_Transolver_varpts_best.pt"
DEFAULT_CA_HEAD = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/pipe_gradhead_tblk2_best.pt"
BUDGETS = (0.375, 0.5, 0.75)


def _relative_l2_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + 1e-12))


@torch.no_grad()
def evaluate_method(method: str, checkpoint: str, data_path: str, device: str,
                    samples: int, seeds: tuple[int, ...], output: Path,
                    ca_head_checkpoint: str = DEFAULT_CA_HEAD) -> dict:
    data = PipeAdapter(data_path).load()
    model = load_pipe_operator(checkpoint, device=device,
                               verify_legacy_hash=(str(Path(checkpoint).resolve()) == DEFAULT_CKPT))
    context, patched = install_compensation(model)
    data.y_normalizer.to(device)
    count = min(samples, len(data.test.sample_ids))
    reference = index_grid(data.shape)
    full_count = len(reference)
    raw = []
    distances = None
    adaptive_method = None
    ca_head = None
    if method == "pi":
        distances = [pipe_wall_distance(data.test.pos[i].numpy(), data.shape) for i in range(count)]
    elif method == "cpg":
        adaptive_method = PipeCPG(reference)
    elif method == "ca":
        adaptive_method = PipeCA(reference)
        ca_head = PipeCAHead(model).to(device).eval()
        head_state = torch.load(ca_head_checkpoint, map_location="cpu", weights_only=False)
        ca_head.head.load_state_dict(head_state, strict=True)

    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in seeds:
            for local_index in range(count):
                sample_id = data.test.sample_ids[local_index]
                rng = np.random.default_rng(1000 * seed + local_index)
                started = time.perf_counter()
                acquisition_time = 0.0
                if method == "uniform":
                    sampling = sample_uniform(full_count, keep, rng)
                elif method == "pi":
                    sampling = sample_pipe_pi(distances[local_index], keep, rng)
                elif method in ("cpg", "ca"):
                    coarse_count = budget_count(full_count, adaptive_method.coarse_fraction)
                    coarse_indices = rng.choice(full_count, coarse_count, replace=False)
                    coarse_pos = data.test.pos[local_index:local_index + 1, coarse_indices].to(device)
                    context.clear()
                    torch.cuda.synchronize()
                    acquisition_started = time.perf_counter()
                    if method == "cpg":
                        coarse_prediction = data.y_normalizer.decode(model(coarse_pos, coarse_pos))
                        score_or_prediction = coarse_prediction[0].cpu().numpy()
                    else:
                        standardized = ca_head(coarse_pos, coarse_pos)[0].cpu().numpy().reshape(-1)
                        score_or_prediction = ca_head.to_gradient_score(standardized)
                    torch.cuda.synchronize()
                    acquisition_time = time.perf_counter() - acquisition_started
                    # CA's historical evaluator uses a separate refinement RNG. Keeping
                    # this stream independent is required for exact migration comparison.
                    refine_rng = (np.random.default_rng(7_000_000 + 1000 * seed + local_index)
                                  if method == "ca" else rng)
                    sampling = adaptive_method.refine(
                        coarse_indices, score_or_prediction, keep, budget, refine_rng)
                else:
                    raise ValueError(f"unsupported method: {method}")
                sample_time = time.perf_counter() - started

                indices = sampling.indices
                pos = data.test.pos[local_index:local_index + 1, indices].to(device)
                weights = torch.as_tensor(sampling.compensation, dtype=torch.float32, device=device)
                context.set(None if method == "uniform" else weights)
                torch.cuda.synchronize()
                infer_started = time.perf_counter()
                prediction = data.y_normalizer.decode(model(pos, pos))
                torch.cuda.synchronize()
                inference_time = time.perf_counter() - infer_started
                context.clear()

                sparse = prediction[0].cpu().numpy()
                recon_started = time.perf_counter()
                reconstructed, fallback_count = reconstruct_field(indices, sparse, reference)
                reconstruction_time = time.perf_counter() - recon_started
                target = data.test.target[local_index].numpy()
                raw.append({
                    "method": method,
                    "sample_id": sample_id,
                    "seed": seed,
                    "budget_fraction": budget,
                    "budget_count": keep,
                    "relative_l2": _relative_l2_numpy(reconstructed, target),
                    "interpolation_fallback_count": fallback_count,
                    "sampling_seconds": sample_time,
                    "acquisition_seconds": acquisition_time,
                    "inference_seconds": inference_time,
                    "reconstruction_seconds": reconstruction_time,
                    "sampling_metadata": sampling.metadata,
                })
                print(f"method={method} budget={budget:.3f} seed={seed} "
                      f"sample={sample_id} rel={raw[-1]['relative_l2']:.6f}", flush=True)

    summary = {"method": method, "samples": count, "seeds": list(seeds),
               "patched_attention_layers": patched, "budgets": {}}
    for budget in BUDGETS:
        records = [row for row in raw if row["budget_fraction"] == budget]
        values = np.asarray([row["relative_l2"] for row in records])
        per_seed = {
            str(seed): float(np.mean([row["relative_l2"] for row in records if row["seed"] == seed]))
            for seed in seeds
        }
        summary["budgets"][f"{budget:.3f}"] = {
            "count": budget_count(full_count, budget),
            "mean_relative_l2": float(values.mean()),
            "std_relative_l2_over_records": float(values.std()),
            "per_seed_mean": per_seed,
            "records": len(records),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_name(output.stem + "_raw.jsonl")
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def compare_with_uniform(method_summary: dict, uniform_summary: dict) -> float:
    method = [method_summary["budgets"][f"{budget:.3f}"]["mean_relative_l2"] for budget in BUDGETS]
    uniform = [uniform_summary["budgets"][f"{budget:.3f}"]["mean_relative_l2"] for budget in BUDGETS]
    return relative_area_under_error_curve(BUDGETS, method, uniform)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("uniform", "pi", "cpg", "ca"), required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ca-head-checkpoint", default=DEFAULT_CA_HEAD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_method(args.method, args.checkpoint, args.data_path, args.device,
                              args.samples, tuple(args.seeds), args.output,
                              args.ca_head_checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
