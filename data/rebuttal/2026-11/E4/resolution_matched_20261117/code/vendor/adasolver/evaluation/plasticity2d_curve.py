from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import Plasticity2DAdapter, unflatten_temporal_field
from adasolver.evaluation.plasticity2d import reconstruct_temporal_field
from adasolver.inference.ca import lift_head_score, load_plasticity_ca_head, sample_plasticity_ca
from adasolver.inference.cpg import coarse_gradient_density, sample_plasticity_cpg
from adasolver.inference.pi import sample_plasticity_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.models import load_legacy_plasticity2d_operator
from adasolver.models.legacy_transolver import install_legacy_compensation
from adasolver.utils.seed import make_numpy_rng

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/plas/plas_N987_T20.mat"
DEFAULT_CHECKPOINT = "/mnt/ai-material-data/zhijun/CarD/Adasolver_official/checkpoints/plas_allT_Transolver_varpts_best.pt"
DEFAULT_MANIFEST = "checkpoints/plasticity2d/legacy_allt_manifest.json"
BUDGETS = (0.375, 0.5, 0.75)


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def evaluate_legacy_curve(method: str, checkpoint: str | Path, manifest: str | Path,
                           data_path: str | Path, output: str | Path,
                           device: str = "cuda:0", samples: int | None = None,
                           seeds: tuple[int, ...] = (0, 1, 2),
                           ca_checkpoint: str | Path | None = None,
                           ca_manifest: str | Path | None = None,
                           split_name: str = "test",
                           cpg_coarse_fraction: float = 0.25,
                           cpg_amp: float = 8.0,
                           cpg_beta: float = 1.0,
                           ca_coarse_fraction: float = 0.10,
                           ca_beta: float = 1.0) -> dict:
    data = Plasticity2DAdapter(data_path).load()
    model = load_legacy_plasticity2d_operator(checkpoint, manifest, device=device)
    compensation_context = install_legacy_compensation(model)
    ca_head = None
    if method == "ca":
        if ca_checkpoint is None or ca_manifest is None:
            raise ValueError("CA requires --ca-checkpoint and --ca-manifest")
        ca_head = load_plasticity_ca_head(model, ca_checkpoint, ca_manifest, device=device)
    split = data.splits[split_name]
    count = len(split.sample_ids) if samples is None else min(samples, len(split.sample_ids))
    if count < 1:
        raise ValueError("samples must select at least one test sample")
    reference = np.asarray(data.train.pos[0].numpy(), dtype=np.float64)
    targets = unflatten_temporal_field(split.target[:count], data.layout).numpy()
    raw = []
    for budget in BUDGETS:
        keep = int(round(data.shape[0] * data.shape[1] * budget))
        for seed in seeds:
            for local_index, sample_id in enumerate(split.sample_ids[:count]):
                rng = make_numpy_rng(seed, sample_id)
                started = time.perf_counter()
                if method == "uniform":
                    sampling = sample_uniform(len(reference), keep, rng)
                elif method == "pi":
                    sampling = sample_plasticity_pi(keep, rng, data.shape)
                elif method == "cpg":
                    coarse_count = max(8, int(round(cpg_coarse_fraction * len(reference))))
                    coarse_indices = rng.choice(len(reference), coarse_count, replace=False)
                    coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(device)
                    coarse_fx = split.fx[local_index:local_index + 1, coarse_indices].to(device)
                    with torch.no_grad():
                        coarse_prediction = model(coarse_pos, coarse_fx)[0].cpu().numpy()
                    density = coarse_gradient_density(reference[coarse_indices], coarse_prediction,
                                                      reference, k=8, amp=cpg_amp)
                    sampling = sample_plasticity_cpg(coarse_indices, density, keep, rng, beta=cpg_beta)
                elif method == "ca":
                    coarse_count = max(8, int(round(ca_coarse_fraction * len(reference))))
                    coarse_indices = rng.choice(len(reference), coarse_count, replace=False)
                    coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(device)
                    coarse_fx = split.fx[local_index:local_index + 1, coarse_indices].to(device)
                    with torch.no_grad():
                        coarse_standardized = ca_head(coarse_pos, coarse_fx)[0].cpu().numpy().reshape(-1)
                    coarse_score = ca_head.decode_score(coarse_standardized)
                    density = lift_head_score(reference[coarse_indices], coarse_score, reference)
                    sampling = sample_plasticity_ca(coarse_indices, density, keep, rng, beta=ca_beta)
                else:
                    raise ValueError(f"unsupported method: {method}")
                sample_time = time.perf_counter() - started
                pos = split.pos[local_index:local_index + 1, sampling.indices].to(device)
                fx = split.fx[local_index:local_index + 1, sampling.indices].to(device)
                weights = torch.as_tensor(sampling.compensation, dtype=torch.float32, device=device)[None]
                compensation_context.clear()
                if method != "uniform":
                    compensation_context.set(weights)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()
                with torch.no_grad():
                    prediction = model(pos, fx, )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                infer_time = time.perf_counter() - infer_start
                sparse = prediction[0].detach().cpu().numpy()
                compensation_context.clear()
                reconstructed, fallback, metrics = reconstruct_temporal_field(
                    sampling.indices, sparse, reference, data.layout, targets[local_index])
                raw.append({
                    "benchmark": "plasticity2d", "method": method,
                    "sample_id": sample_id, "seed": seed,
                    "budget_fraction": budget, "budget_count": keep,
                    "selected_indices_sha256": _indices_hash(sampling.indices),
                    "per_frame_relative_l2": metrics.per_frame_relative_l2.tolist(),
                    "sequence_mean_relative_l2": metrics.sequence_mean_relative_l2,
                    "terminal_relative_l2": metrics.terminal_relative_l2,
                    "interpolation_fallback_count": fallback,
                    "sampling_seconds": sample_time, "inference_seconds": infer_time,
                    "status": "ok", "sampling_metadata": sampling.metadata,
                })
    summary = {"benchmark": "plasticity2d", "variant": "legacy_all_timestep",
               "method": method, "samples": count, "seeds": list(seeds),
               "budgets": {}}
    for budget in BUDGETS:
        rows = [r for r in raw if r["budget_fraction"] == budget]
        vectors = np.asarray([r["per_frame_relative_l2"] for r in rows])
        summary["budgets"][f"{budget:.3f}"] = {
            "count": keep if False else int(round(len(reference) * budget)),
            "mean_sequence_relative_l2": float(np.mean([r["sequence_mean_relative_l2"] for r in rows])),
            "mean_per_frame_relative_l2": vectors.mean(axis=0).tolist(),
            "records": len(rows),
        }
    summary["raw_records"] = raw
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("uniform", "pi", "cpg", "ca"), required=True)
    parser.add_argument("--ca-checkpoint")
    parser.add_argument("--ca-manifest")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_legacy_curve(args.method, args.checkpoint, args.manifest,
                                   args.data_path, args.output, args.device,
                                   args.samples, tuple(args.seeds), args.ca_checkpoint,
                                   args.ca_manifest)
    print(json.dumps({"method": result["method"], "samples": result["samples"],
                      "budgets": result["budgets"]}, indent=2))


if __name__ == "__main__":
    main()
