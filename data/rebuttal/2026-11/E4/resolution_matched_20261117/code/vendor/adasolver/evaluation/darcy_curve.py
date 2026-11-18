from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import DarcyAdapter
from adasolver.inference.ca import DarcyCA, load_darcy_ca_head
from adasolver.inference.cpg import DarcyCPG
from adasolver.inference.pi import sample_darcy_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.models import load_darcy_operator
from adasolver.utils import budget_count
from adasolver.utils.seed import make_numpy_rng
from adasolver.utils.interpolation import reconstruct_field

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy"
DEFAULT_CHECKPOINT = "checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_best.pt"
DEFAULT_MANIFEST = "checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_manifest.json"
BUDGETS = (0.375, 0.5, 0.75)


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _relative_l2_record(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    numerator = float(np.linalg.norm(prediction - target))
    denominator = float(np.linalg.norm(target))
    return numerator / max(denominator, 1e-12), numerator, denominator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_darcy_method(
    output: str | Path, method: str, checkpoint: str = DEFAULT_CHECKPOINT,
    manifest: str | Path = DEFAULT_MANIFEST, data_path: str = DEFAULT_DATA,
    device: str = "cuda:0", samples: int = 200, seeds: tuple[int, ...] = (0, 1, 2),
    pi_amp: float = 8.0, pi_beta: float = 0.75, pi_prior: str = "interface_contrast",
    pi_ell: float = 0.05, pi_score_power: float = 1.0,
    pi_feature: str = "multiscale", pi_feature_scales: tuple[int, ...] = (1, 2, 4, 8),
    pi_feature_rho: float = 0.25, pi_feature_cap: float = 2.0,
    cpg_fraction: float = 0.25, cpg_patch_count: int = 64,
    cpg_kappa: float = 4.0, cpg_beta: float = 0.75,
    cpg_acquisition: str = "grad", cpg_score_power: float = 1.0,
    cpg_clip_quantile: float | None = None, sample_start: int = 0,
    ca_head: str | Path | None = None, ca_manifest: str | Path | None = None,
    ca_coarse_fraction: float = 0.10, ca_patch_count: int = 32,
    ca_kappa: float = 2.0, ca_beta: float = 1.0,
) -> dict:
    if method not in {"uniform", "pi", "cpg", "ca"}:
        raise ValueError(f"unsupported Darcy method: {method}")
    if samples < 1 or not seeds:
        raise ValueError("samples and seeds must be non-empty")
    torch_device = torch.device(device)
    data = DarcyAdapter(data_path).load()
    model = load_darcy_operator(checkpoint, manifest, device=torch_device)
    context, patched = install_compensation(model)
    data.y_normalizer.to(torch_device)
    split = data.test
    if sample_start < 0 or sample_start >= len(split.sample_ids):
        raise ValueError("sample_start must address the test split")
    count = min(samples, len(split.sample_ids) - sample_start)
    reference = DarcyAdapter(data_path).reference_coords().numpy().astype(np.float64)
    full_count = len(reference)
    ca = None
    if method == "ca":
        if ca_head is None or ca_manifest is None:
            raise ValueError("ca_head and ca_manifest are required for method=ca")
        ca_operator = load_darcy_ca_head(model, ca_head, ca_manifest, device=torch_device)
        ca = DarcyCA(reference, ca_operator, patch_count=ca_patch_count,
                     coarse_fraction=ca_coarse_fraction, kappa=ca_kappa, beta=ca_beta)
    cpg = DarcyCPG(
        reference, patch_count=cpg_patch_count, coarse_fraction=cpg_fraction,
        kappa=cpg_kappa, beta=cpg_beta, acquisition=cpg_acquisition,
        score_power=cpg_score_power, clip_quantile=cpg_clip_quantile,
    )
    raw: list[dict] = []

    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in seeds:
            for local_index, sample_id in enumerate(split.sample_ids[sample_start:sample_start + count], start=sample_start):
                rng = make_numpy_rng(seed, sample_id)
                started = time.perf_counter()
                if method == "uniform":
                    sampling = sample_uniform(full_count, keep, rng)
                    acquisition_seconds = 0.0
                elif method == "pi":
                    coefficients = data.x_normalizer.decode(
                        split.fx[local_index].reshape(-1, 1)
                    ).numpy()
                    acquisition_started = time.perf_counter()
                    sampling = sample_darcy_pi(
                        coefficients, reference, keep, rng, amp=pi_amp, beta=pi_beta,
                        prior=pi_prior, ell=pi_ell, score_power=pi_score_power,
                        feature=pi_feature, scales=pi_feature_scales,
                        feature_rho=pi_feature_rho, feature_cap=pi_feature_cap,
                    )
                    acquisition_seconds = time.perf_counter() - acquisition_started
                elif method == "cpg":
                    coarse_indices = rng.choice(full_count, cpg.coarse_count(), replace=False)
                    coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(torch_device)
                    coarse_fx = split.fx[local_index:local_index + 1, coarse_indices].to(torch_device)
                    acquisition_started = time.perf_counter()
                    coarse_prediction = model(coarse_pos, coarse_fx)[0].cpu().numpy()
                    sampling = cpg.refine(coarse_indices, coarse_prediction, keep, rng)
                    acquisition_seconds = time.perf_counter() - acquisition_started
                else:
                    coarse_indices = rng.choice(full_count, ca.coarse_count(), replace=False)
                    coarse_pos = split.pos[local_index:local_index + 1, coarse_indices].to(torch_device)
                    coarse_fx = split.fx[local_index:local_index + 1, coarse_indices].to(torch_device)
                    acquisition_started = time.perf_counter()
                    sampling = ca.refine(coarse_indices, coarse_pos, coarse_fx, keep, rng)
                    acquisition_seconds = time.perf_counter() - acquisition_started
                sampling_seconds = time.perf_counter() - started
                indices = sampling.indices
                pos = split.pos[local_index:local_index + 1, indices].to(torch_device)
                fx = split.fx[local_index:local_index + 1, indices].to(torch_device)
                weights = torch.as_tensor(sampling.compensation, dtype=torch.float32, device=torch_device)
                context.clear()
                if method != "uniform":
                    context.set(weights)
                _sync(torch_device)
                infer_started = time.perf_counter()
                prediction = data.y_normalizer.decode(model(pos, fx))[0].cpu().numpy()
                _sync(torch_device)
                inference_seconds = time.perf_counter() - infer_started
                context.clear()
                reconstruct_started = time.perf_counter()
                reconstructed, fallback = reconstruct_field(indices, prediction, reference)
                reconstruction_seconds = time.perf_counter() - reconstruct_started
                target = split.target[local_index].numpy()
                error, numerator, denominator = _relative_l2_record(reconstructed, target)
                raw.append({
                    "benchmark": "darcy", "variant": "test_selected_operator_comparison",
                    "method": method, "sample_id": sample_id, "seed": seed,
                    "budget_fraction": budget, "budget_count": keep,
                    "selected_indices_sha256": _indices_hash(indices),
                    "relative_l2": error, "relative_l2_numerator": numerator,
                    "relative_l2_denominator": denominator,
                    "interpolation_fallback_count": fallback,
                    "sampling_seconds": sampling_seconds,
                    "acquisition_seconds": acquisition_seconds,
                    "inference_seconds": inference_seconds,
                    "reconstruction_seconds": reconstruction_seconds,
                    "sampling_metadata": sampling.metadata,
                })
    summary = {
        "benchmark": "darcy", "variant": "test_selected_operator_comparison",
        "method": method, "checkpoint": str(Path(checkpoint).resolve()),
        "manifest": str(Path(manifest).resolve()), "samples": count,
        "sample_start": sample_start, "seeds": list(seeds),
        "patched_attention_layers": patched, "budgets": {},
    }
    for budget in BUDGETS:
        rows = [row for row in raw if row["budget_fraction"] == budget]
        values = np.asarray([row["relative_l2"] for row in rows], dtype=np.float64)
        summary["budgets"][f"{budget:.3f}"] = {
            "count": budget_count(full_count, budget),
            "mean_relative_l2": float(values.mean()),
            "std_relative_l2_over_records": float(values.std()),
            "per_seed_mean": {str(seed): float(np.mean([
                row["relative_l2"] for row in rows if row["seed"] == seed
            ])) for seed in seeds},
            "mean_interpolation_fallback_count": float(np.mean([
                row["interpolation_fallback_count"] for row in rows
            ])), "records": len(rows),
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
    )
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("uniform", "pi", "cpg", "ca"), required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pi-amp", type=float, default=8.0)
    parser.add_argument("--pi-beta", type=float, default=0.75)
    parser.add_argument(
        "--pi-prior", choices=("interface_contrast", "interface_distance"),
        default="interface_contrast",
    )
    parser.add_argument("--pi-ell", type=float, default=0.05)
    parser.add_argument("--pi-score-power", type=float, default=1.0)
    parser.add_argument("--cpg-fraction", type=float, default=0.25)
    parser.add_argument("--cpg-patch-count", type=int, default=64)
    parser.add_argument("--cpg-kappa", type=float, default=4.0)
    parser.add_argument("--cpg-beta", type=float, default=0.75)
    parser.add_argument(
        "--cpg-acquisition", choices=("grad", "abs", "lap", "grad_lap"),
        default="grad",
    )
    parser.add_argument("--cpg-score-power", type=float, default=1.0)
    parser.add_argument("--cpg-clip-quantile", type=float, default=None)
    parser.add_argument("--ca-head", type=Path, default=None)
    parser.add_argument("--ca-manifest", type=Path, default=None)
    parser.add_argument("--ca-coarse-fraction", type=float, default=0.10)
    parser.add_argument("--ca-patch-count", type=int, default=32)
    parser.add_argument("--ca-kappa", type=float, default=2.0)
    parser.add_argument("--ca-beta", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate_darcy_method(
        args.output, args.method, args.checkpoint, args.manifest, args.data_path,
        args.device, args.samples, tuple(args.seeds),
        pi_amp=args.pi_amp, pi_beta=args.pi_beta, pi_prior=args.pi_prior,
        pi_ell=args.pi_ell, pi_score_power=args.pi_score_power,
        cpg_fraction=args.cpg_fraction, cpg_patch_count=args.cpg_patch_count,
        cpg_kappa=args.cpg_kappa, cpg_beta=args.cpg_beta,
        cpg_acquisition=args.cpg_acquisition,
        cpg_score_power=args.cpg_score_power,
        cpg_clip_quantile=args.cpg_clip_quantile,
        sample_start=args.sample_start,
        ca_head=args.ca_head, ca_manifest=args.ca_manifest,
        ca_coarse_fraction=args.ca_coarse_fraction, ca_patch_count=args.ca_patch_count,
        ca_kappa=args.ca_kappa, ca_beta=args.ca_beta,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
