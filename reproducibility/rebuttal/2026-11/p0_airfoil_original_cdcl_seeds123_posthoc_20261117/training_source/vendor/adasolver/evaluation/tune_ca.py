from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.data import PipeAdapter
from adasolver.inference.adaptive import build_patches, comp_weights_for_subset
from adasolver.inference.adaptive_fast import build_patch_cache, refine_once
from adasolver.inference.ca import PipeCAHead
from adasolver.inference.acquisition import grad_score
from adasolver.layers import install_compensation
from adasolver.models import load_pipe_operator
from adasolver.utils.budget import budget_count
from adasolver.utils.interpolation import index_grid, reconstruct_field

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"
DEFAULT_OPERATOR = "checkpoints/pipe/pipe_transolver3_varpts_nmin040_best.pt"
DEFAULT_HEAD = "checkpoints/pipe/pipe_ca_head_nmin040_best.pt"
BUDGETS = (0.375, 0.5, 0.75)


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def rel2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--method", choices=("ca", "cpg"), default="ca")
    parser.add_argument("--head", default=DEFAULT_HEAD)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--samples", type=int, default=50,
                        help="number of samples when sample_start is zero")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None,
                        help="exclusive test index; use with --confirm-frozen for sealed confirmation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--r0s", type=float, nargs="+", default=[0.10])
    parser.add_argument("--Ks", type=int, nargs="+", default=[48])
    parser.add_argument("--placements", nargs="+", default=["random", "fps"], choices=["random", "fps"])
    parser.add_argument("--kappas", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--output", type=Path, default=Path("results/pipe/summary/ca_tuning_calibration.json"))
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="required when samples exceeds the calibration subset")
    args = parser.parse_args()
    if args.samples > 50 and not args.confirm_frozen:
        raise SystemExit("refusing non-calibration search: pass --confirm-frozen only for a frozen config")

    data = PipeAdapter(args.data_path).load()
    sample_start = max(0, args.sample_start)
    sample_end = args.sample_end if args.sample_end is not None else sample_start + args.samples
    if sample_end <= sample_start or sample_end > len(data.test.sample_ids):
        raise SystemExit(f"invalid test range [{sample_start}, {sample_end})")
    count = sample_end - sample_start
    device = torch.device(args.device)
    model = load_pipe_operator(args.operator, device=device)
    context, patched = install_compensation(model)
    head = None
    if args.method == "ca":
        head = PipeCAHead(model).to(device).eval()
        head.head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=False), strict=True)
    data.y_normalizer.to(device)
    reference = index_grid(data.shape)
    target = data.test.target[sample_start:sample_end].numpy()
    full_count = len(reference)

    # Uniform is computed once and reused as the calibration denominator.
    uniform_values = {budget: [] for budget in BUDGETS}
    for budget in BUDGETS:
        keep = budget_count(full_count, budget)
        for seed in args.seeds:
            for local_index in range(count):
                data_index = sample_start + local_index
                rng = np.random.default_rng(1000 * seed + data_index)
                indices = rng.choice(full_count, keep, replace=False)
                pos = data.test.pos[data_index:data_index + 1, indices].to(device)
                context.clear()
                with torch.no_grad():
                    prediction = data.y_normalizer.decode(model(pos, pos))[0].cpu().numpy()
                reconstructed, _ = reconstruct_field(indices, prediction, reference)
                uniform_values[budget].append(rel2(reconstructed, target[local_index]))

    # Cache coarse indices and learned scores. They depend on r0, seed, and sample,
    # but not on K, placement, kappa, or beta.
    coarse_cache = {}
    for r0 in args.r0s:
        coarse_count = budget_count(full_count, r0)
        for seed in args.seeds:
            for local_index in range(count):
                data_index = sample_start + local_index
                rng = np.random.default_rng(1000 * seed + data_index)
                coarse_indices = rng.choice(full_count, coarse_count, replace=False)
                pos = data.test.pos[data_index:data_index + 1, coarse_indices].to(device)
                context.clear()
                with torch.no_grad():
                    coarse_prediction = data.y_normalizer.decode(model(pos, pos))[0].cpu().numpy()
                if args.method == "ca":
                    with torch.no_grad():
                        standardized = head(pos, pos)[0].cpu().numpy().reshape(-1)
                    score = PipeCAHead.to_gradient_score(standardized)
                else:
                    score = grad_score(reference[coarse_indices], coarse_prediction, k=8)
                coarse_cache[(r0, seed, local_index)] = (coarse_indices, score)

    results = []
    raw_records = []
    for r0, K, placement, kappa, beta in itertools.product(
            args.r0s, args.Ks, args.placements, args.kappas, args.betas):
        config = {"r0": r0, "K": K, "placement": placement,
                  "kappa": kappa, "beta": beta}
        cid = config_hash(config)
        patch_id, patch_mass = build_patches(reference, K, seed=0)
        patch_cache = build_patch_cache(reference, patch_id, K)
        per_budget = {}
        started = time.perf_counter()
        for budget in BUDGETS:
            keep = budget_count(full_count, budget)
            values = []
            for seed in args.seeds:
                for local_index in range(count):
                    data_index = sample_start + local_index
                    coarse_indices, score = coarse_cache[(r0, seed, local_index)]
                    refine_rng = np.random.default_rng(7_000_000 + 1000 * seed + data_index)
                    selected, info = refine_once(
                        coarse_indices, score, reference, patch_id, patch_mass, keep,
                        kappa=kappa, seed=int(refine_rng.integers(1 << 30)),
                        patch_cache=patch_cache, placement=placement)
                    compensation = comp_weights_for_subset(selected, reference, full_count, beta=beta)
                    pos = data.test.pos[data_index:data_index + 1, selected].to(device)
                    context.set(torch.as_tensor(compensation, dtype=torch.float32, device=device))
                    with torch.no_grad():
                        prediction = data.y_normalizer.decode(model(pos, pos))[0].cpu().numpy()
                    context.clear()
                    reconstructed, fallback = reconstruct_field(selected, prediction, reference)
                    value = rel2(reconstructed, target[local_index])
                    values.append(value)
                    raw_records.append({
                        "config_id": cid, "config": config, "budget": budget,
                        "seed": seed, "sample_id": data.test.sample_ids[data_index],
                        "relative_l2": value, "fallback_count": fallback,
                        "n_selected": len(selected), "refine_info": info,
                    })
            mean = float(np.mean(values))
            uniform_mean = float(np.mean(uniform_values[budget]))
            per_budget[f"{budget:.3f}"] = {
                "mean_relative_l2": mean,
                "uniform_mean_relative_l2": uniform_mean,
                "relative_to_uniform": mean / uniform_mean,
                "records": len(values),
            }
        method_curve = [per_budget[f"{budget:.3f}"]["mean_relative_l2"] for budget in BUDGETS]
        uniform_curve = [float(np.mean(uniform_values[budget])) for budget in BUDGETS]
        rauec = float(np.trapezoid(method_curve, BUDGETS) / np.trapezoid(uniform_curve, BUDGETS))
        results.append({"config_id": cid, "config": config, "rAUEC": rauec,
                        "budgets": per_budget, "elapsed_seconds": time.perf_counter() - started})
        print(f"{cid} {config} rAUEC={rauec:.6f}", flush=True)

    results.sort(key=lambda row: row["rAUEC"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name(args.output.stem + "_raw.jsonl")
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_records), encoding="utf-8")
    payload = {
        "benchmark": "pipe", "method": args.method,
        "stage": "frozen_confirmation" if args.confirm_frozen else "calibration_search",
        "operator": str(Path(args.operator).resolve()), "head": str(Path(args.head).resolve()),
        "samples": count, "seeds": args.seeds, "budgets": list(BUDGETS),
        "candidate_count": len(results), "selection": "minimum calibration rAUEC",
        "top": results[:10], "all_results": results,
        "raw_records": len(raw_records), "raw_path": str(raw_path),
        "disclosure": ("Calibration-only search; do not use top result as final test claim without sealed confirmation."
                       if not args.confirm_frozen else
                       "Frozen configuration confirmation; no candidate selection was performed on this sample range."),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
