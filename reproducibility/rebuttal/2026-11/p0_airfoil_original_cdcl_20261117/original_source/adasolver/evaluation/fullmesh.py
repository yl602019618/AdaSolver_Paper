from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adasolver.data import PipeAdapter
from adasolver.data.base import BenchmarkSplit
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3Adapter, load_pipe_operator

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/pipe"
DEFAULT_CKPT = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/pipe_Transolver_varpts_best.pt"


@torch.no_grad()
def evaluate_fullmesh(split: BenchmarkSplit, model: Transolver3Adapter,
                     normalizer, device: str = "cpu", batch_size: int = 4,
                     nsample: int | None = None) -> list[float]:
    """Evaluate a frozen operator on a complete benchmark split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    available = len(split.sample_ids)
    count = available if nsample is None else min(nsample, available)
    if count < 1:
        raise ValueError("fullmesh evaluation requires at least one sample")
    normalizer.to(device)
    errors: list[float] = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        pos = split.pos[start:stop].to(device)
        fx = None if split.fx is None else split.fx[start:stop].to(device)
        pred = normalizer.decode(model(pos, fx))
        target = split.target[start:stop].to(device)
        errors.extend(relative_l2(pred, target).cpu().tolist())
    return errors


@torch.no_grad()
def evaluate_pipe_fullmesh(checkpoint: str, data_path: str, device: str = "cuda:0",
                           batch_size: int = 4, nsample: int | None = None) -> dict:
    data = PipeAdapter(data_path).load()
    model = load_pipe_operator(checkpoint, device=device, verify_legacy_hash=(checkpoint == DEFAULT_CKPT))
    errors = evaluate_fullmesh(data.test, model, data.y_normalizer, device, batch_size, nsample)
    values = torch.tensor(errors, dtype=torch.float64)
    return {
        "benchmark": "pipe",
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": len(errors),
        "mean_relative_l2": float(values.mean()),
        "std_relative_l2": float(values.std(unbiased=False)),
        "per_sample_relative_l2": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nsample", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_pipe_fullmesh(args.checkpoint, args.data_path, args.device,
                                    args.batch_size, args.nsample)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_relative_l2"}, indent=2))


if __name__ == "__main__":
    main()
