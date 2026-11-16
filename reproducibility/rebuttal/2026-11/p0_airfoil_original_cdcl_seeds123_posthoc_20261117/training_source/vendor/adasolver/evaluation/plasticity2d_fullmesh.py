from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adasolver.data import Plasticity2DAdapter, unflatten_temporal_field
from adasolver.models import load_legacy_plasticity2d_operator


DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/plas/plas_N987_T20.mat"
DEFAULT_CHECKPOINT = "/mnt/ai-material-data/zhijun/CarD/Adasolver_official/checkpoints/plas_allT_Transolver_varpts_best.pt"
DEFAULT_MANIFEST = "checkpoints/plasticity2d/legacy_allt_manifest.json"


@torch.no_grad()
def evaluate_legacy_allt_fullmesh(checkpoint: str | Path, manifest: str | Path,
                                  data_path: str | Path, device: str = "cuda:0",
                                  batch_size: int = 8,
                                  nsample: int | None = None) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = Plasticity2DAdapter(data_path).load()
    model = load_legacy_plasticity2d_operator(checkpoint, manifest, device=device)
    split = data.test
    available = len(split.sample_ids)
    count = available if nsample is None else min(nsample, available)
    if count < 1:
        raise ValueError("fullmesh evaluation requires at least one sample")

    records = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        pos = split.pos[start:stop].to(device)
        fx = split.fx[start:stop].to(device)
        target = unflatten_temporal_field(split.target[start:stop].to(device), data.layout)
        prediction = unflatten_temporal_field(model(pos, fx), data.layout)
        numerator = torch.linalg.vector_norm(prediction - target, dim=(1, 3))
        denominator = torch.linalg.vector_norm(target, dim=(1, 3))
        if torch.any(denominator <= 1e-12):
            raise RuntimeError("legacy allT target contains a zero-norm predicted frame")
        per_frame = numerator / denominator
        for offset, values in enumerate(per_frame.cpu().numpy()):
            records.append({
                "sample_id": split.sample_ids[start + offset],
                "per_frame_relative_l2": values.tolist(),
                "sequence_mean_relative_l2": float(values.mean()),
                "terminal_relative_l2": float(values[-1]),
            })

    per_frame_values = np.asarray([record["per_frame_relative_l2"] for record in records])
    sequence_values = np.asarray([record["sequence_mean_relative_l2"] for record in records])
    terminal_values = np.asarray([record["terminal_relative_l2"] for record in records])
    return {
        "benchmark": "plasticity2d",
        "variant": "legacy_all_timestep",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "manifest": str(Path(manifest).expanduser().resolve()),
        "data_path": str(Path(data_path).expanduser().resolve()),
        "split": "legacy_test_last_80_[907,987)",
        "samples": len(records),
        "frames": data.layout.predicted_frames,
        "metric": "mean_of_per_frame_relative_l2",
        "mean_sequence_relative_l2": float(sequence_values.mean()),
        "std_sequence_relative_l2": float(sequence_values.std()),
        "mean_terminal_relative_l2": float(terminal_values.mean()),
        "mean_per_frame_relative_l2": per_frame_values.mean(axis=0).tolist(),
        "std_per_frame_relative_l2": per_frame_values.std(axis=0).tolist(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nsample", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_legacy_allt_fullmesh(
        args.checkpoint, args.manifest, args.data_path, args.device,
        args.batch_size, args.nsample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("records", "mean_per_frame_relative_l2",
                                     "std_per_frame_relative_l2")}, indent=2))


if __name__ == "__main__":
    main()
