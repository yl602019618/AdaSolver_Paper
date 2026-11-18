from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adasolver.data import ElasticityAdapter
from adasolver.evaluation.fullmesh import evaluate_fullmesh
from adasolver.models import load_elasticity_operator

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/elasticity"
DEFAULT_CHECKPOINT = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/elasticity_Transolver_varpts_best.pt"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "checkpoints/elasticity/legacy_t3_manifest.json"
DEFAULT_NORMALIZER = "/mnt/ai-material-data/zhijun/Adasolver_3_official/checkpoints/elasticity_Transolver_varpts_ynorm.npz"


def _check_normalizer(data, archive: str | Path, atol: float = 1e-4) -> None:
    values = np.load(archive)
    for name, actual in (("mean", data.y_normalizer.mean.numpy()),
                         ("std", data.y_normalizer.std.numpy())):
        if name not in values:
            raise ValueError(f"Elasticity normalizer archive lacks {name}")
        if not np.allclose(actual, values[name], atol=atol, rtol=0.0):
            raise ValueError(f"Elasticity normalizer mismatch for {name}")


@torch.no_grad()
def evaluate_elasticity_fullmesh(checkpoint: str = DEFAULT_CHECKPOINT,
                                  manifest: str | Path = DEFAULT_MANIFEST,
                                  normalizer: str | Path = DEFAULT_NORMALIZER,
                                  data_path: str = DEFAULT_DATA,
                                  device: str = "cuda:0", batch_size: int = 1,
                                  nsample: int | None = None) -> dict:
    data = ElasticityAdapter(data_path).load()
    _check_normalizer(data, normalizer)
    model = load_elasticity_operator(checkpoint, manifest, device=device)
    errors = evaluate_fullmesh(data.test, model, data.y_normalizer, device, batch_size, nsample)
    values = torch.tensor(errors, dtype=torch.float64)
    return {
        "benchmark": "elasticity",
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": len(errors),
        "mean_relative_l2": float(values.mean()),
        "std_relative_l2": float(values.std(unbiased=False)),
        "per_sample_relative_l2": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsample", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_elasticity_fullmesh(
        args.checkpoint, args.manifest, args.normalizer, args.data_path,
        args.device, args.batch_size, args.nsample,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_relative_l2"}, indent=2))


if __name__ == "__main__":
    main()
