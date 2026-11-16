from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from adasolver.data import DarcyAdapter
from adasolver.evaluation.fullmesh import evaluate_fullmesh
from adasolver.models import load_darcy_operator

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy"
DEFAULT_CHECKPOINT = "/mnt/ai-material-data/zhijun/CarD/Adasolver_official/checkpoints/darcy_Transolver_varpts_best.pt"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "checkpoints/darcy/legacy_manifest.json"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_normalizer(data, archive: str | Path, expected_sha256: str | None = None,
                      atol: float = 1e-4) -> None:
    archive_path = Path(archive).expanduser().resolve()
    if expected_sha256 is not None and _sha256(archive_path) != expected_sha256:
        raise ValueError(f"Darcy normalizer hash mismatch: {archive_path}")
    values = np.load(archive_path)
    for name, actual in (("mean", data.y_normalizer.mean.numpy()),
                         ("std", data.y_normalizer.std.numpy())):
        if name not in values:
            raise ValueError(f"Darcy normalizer archive lacks {name}")
        if not np.allclose(actual, values[name], atol=atol, rtol=0.0):
            raise ValueError(f"Darcy normalizer mismatch for {name}")


@torch.no_grad()
def evaluate_darcy_fullmesh(checkpoint: str = DEFAULT_CHECKPOINT,
                            manifest: str | Path = DEFAULT_MANIFEST,
                            normalizer: str | Path | None = None,
                            data_path: str = DEFAULT_DATA,
                            device: str = "cuda:0", batch_size: int = 4,
                            nsample: int | None = None) -> dict:
    data = DarcyAdapter(data_path).load()
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    normalizer_path = normalizer or metadata.get("normalizer")
    if normalizer_path is None:
        raise ValueError("Darcy normalizer path is required")
    metadata_normalizer_hash = metadata.get("normalizer_sha256")
    _check_normalizer(data, normalizer_path, metadata_normalizer_hash)
    model = load_darcy_operator(checkpoint, manifest, device=device)
    errors = evaluate_fullmesh(data.test, model, data.y_normalizer, device, batch_size, nsample)
    values = torch.tensor(errors, dtype=torch.float64)
    variant = ("legacy_migration_only" if metadata.get("role") == "migration_only"
               else "paper_matched_test_selected")
    return {
        "benchmark": "darcy",
        "variant": variant,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "normalizer": str(Path(normalizer_path).resolve()),
        "samples": len(errors),
        "mean_relative_l2": float(values.mean()),
        "std_relative_l2": float(values.std(unbiased=False)),
        "per_sample_relative_l2": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalizer")
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nsample", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_darcy_fullmesh(args.checkpoint, args.manifest, args.normalizer,
                                     args.data_path, args.device, args.batch_size, args.nsample)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample_relative_l2"}, indent=2))


if __name__ == "__main__":
    main()
