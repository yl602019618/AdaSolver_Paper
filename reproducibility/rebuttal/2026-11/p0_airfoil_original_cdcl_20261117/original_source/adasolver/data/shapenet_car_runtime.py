"""Load named Car samples with the operator's frozen preprocessing statistics."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from adasolver.data.shapenet_car import CarSample, ShapeNetCarAdapter, _load_normalizers, _unpack
from adasolver.inference.shapenet_car_contract import (
    DEFAULT_MANIFEST, DEFAULT_SPLITS, make_splits, sample_fold, validate_splits,
)
from adasolver.models.adapter import Transolver3ModelSpec
from adasolver.models.checkpoint import load_operator, sha256_file


class CarRuntime:
    def __init__(self, manifest=DEFAULT_MANIFEST, data_path=None, *, require_variable=True):
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.metadata = json.loads(self.manifest_path.read_text())
        meta = self.metadata
        if meta.get("benchmark") != "shapenet_car" or meta.get("architecture") != "Transolver3":
            raise ValueError("expected a ShapeNet-Car Transolver3 manifest")
        if require_variable and (meta.get("nmin_fraction") != .4 or
                                 meta.get("nmin_count") != round(.4 * meta["nfull"])):
            raise ValueError("adaptive evaluation requires nmin_fraction=0.4")
        self.spec = Transolver3ModelSpec(**meta["model"])
        if (self.spec.space_dim, self.spec.fun_dim, self.spec.out_dim) != (3, 7, 4):
            raise ValueError("Car operator dimensions must be 3/7/4")
        def asset(key):
            path = Path(meta[key]).expanduser()
            return path if path.is_absolute() else self.manifest_path.parent / path
        self.checkpoint, self.normalizer_path = asset("checkpoint"), asset("normalizer")
        for key, path in (("checkpoint", self.checkpoint), ("normalizer", self.normalizer_path)):
            expected = meta.get(key + "_sha256")
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"{key} hash mismatch or missing hash")
        self.data_path = Path(data_path or os.environ.get("SHAPENET_CAR_DATA") or meta["data_path"]).expanduser().resolve()
        self.input_normalizer, self.output_normalizer = _load_normalizers(self.normalizer_path)

    def load_operator(self, device="cpu"):
        operator = load_operator(self.checkpoint, self.spec, device,
                                 self.metadata["checkpoint_sha256"])
        operator.requires_grad_(False)
        return operator.eval()

    def sample_ids(self):
        base = self.data_path / "preprocessed_data"
        return [f"{fold.name}/{item.name}" for fold in sorted(base.glob("param*"))
                if fold.is_dir() for item in sorted(fold.iterdir()) if item.is_dir()]

    def splits(self, path=DEFAULT_SPLITS, *, create=False):
        path = Path(path)
        if path.exists():
            result = json.loads(path.read_text())
        elif create:
            result = make_splits(self.sample_ids(), self.metadata)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
        else:
            raise FileNotFoundError(f"initialize Car splits first: {path}")
        validate_splits(result, self.metadata)
        return result

    def sample(self, sample_id: str) -> CarSample:
        sample_fold(sample_id)
        raw = ShapeNetCarAdapter._load_sample(self.data_path / "preprocessed_data" / sample_id, sample_id)
        pos, x, y, surf, sid = _unpack(raw, 0, "named")
        x = torch.from_numpy(np.array(x, copy=True))
        y = torch.from_numpy(np.array(y, copy=True))
        return CarSample(torch.from_numpy(np.array(pos, copy=True)), self.input_normalizer.encode(x),
                         y, self.output_normalizer.encode(y),
                         torch.from_numpy(np.array(surf, copy=True)), sid)

    def binding(self):
        return {"operator_checkpoint": str(self.checkpoint),
                "operator_manifest": str(self.manifest_path),
                "operator_sha256": self.metadata["checkpoint_sha256"],
                "normalizer": str(self.normalizer_path),
                "normalizer_sha256": self.metadata["normalizer_sha256"]}
