"""Memory-mapped Airfoil samples and the frozen operator's normalizer."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from adasolver.models.checkpoint import sha256_file


class AirfoilRuntimeData:
    def __init__(self, data_path, operator_manifest):
        root = Path(data_path)
        self.x, self.y, self.q = [np.load(root / name, mmap_mode="r") for name in
            ("NACA_Cylinder_X.npy", "NACA_Cylinder_Y.npy", "NACA_Cylinder_Q.npy")]
        if self.x.shape != self.y.shape or self.x.shape[1:] != (221, 51) or self.q.shape != (len(self.x), 5, 221, 51):
            raise ValueError("unexpected Airfoil pressure data shapes")
        path = Path(operator_manifest)
        metadata = json.loads(path.read_text())
        normalizer_path = Path(metadata.get("normalizer", str(path).replace("_manifest.json", "_ynorm.npz")))
        with np.load(normalizer_path) as normalizer:
            self.mean = float(normalizer["mean"].item())
            self.std = float(normalizer["std"].item())
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("invalid frozen pressure normalizer")
        self.normalizer_path = str(normalizer_path.resolve())
        self.normalizer_sha256 = sha256_file(normalizer_path)
        self.mandatory = np.arange(50, 171, dtype=np.int64) * 51

    def coords(self, sample_id):
        return np.stack((self.x[sample_id], self.y[sample_id]), axis=-1).reshape(-1, 2).astype(np.float32)

    def target(self, sample_id):
        return np.asarray(self.q[sample_id, 3], dtype=np.float32).reshape(-1, 1)

    def decode(self, prediction):
        # Match UnitTransformer's float32 arithmetic, rather than changing the
        # historical model output by promoting it to float64 here.
        if isinstance(prediction, torch.Tensor):
            return prediction * prediction.new_tensor(self.std) + prediction.new_tensor(self.mean)
        return prediction * np.float32(self.std) + np.float32(self.mean)
