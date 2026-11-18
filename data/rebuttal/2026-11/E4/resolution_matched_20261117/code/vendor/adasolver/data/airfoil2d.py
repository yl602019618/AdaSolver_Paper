from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import BenchmarkSplit
from .pipe import UnitTransformer


@dataclass(frozen=True)
class Airfoil2DData:
    train: BenchmarkSplit
    test: BenchmarkSplit
    y_normalizer: UnitTransformer
    shape: tuple[int, int]
    mandatory_indices: np.ndarray
    target_name: str
    target_channel: int


class Airfoil2DAdapter:
    """Load the structured NACA C-grid pressure benchmark.

    Coordinates are sample-dependent, while the flattened ``(i, j)`` ordering
    is shared by every sample.  The operator target is pressure channel 3.
    """

    name = "airfoil2d"
    required_files = ("NACA_Cylinder_X.npy", "NACA_Cylinder_Y.npy", "NACA_Cylinder_Q.npy")

    def __init__(self, data_path: str | Path, ntrain: int = 1000, ntest: int = 200,
                 shape: tuple[int, int] = (221, 51), target_channel: int = 3):
        self.data_path = Path(data_path).expanduser().resolve()
        self.ntrain = ntrain
        self.ntest = ntest
        self.shape = shape
        self.target_channel = target_channel
        self._data: Airfoil2DData | None = None

    def full_point_count(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def mandatory_indices(self) -> np.ndarray:
        i = np.arange(50, 171, dtype=np.int64)
        return i * self.shape[1]

    def _validate_files(self) -> None:
        missing = [name for name in self.required_files if not (self.data_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing Airfoil files under {self.data_path}: {missing}")

    def load(self) -> Airfoil2DData:
        if self._data is not None:
            return self._data
        self._validate_files()
        x = np.load(self.data_path / "NACA_Cylinder_X.npy", mmap_mode="r")
        y = np.load(self.data_path / "NACA_Cylinder_Y.npy", mmap_mode="r")
        q = np.load(self.data_path / "NACA_Cylinder_Q.npy", mmap_mode="r")
        required = self.ntrain + self.ntest
        if x.shape != y.shape or x.ndim != 3 or x.shape[1:] != self.shape:
            raise ValueError(f"X/Y must both have shape (samples, {self.shape[0]}, {self.shape[1]}): {x.shape}, {y.shape}")
        if q.ndim != 4 or q.shape[0] < required or q.shape[1] <= self.target_channel or q.shape[2:] != self.shape:
            raise ValueError(f"Q must have shape (samples, channels, {self.shape[0]}, {self.shape[1]}): {q.shape}")
        if x.shape[0] < required:
            raise ValueError(f"need {required} samples, got {x.shape[0]}")
        stop = required
        coords_np = np.stack((np.asarray(x[:stop]), np.asarray(y[:stop])), axis=-1).reshape(stop, -1, 2)
        target_np = np.array(q[:stop, self.target_channel], copy=True).reshape(stop, -1, 1)
        coords = torch.from_numpy(coords_np).float()
        target = torch.from_numpy(target_np).float()
        if not torch.isfinite(coords).all() or not torch.isfinite(target).all():
            raise ValueError("Airfoil data contains NaN or Inf")
        normalizer = UnitTransformer(target[:self.ntrain])
        train_target = normalizer.encode(target[:self.ntrain])
        train_pos = coords[:self.ntrain]
        test_pos = coords[self.ntrain:stop]
        train = BenchmarkSplit(train_pos, train_pos, train_target, tuple(range(self.ntrain)))
        test = BenchmarkSplit(test_pos, test_pos, target[self.ntrain:stop], tuple(range(self.ntrain, stop)))
        self._data = Airfoil2DData(train, test, normalizer, self.shape,
                                   self.mandatory_indices.copy(), "pressure", self.target_channel)
        return self._data

    def load_split(self, split: str) -> BenchmarkSplit:
        data = self.load()
        if split == "train":
            return data.train
        if split == "test":
            return data.test
        raise ValueError(f"unknown split: {split}")
