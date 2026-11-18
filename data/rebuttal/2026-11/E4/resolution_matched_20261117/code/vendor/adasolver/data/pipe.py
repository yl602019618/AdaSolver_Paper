from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import BenchmarkSplit


class UnitTransformer:
    """Global channel-wise standardization fitted on training points only."""

    def __init__(self, values: torch.Tensor, eps: float = 1e-8):
        if values.ndim != 3:
            raise ValueError(f"expected (samples, points, channels), got {values.shape}")
        self.mean = values.mean(dim=(0, 1), keepdim=True)
        self.std = values.std(dim=(0, 1), keepdim=True) + eps

    def to(self, device: torch.device | str) -> "UnitTransformer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


@dataclass(frozen=True)
class PipeData:
    train: BenchmarkSplit
    test: BenchmarkSplit
    y_normalizer: UnitTransformer
    shape: tuple[int, int]


class PipeAdapter:
    name = "pipe"
    required_files = ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy")

    def __init__(self, data_path: str | Path, ntrain: int = 1000, ntest: int = 200,
                 shape: tuple[int, int] = (129, 129)):
        self.data_path = Path(data_path).expanduser().resolve()
        self.ntrain = ntrain
        self.ntest = ntest
        self.shape = shape
        self._data: PipeData | None = None

    def full_point_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def _validate_files(self) -> None:
        missing = [name for name in self.required_files if not (self.data_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing Pipe files under {self.data_path}: {missing}")

    def load(self) -> PipeData:
        if self._data is not None:
            return self._data
        self._validate_files()
        x = np.load(self.data_path / "Pipe_X.npy", mmap_mode="r")
        y = np.load(self.data_path / "Pipe_Y.npy", mmap_mode="r")
        q = np.load(self.data_path / "Pipe_Q.npy", mmap_mode="r")
        required_samples = self.ntrain + self.ntest
        if x.shape != y.shape:
            raise ValueError(f"Pipe_X and Pipe_Y shape mismatch: {x.shape} vs {y.shape}")
        if x.shape[0] < required_samples or q.shape[0] < required_samples:
            raise ValueError(f"need {required_samples} samples, got X={x.shape[0]} Q={q.shape[0]}")
        if x.shape[1:] != self.shape:
            raise ValueError(f"expected coordinate shape {self.shape}, got {x.shape[1:]}")
        if q.ndim != 4 or q.shape[1] < 1 or q.shape[2:] != self.shape:
            raise ValueError(
                f"Pipe_Q must be channel-first (samples, channels, {self.shape[0]}, {self.shape[1]}), "
                f"got {q.shape}"
            )

        stop = required_samples
        coords = torch.from_numpy(
            np.stack((np.asarray(x[:stop]), np.asarray(y[:stop])), axis=-1).reshape(stop, -1, 2)
        ).float()
        target = torch.from_numpy(np.array(q[:stop, 0], copy=True).reshape(stop, -1, 1)).float()
        if not torch.isfinite(coords).all() or not torch.isfinite(target).all():
            raise ValueError("Pipe data contains NaN or Inf")

        train_target_raw = target[:self.ntrain]
        normalizer = UnitTransformer(train_target_raw)
        train = BenchmarkSplit(coords[:self.ntrain], coords[:self.ntrain],
                               normalizer.encode(train_target_raw), tuple(range(self.ntrain)))
        test = BenchmarkSplit(coords[self.ntrain:stop], coords[self.ntrain:stop],
                              target[self.ntrain:stop], tuple(range(self.ntrain, stop)))
        if set(train.sample_ids) & set(test.sample_ids):
            raise RuntimeError("Pipe train and test splits overlap")
        self._data = PipeData(train, test, normalizer, self.shape)
        return self._data

    def load_split(self, split: str) -> BenchmarkSplit:
        data = self.load()
        if split == "train":
            return data.train
        if split == "test":
            return data.test
        raise ValueError(f"unknown split: {split}")
