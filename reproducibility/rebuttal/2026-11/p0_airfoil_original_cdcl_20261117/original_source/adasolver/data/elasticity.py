from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import BenchmarkSplit
from .pipe import UnitTransformer


@dataclass(frozen=True)
class ElasticityData:
    train: BenchmarkSplit
    test: BenchmarkSplit
    y_normalizer: UnitTransformer


class ElasticityAdapter:
    """Load the sample-dependent Elasticity point-cloud stress benchmark."""

    name = "elasticity"
    point_count = 972
    coordinate_file = "Random_UnitCell_XY_10.npy"
    target_file = "Random_UnitCell_sigma_10.npy"

    def __init__(self, data_path: str | Path, ntrain: int = 1000, ntest: int = 200,
                 arrays: tuple[np.ndarray, np.ndarray] | None = None):
        self.data_path = Path(data_path).expanduser().resolve()
        self.ntrain = ntrain
        self.ntest = ntest
        self._arrays = arrays
        self._data: ElasticityData | None = None

    def full_point_count(self) -> int:
        return self.point_count

    def _paths(self) -> tuple[Path, Path]:
        candidates = (self.data_path, self.data_path / "elasticity" / "Meshes")
        for root in candidates:
            xy_path = root / self.coordinate_file
            target_path = root / self.target_file
            if xy_path.is_file() and target_path.is_file():
                return xy_path, target_path
        raise FileNotFoundError(
            f"missing Elasticity files below {self.data_path}: "
            f"{self.coordinate_file}, {self.target_file}"
        )

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is not None:
            return self._arrays
        xy_path, target_path = self._paths()
        return np.load(xy_path, mmap_mode="r"), np.load(target_path, mmap_mode="r")

    def load(self) -> ElasticityData:
        if self._data is not None:
            return self._data
        xy, stress = self._load_arrays()
        if xy.ndim != 3 or xy.shape[:2] != (self.point_count, 2):
            raise ValueError(f"Elasticity XY must have shape (972, 2, samples), got {xy.shape}")
        if stress.ndim != 2 or stress.shape[0] != self.point_count:
            raise ValueError(f"Elasticity stress must have shape (972, samples), got {stress.shape}")
        if xy.shape[2] != stress.shape[1]:
            raise ValueError(f"Elasticity sample count mismatch: XY={xy.shape[2]} stress={stress.shape[1]}")
        nsamples = xy.shape[2]
        if nsamples < self.ntrain + self.ntest:
            raise ValueError(f"need at least {self.ntrain + self.ntest} samples, got {nsamples}")

        coords = torch.from_numpy(np.array(xy.transpose(2, 0, 1), copy=True)).float()
        target = torch.from_numpy(np.array(stress.T[:, :, None], copy=True)).float()
        if not torch.isfinite(coords).all() or not torch.isfinite(target).all():
            raise ValueError("Elasticity data contains NaN or Inf")
        if nsamples > 1 and torch.equal(coords[0], coords[-1]):
            raise ValueError("Elasticity coordinates unexpectedly share one geometry across all samples")

        test_start = nsamples - self.ntest
        if test_start < self.ntrain:
            raise ValueError("Elasticity train and test splits overlap")
        normalizer = UnitTransformer(target[:self.ntrain])
        train = BenchmarkSplit(
            coords[:self.ntrain], None, normalizer.encode(target[:self.ntrain]),
            tuple(range(self.ntrain)),
        )
        test = BenchmarkSplit(
            coords[test_start:], None, target[test_start:],
            tuple(range(test_start, nsamples)),
        )
        self._data = ElasticityData(train, test, normalizer)
        return self._data

    def load_split(self, split: str) -> BenchmarkSplit:
        data = self.load()
        if split == "train":
            return data.train
        if split == "test":
            return data.test
        raise ValueError(f"unknown split: {split}")

    def load_ids(self, sample_ids, *, encoded_targets=False) -> BenchmarkSplit:
        """Explicit original IDs, independent of the historical tail-test loader."""
        ids = tuple(int(i) for i in sample_ids)
        xy, stress = self._load_arrays()
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= xy.shape[2]:
            raise ValueError("sample IDs must be nonempty, unique and within the dataset")
        pos = torch.from_numpy(np.array(xy[:, :, ids].transpose(2, 0, 1), copy=True)).float()
        target = torch.from_numpy(np.array(stress[:, ids].T[:, :, None], copy=True)).float()
        if encoded_targets:
            target = self.load().y_normalizer.encode(target)
        return BenchmarkSplit(pos, None, target, ids)
