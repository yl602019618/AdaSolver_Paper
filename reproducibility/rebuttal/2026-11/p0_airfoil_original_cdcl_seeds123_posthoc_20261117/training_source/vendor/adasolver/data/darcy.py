from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch

from .base import BenchmarkSplit
from .pipe import UnitTransformer


@dataclass(frozen=True)
class DarcyData:
    train: BenchmarkSplit
    test: BenchmarkSplit
    x_normalizer: UnitTransformer
    y_normalizer: UnitTransformer
    shape: tuple[int, int]


class DarcyAdapter:
    """Load the piecewise-constant Darcy coefficient-to-pressure benchmark."""

    name = "darcy"
    train_file = "piececonst_r421_N1024_smooth1.mat"
    test_file = "piececonst_r421_N1024_smooth2.mat"
    required_fields = ("coeff", "sol")

    def __init__(
        self,
        data_path: str | Path,
        ntrain: int = 1000,
        ntest: int = 200,
        downsample: int = 5,
        raw_shape: tuple[int, int] = (421, 421),
        arrays: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.ntrain = ntrain
        self.ntest = ntest
        self.downsample = downsample
        self.raw_shape = raw_shape
        self.shape = tuple((size - 1) // downsample + 1 for size in raw_shape)
        self._arrays = arrays
        self._data: DarcyData | None = None

        if downsample <= 0:
            raise ValueError(f"downsample must be positive, got {downsample}")
        if any((size - 1) % downsample for size in raw_shape):
            raise ValueError(f"downsample {downsample} does not preserve endpoints of {raw_shape}")

    def full_point_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def _paths(self) -> tuple[Path, Path]:
        train_path = self.data_path / self.train_file
        test_path = self.data_path / self.test_file
        missing = [path.name for path in (train_path, test_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing Darcy files under {self.data_path}: {missing}")
        return train_path, test_path

    @staticmethod
    def _load_mat(path: Path) -> dict[str, np.ndarray]:
        return scio.loadmat(path, variable_names=list(DarcyAdapter.required_fields))

    def _load_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self._arrays is not None:
            return self._arrays
        train_path, test_path = self._paths()
        return self._load_mat(train_path), self._load_mat(test_path)

    def _extract_source(
        self, source: dict[str, np.ndarray], count: int, source_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [field for field in self.required_fields if field not in source]
        if missing:
            raise KeyError(f"Darcy {source_name} source is missing fields: {missing}")
        coeff = source["coeff"]
        target = source["sol"]
        if coeff.ndim != 3 or coeff.shape[1:] != self.raw_shape:
            raise ValueError(
                f"Darcy {source_name} coeff must have shape (samples, {self.raw_shape[0]}, "
                f"{self.raw_shape[1]}), got {coeff.shape}"
            )
        if target.shape != coeff.shape:
            raise ValueError(
                f"Darcy {source_name} sol must match coeff shape: {target.shape} vs {coeff.shape}"
            )
        if coeff.shape[0] < count:
            raise ValueError(f"need {count} Darcy {source_name} samples, got {coeff.shape[0]}")

        sampled_coeff = np.array(
            coeff[:count, :: self.downsample, :: self.downsample], copy=True
        )[:, : self.shape[0], : self.shape[1]]
        sampled_target = np.array(
            target[:count, :: self.downsample, :: self.downsample], copy=True
        )[:, : self.shape[0], : self.shape[1]]
        if sampled_coeff.shape[1:] != self.shape or sampled_target.shape[1:] != self.shape:
            raise ValueError(
                f"Darcy downsample expected {self.shape}, got "
                f"coeff={sampled_coeff.shape[1:]} sol={sampled_target.shape[1:]}"
            )

        coeff_tensor = torch.from_numpy(sampled_coeff.reshape(count, -1, 1)).float()
        target_tensor = torch.from_numpy(sampled_target.reshape(count, -1, 1)).float()
        if not torch.isfinite(coeff_tensor).all() or not torch.isfinite(target_tensor).all():
            raise ValueError(f"Darcy {source_name} data contains NaN or Inf")
        return coeff_tensor, target_tensor

    def reference_coords(self) -> torch.Tensor:
        x = np.linspace(0.0, 1.0, self.shape[0])
        y = np.linspace(0.0, 1.0, self.shape[1])
        grid_x, grid_y = np.meshgrid(x, y)
        return torch.from_numpy(np.column_stack((grid_x.ravel(), grid_y.ravel()))).float()

    def load(self) -> DarcyData:
        if self._data is not None:
            return self._data
        train_source, test_source = self._load_arrays()
        train_coeff, train_target_raw = self._extract_source(
            train_source, self.ntrain, "train"
        )
        test_coeff, test_target = self._extract_source(test_source, self.ntest, "test")

        x_normalizer = UnitTransformer(train_coeff)
        y_normalizer = UnitTransformer(train_target_raw)
        coords = self.reference_coords()
        train_pos = coords.unsqueeze(0).repeat(self.ntrain, 1, 1)
        test_pos = coords.unsqueeze(0).repeat(self.ntest, 1, 1)
        train = BenchmarkSplit(
            train_pos,
            x_normalizer.encode(train_coeff),
            y_normalizer.encode(train_target_raw),
            tuple(range(self.ntrain)),
        )
        test = BenchmarkSplit(
            test_pos,
            x_normalizer.encode(test_coeff),
            test_target,
            tuple(range(self.ntrain, self.ntrain + self.ntest)),
        )
        if set(train.sample_ids) & set(test.sample_ids):
            raise RuntimeError("Darcy train and test splits overlap")
        self._data = DarcyData(train, test, x_normalizer, y_normalizer, self.shape)
        return self._data

    def load_split(self, split: str) -> BenchmarkSplit:
        data = self.load()
        if split == "train":
            return data.train
        if split == "test":
            return data.test
        raise ValueError(f"unknown split: {split}")
