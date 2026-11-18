"""Variable-point ShapeNet-Car data adapter and normalization contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CarSample:
    pos_raw: torch.Tensor
    x_norm: torch.Tensor
    target_phys: torch.Tensor
    target_norm: torch.Tensor
    surface_mask: torch.Tensor
    sample_id: str | int

    @property
    def pos(self) -> torch.Tensor:
        return self.pos_raw

    @property
    def target(self) -> torch.Tensor:
        """Return the normalized training target on every split."""
        return self.target_norm

    def __len__(self) -> int:
        return int(self.pos_raw.shape[0])


class CarNormalizer:
    """Channel-wise affine normalizer with an explicit population standard deviation."""

    def __init__(self, mean: Any, std: Any, eps: float = 1e-8):
        self.mean = torch.as_tensor(mean).float().clone()
        self.std = torch.as_tensor(std).float().clamp_min(eps).clone()

    def to(self, device: torch.device | str) -> "CarNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


@dataclass(frozen=True)
class CarData:
    train: list[CarSample]
    test: list[CarSample]
    input_normalizer: CarNormalizer
    output_normalizer: CarNormalizer
    normalization_source: str

    @property
    def input_mean(self) -> torch.Tensor:
        return self.input_normalizer.mean

    @property
    def input_std(self) -> torch.Tensor:
        return self.input_normalizer.std

    @property
    def output_mean(self) -> torch.Tensor:
        return self.output_normalizer.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.output_normalizer.std

    @property
    def y_normalizer(self) -> CarNormalizer:
        return self.output_normalizer


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"Car {name} contains NaN or Inf")
    return array


def _unpack(item: Any, index: int, split: str) -> tuple[np.ndarray, ...]:
    if isinstance(item, Mapping):
        def get(*keys: str) -> Any:
            return next((item[key] for key in keys if key in item), None)

        pos = get("pos_raw", "pos")
        x = get("x", "features")
        y = get("y", "target", "target_phys")
        mask = get("surface_mask", "surf", "mask")
        sample_id = get("sample_id", "id")
    else:
        if len(item) not in (4, 5):
            raise ValueError("Car sample must be (pos, x, y, surface_mask[, sample_id])")
        pos, x, y, mask = item[:4]
        sample_id = item[4] if len(item) == 5 else None

    if any(value is None for value in (pos, x, y, mask)):
        raise ValueError("Car sample is missing a required field")
    pos = _array(pos, "pos")
    x = _array(x, "x")
    y = _array(y, "y")
    mask = _array(mask, "surface_mask")
    if (
        pos.ndim != 2
        or pos.shape[1] != 3
        or x.shape != (len(pos), 7)
        or y.shape != (len(pos), 4)
        or mask.shape != (len(pos),)
    ):
        raise ValueError(
            f"invalid Car sample shapes: pos={pos.shape}, x={x.shape}, "
            f"y={y.shape}, surf={mask.shape}"
        )
    sample_id = f"{split}:{index}" if sample_id is None else sample_id
    return (
        pos.astype(np.float32, copy=False),
        x.astype(np.float32, copy=False),
        y.astype(np.float32, copy=False),
        mask.astype(bool, copy=False),
        sample_id,
    )


def _population_stats(arrays: Sequence[np.ndarray], channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    if count == 0:
        raise ValueError("cannot fit Car normalization on an empty split")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-8)
    return torch.from_numpy(mean).float(), torch.from_numpy(std).float()


def _load_normalizers(path: Path) -> tuple[CarNormalizer, CarNormalizer]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        required = {"mean_in", "std_in", "mean_out", "std_out"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(f"Car normalizer is missing keys: {sorted(missing)}")
        input_normalizer = CarNormalizer(values["mean_in"], values["std_in"])
        output_normalizer = CarNormalizer(values["mean_out"], values["std_out"])
    if input_normalizer.mean.shape != (7,) or output_normalizer.mean.shape != (4,):
        raise ValueError("Car normalizer must contain 7 input and 4 output channels")
    return input_normalizer, output_normalizer


class ShapeNetCarAdapter:
    name = "shapenet_car"

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        arrays: tuple[Sequence[Any], Sequence[Any]] | None = None,
        loader: Callable[..., Any] | None = None,
        ntrain: int | None = None,
        ntest: int | None = None,
        config: Mapping[str, Any] | None = None,
        normalizer_path: str | Path | None = None,
    ):
        self.config = dict(config or {})
        self.data_path = Path(data_path).expanduser().resolve() if data_path is not None else None
        self.arrays = arrays
        self.loader = loader or self.config.get("loader")
        self.ntrain = ntrain if ntrain is not None else self.config.get("ntrain")
        self.ntest = ntest if ntest is not None else self.config.get("ntest")
        configured_normalizer = normalizer_path or self.config.get("normalizer_path")
        self.normalizer_path = (
            Path(configured_normalizer).expanduser().resolve()
            if configured_normalizer is not None
            else None
        )
        self.test_fold = int(self.config.get("test_fold", 0))
        self._data: CarData | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ShapeNetCarAdapter":
        return cls(config=config, data_path=config.get("data_path"))

    @staticmethod
    def _load_sample(directory: Path, sample_id: str) -> dict[str, Any]:
        required = ("pos.npy", "x.npy", "y.npy", "surf.npy")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing Car files under {directory}: {missing}")
        return {
            "pos": np.load(directory / "pos.npy", mmap_mode="r"),
            "x": np.load(directory / "x.npy", mmap_mode="r"),
            "y": np.load(directory / "y.npy", mmap_mode="r"),
            "surf": np.load(directory / "surf.npy", mmap_mode="r"),
            "sample_id": sample_id,
        }

    def _read_simple(self, name: str) -> list[dict[str, Any]] | None:
        assert self.data_path is not None
        root = self.data_path / name
        if not root.is_dir():
            return None
        return [
            self._load_sample(directory, f"{name}/{directory.name}")
            for directory in sorted(path for path in root.iterdir() if path.is_dir())
        ]

    def _read(self) -> tuple[list[Any], list[Any]]:
        if self.arrays is not None:
            return list(self.arrays[0]), list(self.arrays[1])
        if self.loader is not None:
            try:
                result = self.loader(self.data_path)
            except TypeError:
                result = self.loader(self.data_path, config=self.config)
            if isinstance(result, Mapping):
                return list(result["train"]), list(result["test"])
            return list(result[0]), list(result[1])
        if self.data_path is None:
            raise ValueError("data_path, arrays, or loader is required")

        simple_train = self._read_simple("train")
        simple_test = self._read_simple("test")
        if simple_train is not None and simple_test is not None:
            return simple_train, simple_test

        preprocessed_root = self.data_path / "preprocessed_data"
        if not preprocessed_root.is_dir():
            raise FileNotFoundError(
                f"expected train/test layout or preprocessed_data under {self.data_path}"
            )
        fold_dirs = sorted(
            path
            for path in preprocessed_root.iterdir()
            if path.is_dir() and path.name.startswith("param")
        )
        if not fold_dirs:
            raise ValueError(f"no param folds found under {preprocessed_root}")

        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        expected_test_fold = f"param{self.test_fold}"
        for fold_dir in fold_dirs:
            destination = test if fold_dir.name == expected_test_fold else train
            destination.extend(
                self._load_sample(directory, f"{fold_dir.name}/{directory.name}")
                for directory in sorted(path for path in fold_dir.iterdir() if path.is_dir())
            )
        if not test:
            raise ValueError(f"Car test fold {expected_test_fold} is empty or absent")
        return train, test

    def load(self) -> CarData:
        if self._data is not None:
            return self._data
        train_raw, test_raw = self._read()
        if self.ntrain is not None:
            train_raw = train_raw[:self.ntrain]
        if self.ntest is not None:
            test_raw = test_raw[:self.ntest]
        train = [_unpack(item, index, "train") for index, item in enumerate(train_raw)]
        test = [_unpack(item, index, "test") for index, item in enumerate(test_raw)]
        if not train:
            raise ValueError("Car training split is empty")
        if not test:
            raise ValueError("Car test split is empty")
        train_ids = {sample[4] for sample in train}
        test_ids = {sample[4] for sample in test}
        if train_ids & test_ids:
            raise ValueError("Car train and test sample IDs overlap")

        if self.normalizer_path is None:
            input_mean, input_std = _population_stats([sample[1] for sample in train], 7)
            output_mean, output_std = _population_stats([sample[2] for sample in train], 4)
            input_normalizer = CarNormalizer(input_mean, input_std)
            output_normalizer = CarNormalizer(output_mean, output_std)
            normalization_source = "fit:train_population"
        else:
            input_normalizer, output_normalizer = _load_normalizers(self.normalizer_path)
            normalization_source = str(self.normalizer_path)

        def build(items: Sequence[tuple[np.ndarray, ...]]) -> list[CarSample]:
            samples = []
            for pos, x, y, mask, sample_id in items:
                pos_tensor = torch.from_numpy(np.array(pos, dtype=np.float32, copy=True))
                x_tensor = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
                target_phys = torch.from_numpy(np.array(y, dtype=np.float32, copy=True))
                samples.append(
                    CarSample(
                        pos_raw=pos_tensor,
                        x_norm=input_normalizer.encode(x_tensor),
                        target_phys=target_phys,
                        target_norm=output_normalizer.encode(target_phys),
                        surface_mask=torch.from_numpy(np.array(mask, dtype=bool, copy=True)),
                        sample_id=sample_id,
                    )
                )
            return samples

        self._data = CarData(
            train=build(train),
            test=build(test),
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
            normalization_source=normalization_source,
        )
        return self._data

    def load_split(self, split: str) -> list[CarSample]:
        data = self.load()
        if split == "train":
            return data.train
        if split == "test":
            return data.test
        raise ValueError(f"unknown split: {split}")

    def full_point_count(self, sample: int | CarSample | None = None) -> int | list[int]:
        samples = self.load().train + self.load().test
        if sample is not None:
            return len(samples[sample]) if isinstance(sample, int) else len(sample)
        counts = [len(item) for item in samples]
        return counts[0] if len(set(counts)) == 1 else counts


CarAdapter = ShapeNetCarAdapter
