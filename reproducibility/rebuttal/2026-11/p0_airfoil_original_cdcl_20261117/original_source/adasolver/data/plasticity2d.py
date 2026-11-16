from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.io import loadmat

from .base import BenchmarkSplit
from .pipe import UnitTransformer


PLASTICITY_DATA_SHA256 = "b58681e7777531b34889da46b506d2f18845c5d7b194859394dfd958cf28178f"


@dataclass(frozen=True)
class PlasticityLayout:
    source_frames: int = 20
    frame_start: int = 1
    predicted_frames: int = 19
    displacement_channels: int = 2
    order: str = "frame_major_channel_minor"

    def __post_init__(self):
        if (self.frame_start < 0 or self.predicted_frames < 1
                or self.frame_start + self.predicted_frames > self.source_frames
                or self.displacement_channels != 2
                or self.order != "frame_major_channel_minor"):
            raise ValueError("invalid Plasticity temporal layout")

    @property
    def out_dim(self) -> int:
        return self.predicted_frames * self.displacement_channels


@dataclass(frozen=True)
class Plasticity2DData:
    splits: Mapping[str, BenchmarkSplit]
    input_normalizer: UnitTransformer
    shape: tuple[int, int]
    layout: PlasticityLayout
    time_values: torch.Tensor
    reference_xy: torch.Tensor
    frame0_displacement: torch.Tensor
    contact_indices: np.ndarray

    @property
    def train(self) -> BenchmarkSplit:
        return self.splits["train"]

    @property
    def test(self) -> BenchmarkSplit:
        return self.splits["test"]


def flatten_temporal_field(values: np.ndarray | torch.Tensor,
                           layout: PlasticityLayout) -> np.ndarray | torch.Tensor:
    expected = (layout.predicted_frames, layout.displacement_channels)
    if tuple(values.shape[-2:]) != expected:
        raise ValueError(f"expected trailing temporal shape {expected}, got {tuple(values.shape[-2:])}")
    return values.reshape(*values.shape[:-2], layout.out_dim)


def unflatten_temporal_field(values: np.ndarray | torch.Tensor,
                             layout: PlasticityLayout) -> np.ndarray | torch.Tensor:
    if values.shape[-1] != layout.out_dim:
        raise ValueError(f"expected final dimension {layout.out_dim}, got {values.shape[-1]}")
    return values.reshape(*values.shape[:-1], layout.predicted_frames,
                          layout.displacement_channels)


class Plasticity2DAdapter:
    """Load the single-forward, all-timestep Plasticity displacement task."""

    name = "plasticity2d"
    shape = (101, 31)
    layout = PlasticityLayout()
    legacy_split_ranges = {"train": (0, 900), "unused": (900, 907), "test": (907, 987)}

    def __init__(self, data_path: str | Path,
                 split_ranges: Mapping[str, tuple[int, int]] | None = None,
                 arrays: tuple[np.ndarray, np.ndarray] | None = None, *,
                 layout: PlasticityLayout | None = None,
                 input_normalizer_path: str | Path | None = None):
        self.data_path = Path(data_path).expanduser().resolve()
        self.split_ranges = dict(split_ranges or self.legacy_split_ranges)
        self._arrays = arrays
        self.layout = layout or PlasticityLayout()
        self.input_normalizer_path = input_normalizer_path
        self._data: Plasticity2DData | None = None
        self._validate_split_ranges()

    def full_point_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def _validate_split_ranges(self) -> None:
        if "train" not in self.split_ranges or "test" not in self.split_ranges:
            raise ValueError("split_ranges must define train and test")
        occupied: set[int] = set()
        for name, bounds in self.split_ranges.items():
            if len(bounds) != 2 or bounds[0] < 0 or bounds[1] <= bounds[0]:
                raise ValueError(f"invalid split range for {name}: {bounds}")
            indices = set(range(*bounds))
            if occupied & indices:
                raise ValueError(f"split {name} overlaps another split")
            occupied.update(indices)

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is not None:
            return self._arrays
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Plasticity MAT file not found: {self.data_path}")
        values = loadmat(self.data_path, variable_names=("input", "output"))
        if "input" not in values or "output" not in values:
            raise ValueError("Plasticity MAT must contain input and output")
        return values["input"], values["output"]

    def load(self) -> Plasticity2DData:
        if self._data is not None:
            return self._data
        profile, output = self._load_arrays()
        required_samples = max(stop for _, stop in self.split_ranges.values())
        expected_output = (self.shape[0], self.shape[1], self.layout.source_frames, 4)
        if profile.ndim != 2 or profile.shape[1] != self.shape[0]:
            raise ValueError(f"input must have shape (samples, {self.shape[0]}), got {profile.shape}")
        if output.ndim != 5 or tuple(output.shape[1:]) != expected_output:
            raise ValueError(f"output must have shape (samples, {expected_output}), got {output.shape}")
        if profile.shape[0] != output.shape[0] or profile.shape[0] < required_samples:
            raise ValueError(f"need {required_samples} aligned samples, got input={profile.shape[0]} output={output.shape[0]}")
        if not np.isfinite(profile).all() or not np.isfinite(output).all():
            raise ValueError("Plasticity data contains NaN or Inf")

        sample_count = profile.shape[0]
        point_count = self.full_point_count()
        i, j = np.meshgrid(np.linspace(0.0, 1.0, self.shape[0]),
                           np.linspace(0.0, 1.0, self.shape[1]), indexing="ij")
        pos_one = np.stack((i.reshape(-1), j.reshape(-1)), axis=-1)
        pos = torch.from_numpy(np.broadcast_to(pos_one, (sample_count, point_count, 2)).copy()).float()
        fx_np = np.broadcast_to(profile[:, :, None, None],
                                (sample_count, *self.shape, 1)).reshape(sample_count, point_count, 1).copy()
        fx = torch.from_numpy(fx_np).float()
        train_start, train_stop = self.split_ranges["train"]
        if self.input_normalizer_path is None:
            input_normalizer = UnitTransformer(fx[train_start:train_stop])
        else:
            with np.load(self.input_normalizer_path) as stats:
                mean = np.array(stats["mean"], copy=True)
                std = np.array(stats["std"], copy=True)
            if (mean.shape != (1, 1, 1) or std.shape != (1, 1, 1)
                    or not np.isfinite(mean).all() or not np.isfinite(std).all()
                    or np.any(std <= 0)):
                raise ValueError("invalid Plasticity input normalizer")
            input_normalizer = UnitTransformer.__new__(UnitTransformer)
            input_normalizer.mean = torch.from_numpy(mean).float()
            input_normalizer.std = torch.from_numpy(std).float()
        fx = input_normalizer.encode(fx)

        temporal = output[..., self.layout.frame_start:self.layout.frame_start + self.layout.predicted_frames, 2:4]
        temporal = temporal.reshape(sample_count, point_count,
                                    self.layout.predicted_frames, self.layout.displacement_channels)
        target = torch.from_numpy(np.array(flatten_temporal_field(temporal, self.layout), copy=True)).float()
        reference_xy = torch.from_numpy(np.array(output[..., 0, :2], copy=True).reshape(sample_count, point_count, 2)).float()
        frame0 = torch.from_numpy(np.array(output[..., 0, 2:4], copy=True).reshape(sample_count, point_count, 2)).float()

        splits = {}
        for name, (start, stop) in self.split_ranges.items():
            splits[name] = BenchmarkSplit(pos[start:stop], fx[start:stop], target[start:stop],
                                          tuple(range(start, stop)))
        contact = np.arange(self.shape[0], dtype=np.int64) * self.shape[1]
        time_values = torch.linspace(0.0, 1.0, self.layout.source_frames)[
            self.layout.frame_start:self.layout.frame_start + self.layout.predicted_frames]
        self._data = Plasticity2DData(splits, input_normalizer, self.shape, self.layout,
                                      time_values, reference_xy, frame0, contact)
        return self._data

    def load_split(self, split: str) -> BenchmarkSplit:
        data = self.load()
        try:
            return data.splits[split]
        except KeyError as error:
            raise ValueError(f"unknown split: {split}") from error
