"""DrivAerML surface-cloud adapter.

The external dataset is intentionally loaded one car at a time: point counts
vary by design and are too large for a dense stacked BenchmarkSplit.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import torch


@dataclass(frozen=True)
class DriverMLGeometry:
    run_id: int
    pos: torch.Tensor
    normal: torch.Tensor
    area: torch.Tensor
    pos_norm: torch.Tensor
    a_ref: float

    @property
    def point_count(self) -> int:
        return int(self.pos.shape[0])

    def features(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        pos = self.pos_norm if indices is None else self.pos_norm[indices]
        normal = self.normal if indices is None else self.normal[indices]
        return torch.cat((pos, normal), dim=-1)


@dataclass(frozen=True)
class DriverMLSample(DriverMLGeometry):
    pressure: torch.Tensor
    wss: torch.Tensor | None

    def geometry(self) -> DriverMLGeometry:
        return DriverMLGeometry(self.run_id, self.pos, self.normal, self.area,
                                self.pos_norm, self.a_ref)


@dataclass(frozen=True)
class DriverMLSplit:
    adapter: "DriverMLAdapter"
    sample_ids: tuple[int, ...]

    def __len__(self):
        return len(self.sample_ids)

    def __iter__(self) -> Iterator[DriverMLSample]:
        for run_id in self.sample_ids:
            yield self.adapter.load(run_id)


class DriverMLAdapter:
    name = "driverml"

    def __init__(self, data_path: str | Path,
                 stats_path: str | Path | None = None,
                 labels_path: str | Path | None = None,
                 split_path: str | Path | None = None):
        defaults = Path(__file__).resolve().parents[2] / "configs/benchmarks"
        self.data_path = Path(data_path)
        self.stats_path = Path(stats_path) if stats_path else defaults / "driverml_normalizer.json"
        self.stats = json.loads(self.stats_path.read_text())
        self.pos_min = torch.tensor(self.stats["pos_min"], dtype=torch.float32)
        self.pos_size = torch.tensor(self.stats["pos_max"], dtype=torch.float32) - self.pos_min
        if not torch.isfinite(self.pos_size).all() or torch.any(self.pos_size <= 0):
            raise ValueError("DriverML position normalization range must be positive")
        for key in ("pressure_std", "wss_std"):
            std = torch.as_tensor(self.stats[key])
            if not torch.isfinite(std).all() or torch.any(std <= 0):
                raise ValueError(f"invalid DriverML {key}")
        self.labels_path = Path(labels_path) if labels_path else defaults / "driverml_labels.csv"
        self.split_path = Path(split_path) if split_path else defaults / "driverml_splits.json"
        self.splits = json.loads(self.split_path.read_text())["splits"]
        canonical = [set(self.splits[s]) for s in ("train", "val", "test")]
        if any(canonical[i] & canonical[j] for i in range(3) for j in range(i)):
            raise ValueError("DriverML train/val/test IDs overlap")
        if not set(self.splits['screen']).issubset(self.splits['calibration']):
            raise ValueError('DriverML screen must be a subset of calibration')
        calibration, confirmation = set(self.splits['calibration']), set(self.splits['confirmation'])
        if calibration & confirmation or calibration | confirmation != canonical[1]:
            raise ValueError('DriverML calibration/confirmation must partition validation')

    def load_split(self, split: str) -> DriverMLSplit:
        return DriverMLSplit(self, self.run_ids(self.splits, split))

    def cd_labels(self) -> dict[int, float]:
        with self.labels_path.open(newline="") as handle:
            return {int(row["run_id"]): float(row["cd"]) for row in csv.DictReader(handle)}

    def run_ids(self, split_ids: dict[str, list[int]], split: str) -> tuple[int, ...]:
        if split not in split_ids:
            raise ValueError(f"unknown DriverML split: {split}")
        ids = tuple(int(r) for r in split_ids[split])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate DriverML split IDs")
        return ids

    def _directory(self, run_id: int) -> Path:
        return self.data_path / f"run_{run_id}"

    def _has_required_files(self, run_id: int) -> bool:
        d = self._directory(run_id)
        return all((d / name).is_file() for name in (
            "surface_position_vtp.pt", "surface_normal.pt", "surface_area.pt",
            "surface_pressure.pt", "surface_wallshearstress.pt"))

    def load(self, run_id: int, *, include_wss: bool = True) -> DriverMLSample:
        d = self._directory(int(run_id))
        pos = torch.load(d / "surface_position_vtp.pt", weights_only=True).float()
        normal = torch.load(d / "surface_normal.pt", weights_only=True).float()
        area = torch.load(d / "surface_area.pt", weights_only=True).float().reshape(-1)
        pressure = torch.load(d / "surface_pressure.pt", weights_only=True).float().reshape(-1)
        wss = torch.load(d / "surface_wallshearstress.pt", weights_only=True).float() if include_wss else None
        self._validate(pos, normal, area, pressure, wss)
        pos_norm = ((pos - self.pos_min) / self.pos_size).clamp(0.0, 1.0)
        return DriverMLSample(int(run_id), pos, normal, area, pos_norm,
                              self._a_ref(int(run_id)), pressure, wss)

    @staticmethod
    def _validate(pos, normal, area, pressure, wss) -> None:
        n = pos.shape[0]
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"position must be (N,3), got {tuple(pos.shape)}")
        if normal.shape != (n, 3) or (wss is not None and wss.shape != (n, 3)):
            raise ValueError("normal and WSS must align with position as (N,3)")
        if area.shape != (n,) or pressure.shape != (n,):
            raise ValueError("area and pressure must align with position as (N,)")
        for name, value in (("pos", pos), ("normal", normal), ("area", area),
                            ("pressure", pressure), ("wss", wss)):
            if value is not None and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if torch.any(area <= 0):
            raise ValueError("surface areas must be positive")

    def _a_ref(self, run_id: int) -> float:
        candidates = [self._directory(run_id) / f"geo_ref_{run_id}.csv"]
        if self.labels_path:
            candidates.append(self.labels_path.parent / f"geo_ref_{run_id}.csv")
        for path in candidates:
            if path.is_file():
                with path.open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                area = float(row["aRef"])
                if not math.isfinite(area) or area <= 0:
                    raise ValueError("DriverML aRef must be positive and finite")
                return area
        raise FileNotFoundError(f"no geo_ref CSV for run_{run_id}")

    def full_point_count(self, sample: DriverMLSample) -> int:
        return sample.point_count

    def reference_coords(self, sample: DriverMLSample) -> torch.Tensor:
        return sample.pos

    def inputs(self, sample: DriverMLSample, indices: torch.Tensor | None = None) -> torch.Tensor:
        return sample.features(indices)

    def targets(self, sample: DriverMLSample, indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        targets = {"pressure": sample.pressure}
        if sample.wss is not None:
            targets['wss'] = sample.wss
        return targets if indices is None else {name:value[indices] for name,value in targets.items()}
