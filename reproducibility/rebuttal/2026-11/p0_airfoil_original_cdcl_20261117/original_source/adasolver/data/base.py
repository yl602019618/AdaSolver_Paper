from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from collections.abc import Iterable

import torch


@dataclass(frozen=True)
class BenchmarkSplit:
    pos: torch.Tensor
    fx: torch.Tensor | None
    target: torch.Tensor
    sample_ids: tuple[int, ...]

    def reference_coords(self, sample_index: int) -> torch.Tensor:
        """Return the full physical reference coordinates for one sample."""
        return self.pos[sample_index]


class BenchmarkAdapter(Protocol):
    name: str

    def load_split(self, split: str) -> BenchmarkSplit: ...

    def full_point_count(self) -> int: ...


class VariableNBenchmarkAdapter(Protocol):
    """Lazy per-sample surfaces/clouds that cannot be stacked into BenchmarkSplit."""
    name: str

    def load_split(self, split: str) -> Iterable[Any]: ...

    def full_point_count(self, sample: Any) -> int: ...

    def reference_coords(self, sample: Any) -> torch.Tensor: ...
