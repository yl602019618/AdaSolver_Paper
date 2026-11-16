from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import BenchmarkAdapter, VariableNBenchmarkAdapter
from .darcy import DarcyAdapter
from .elasticity import ElasticityAdapter
from .pipe import PipeAdapter
from .shapenet_car import ShapeNetCarAdapter
from .driverml import DriverMLAdapter
from .superwing import SuperWingAdapter


AdapterFactory = Callable[[str | Path], BenchmarkAdapter | VariableNBenchmarkAdapter]

_ADAPTERS: dict[str, AdapterFactory] = {
    "darcy": DarcyAdapter,
    "elasticity": ElasticityAdapter,
    "pipe": PipeAdapter,
    "shapenet_car": ShapeNetCarAdapter,
    "driverml": DriverMLAdapter,
    "superwing": SuperWingAdapter,
}


def create_adapter(name: str, data_path: str | Path) -> BenchmarkAdapter | VariableNBenchmarkAdapter:
    try:
        factory = _ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark: {name}") from error
    return factory(data_path)


def registered_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
