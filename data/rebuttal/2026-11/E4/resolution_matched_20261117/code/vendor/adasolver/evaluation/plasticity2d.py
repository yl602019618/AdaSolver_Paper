from __future__ import annotations

import numpy as np

from adasolver.data.plasticity2d import PlasticityLayout, unflatten_temporal_field
from adasolver.metrics.plasticity2d import TemporalFieldMetrics, temporal_relative_l2
from adasolver.utils.interpolation import reconstruct_field


def reconstruct_temporal_field(indices: np.ndarray, prediction: np.ndarray,
                               reference_coords: np.ndarray,
                               layout: PlasticityLayout = PlasticityLayout(),
                               target: np.ndarray | None = None,
                               ) -> tuple[np.ndarray, int, TemporalFieldMetrics | None]:
    flat, fallback_count = reconstruct_field(indices, prediction, reference_coords)
    temporal = unflatten_temporal_field(flat, layout)
    metrics = None if target is None else temporal_relative_l2(temporal, target)
    return temporal, fallback_count, metrics
