"""Chunked nearest-cell reconstruction and quadrature for irregular surfaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class SurfaceMap:
    reference_to_selected: np.ndarray
    cell_counts: np.ndarray

    def reconstruct(self, values):
        if len(values) != len(self.cell_counts):
            raise ValueError("reconstruction values must match the selected point order")
        if isinstance(values, torch.Tensor):
            return values[torch.from_numpy(self.reference_to_selected).to(values.device)]
        return np.asarray(values)[self.reference_to_selected]

    def compensation(self, beta):
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("compensation beta must lie in [0,1]")
        return np.power(self.cell_counts / self.cell_counts.mean(), beta).astype(np.float32)


def nearest_surface_map(reference, selected_indices, query_chunk=131072, workers=8):
    coords = np.asarray(reference, dtype=np.float64)
    selected = np.asarray(selected_indices, dtype=np.int64)
    n = len(coords)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("surface coordinates must be finite (N,3)")
    if len(selected) == 0 or len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= n):
        raise ValueError("selected surface indices must be nonempty, unique, and in bounds")
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    mapping = np.empty(n, dtype=np.int64)
    if len(selected) < n:
        tree = cKDTree(coords[selected])
        for start in range(0, n, query_chunk):
            mapping[start:start + query_chunk] = tree.query(coords[start:start + query_chunk],
                                                           k=1, workers=workers)[1]
    # Preserve the exact selected rows even when different indices coincide.
    mapping[selected] = np.arange(len(selected))
    counts = np.bincount(mapping, minlength=len(selected)).astype(np.float64)
    return SurfaceMap(mapping, counts)
