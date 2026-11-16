"""Lazy fixed-grid SuperWing adapter; geometry/input views contain no labels."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from adasolver.inference.superwing_contract import ROOT, NFULL, GRID
from adasolver.metrics.superwing import surface_geometry, coefficient_weights
from adasolver.utils.interpolation import index_grid, reconstruct_field


@dataclass(frozen=True)
class SuperWingGeometry:
    sample_id: int
    shape_id: int
    pos: np.ndarray
    model_pos: np.ndarray
    condition: np.ndarray
    vertices: np.ndarray
    surface: object
    weights: np.ndarray
    aoa: float
    mach: float
    s_ref: float

    def inputs(self, indices=None):
        pos = self.model_pos if indices is None else self.model_pos[indices]
        return pos, np.broadcast_to(self.condition, (len(pos), 2)).copy()

    def features(self, indices=None):
        pos, condition = self.inputs(indices)
        return np.concatenate([pos, condition], axis=-1)


@dataclass(frozen=True)
class SuperWingSample:
    geometry: SuperWingGeometry
    target: np.ndarray
    structured: np.ndarray

    @property
    def sample_id(self):
        return self.geometry.sample_id


@dataclass(frozen=True)
class SuperWingSplit:
    adapter: 'SuperWingAdapter'
    sample_ids: tuple[int, ...]

    def __len__(self):
        return len(self.sample_ids)

    def __iter__(self):
        for sample_id in self.sample_ids:
            yield self.adapter.load(sample_id)


class SuperWingAdapter:
    name = 'superwing'

    def __init__(self, data_path, stats_path=None, split_path=None, geometry_cache_size=8):
        self.data_path = Path(data_path)
        self.stats_path = Path(stats_path or ROOT/'configs/benchmarks/superwing_normalizer.npz')
        self.split_path = Path(split_path or ROOT/'configs/benchmarks/superwing_splits.json')
        with np.load(self.stats_path) as stats:
            self.pos_mean, self.pos_std = stats['pos_mean'], stats['pos_std']
            self.fx_mean, self.fx_std = stats['fx_mean'], stats['fx_std']
        for mean, std, size in ((self.pos_mean, self.pos_std, 3), (self.fx_mean, self.fx_std, 2)):
            if mean.shape != (size,) or std.shape != (size,) or not np.isfinite([mean, std]).all() or np.any(std <= 0):
                raise ValueError('invalid frozen SuperWing normalization')
        self.metadata = json.loads(self.split_path.read_text())
        self.splits = self.metadata['splits']
        self.arrays = {key: np.load(self.data_path/f'{key}.npy', mmap_mode='r')
                       for key in ('index', 'geom0', 'origingeom', 'data')}
        nsamples, nshapes = len(self.arrays['index']), len(self.arrays['geom0'])
        expected = dict(index=(nsamples, 12), data=(nsamples, 3, *GRID),
                        geom0=(nshapes, 3, *GRID), origingeom=(nshapes, 3, GRID[0]+1, GRID[1]+1))
        if any(self.arrays[k].shape != v for k, v in expected.items()):
            raise ValueError('SuperWing array layout does not match the cell/vertex contract')
        self._validate_splits()
        self.reference = index_grid(GRID)
        self._geometry = OrderedDict()
        self.geometry_cache_size = geometry_cache_size

    def _validate_splits(self):
        index = self.arrays['index']
        canonical = [set(self.splits[k]) for k in ('fit', 'calibration', 'test')]
        shapes = [set(index[sorted(ids), 0].astype(int)) for ids in canonical]
        for i in range(3):
            if len(canonical[i]) != len(self.splits[('fit', 'calibration', 'test')[i]]):
                raise ValueError('duplicate SuperWing sample IDs')
            for j in range(i):
                if canonical[i] & canonical[j] or shapes[i] & shapes[j]:
                    raise ValueError('SuperWing splits must be sample- and shape-disjoint')
        if set.union(*canonical) != set(range(len(index))):
            raise ValueError('canonical splits must partition all samples')
        children = [set(self.splits[k]) for k in ('head_val', 'strategy_calibration', 'confirmation')]
        if set.union(*children) != canonical[1]:
            raise ValueError('development splits must partition calibration')
        child_shapes = [set(index[sorted(ids), 0].astype(int)) for ids in children]
        if any(children[i] & children[j] or child_shapes[i] & child_shapes[j] for i in range(3) for j in range(i)):
            raise ValueError('development splits must be shape-disjoint')

    def load_split(self, split):
        split = 'fit' if split == 'train' else split
        return SuperWingSplit(self, tuple(map(int, self.splits[split])))

    def geometry(self, sample_id):
        sample_id = int(sample_id)
        row = self.arrays['index'][sample_id]
        shape_id = int(row[0])
        if shape_id not in self._geometry:
            pos = np.asarray(self.arrays['geom0'][shape_id]).transpose(1, 2, 0).reshape(NFULL, 3).copy()
            vertices = np.asarray(self.arrays['origingeom'][shape_id]).transpose(1, 2, 0).copy()
            if not np.isfinite(pos).all():
                raise ValueError('non-finite model coordinates')
            normalized = ((pos-self.pos_mean)/self.pos_std).astype(np.float32)
            self._geometry[shape_id] = pos, vertices, normalized, surface_geometry(vertices)
            if len(self._geometry) > self.geometry_cache_size:
                self._geometry.popitem(last=False)
        pos, vertices, normalized, surface = self._geometry[shape_id]
        self._geometry.move_to_end(shape_id)
        condition = (np.asarray(row[[3, 2]], dtype=np.float32)-self.fx_mean)/self.fx_std
        weights = coefficient_weights(surface, float(row[2]), float(row[4]))
        return SuperWingGeometry(sample_id, shape_id, pos, normalized, condition, vertices, surface,
                                 weights, float(row[2]), float(row[3]), float(row[4]))

    def load(self, sample_id):
        geometry = self.geometry(sample_id)
        target = np.asarray(self.arrays['data'][sample_id], dtype=np.float32).transpose(1, 2, 0).reshape(NFULL, 3).copy()
        structured = np.asarray(self.arrays['index'][sample_id, 9:12]).copy()
        if not np.isfinite(target).all() or not np.isfinite(structured).all():
            raise ValueError('non-finite SuperWing targets')
        return SuperWingSample(geometry, target, structured)

    def full_point_count(self, sample=None):
        return NFULL

    def reference_coords(self, sample=None):
        return self.reference

    def inputs(self, sample):
        return sample.geometry.inputs()

    def targets(self, sample):
        return sample.target

    def decode_targets(self, prediction):
        """Operator already predicts dataset-scaled channels; there is no y z-score."""
        return prediction

    def reconstruct(self, indices, prediction):
        return reconstruct_field(indices, prediction, self.reference)
