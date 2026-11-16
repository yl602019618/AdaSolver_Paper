"""Shared DriverML score processing and final-set sampling."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from adasolver.inference.base import SamplingResult
from adasolver.inference.driverml_contract import validate_coarse_indices


@dataclass(frozen=True)
class DriverMLSelection:
    indices: np.ndarray
    metadata: dict

    def finalize(self, surface_map, *, beta, uniform=False):
        count = len(self.indices)
        nfull = len(surface_map.reference_to_selected)
        compensation = np.ones(count, dtype=np.float32) if uniform else surface_map.compensation(beta)
        density = np.full(count, 1.0/nfull) if uniform else 1.0/(count*surface_map.cell_counts)
        result = SamplingResult(self.indices, density, compensation,
                                {**self.metadata, 'density_backend': 'uniform' if uniform else 'nearest_cell_count',
                                 'beta': 0.0 if uniform else beta})
        result.validate(nfull, count)
        return result


@dataclass(frozen=True)
class CoarsePrediction:
    indices: np.ndarray
    pressure: np.ndarray
    nfull: int
    wss: np.ndarray | None = None

    def __post_init__(self):
        validate_coarse_indices(self.indices, self.nfull)
        if np.asarray(self.pressure).shape != (len(self.indices),) or not np.isfinite(self.pressure).all():
            raise ValueError('coarse pressure must align with its original indices')
        if self.wss is not None and (np.asarray(self.wss).shape != (len(self.indices), 3) or not np.isfinite(self.wss).all()):
            raise ValueError('coarse WSS must align with its original indices')


def normalize_score(raw, quantile=0.99):
    score = np.asarray(raw, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError('acquisition scores must be finite and nonnegative')
    lo = score.min()
    hi = np.quantile(score, quantile)
    return np.clip((score-lo)/(hi-lo+1e-12), 0, 1)


def select_from_score(score, count, rng, *, amp, coarse=None, metadata=None,score_power=1.0):
    score = normalize_score(score)
    if not np.isfinite(score_power) or score_power<=0:raise ValueError('score power must be finite and positive')
    if score_power!=1.0:score=score**score_power
    n = len(score)
    if not 1 <= count <= n or not np.isfinite(amp) or amp < 0:
        raise ValueError('invalid final point count or density amplitude')
    coarse = np.asarray(coarse if coarse is not None else [], dtype=np.int64)
    if len(coarse):
        validate_coarse_indices(coarse, n)
    if count < len(coarse):
        raise ValueError('final count cannot be smaller than the fixed coarse set')
    # Exponential races sample weighted points without replacement in O(N).
    priority = rng.exponential(size=n)/(1+amp*score)
    priority[coarse] = np.inf
    extra_count = count-len(coarse)
    extra = np.argpartition(priority, extra_count-1)[:extra_count] if extra_count else np.empty(0, dtype=np.int64)
    indices = np.concatenate((coarse, extra)).astype(np.int64)
    return DriverMLSelection(indices, {**(metadata or {}), 'amp': amp,'score_power':score_power,
                                      'coarse_count': len(coarse), 'sampler': 'exponential_race_wor'})


def local_variation(coords, values, *, kind='lap_proxy', k=16, query_chunk=65536, workers=8):
    points = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if points.shape != (len(values), 3) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('variation needs aligned coordinates and at least two finite values')
    if kind not in ('lap_proxy', 'grad_sq') or k < 1:
        raise ValueError('unknown variation or invalid neighborhood')
    k = min(k, len(values)-1)
    tree = cKDTree(points)
    score = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), query_chunk):
        end = min(start+query_chunk, len(values))
        distance, neighbors = tree.query(points[start:end], k=k+1, workers=workers)
        delta = values[neighbors[:, 1:]]-values[start:end, None]
        if kind == 'lap_proxy':
            score[start:end] = np.abs(delta.mean(axis=1))
        else:
            score[start:end] = (delta**2/(distance[:, 1:]**2+1e-12)).mean(axis=1)
    return score


def lift_score(reference, coarse_coords, score, query_chunk=131072, workers=8):
    reference = np.asarray(reference, dtype=np.float64)
    tree = cKDTree(np.asarray(coarse_coords, dtype=np.float64))
    result = np.empty(len(reference), dtype=np.float64)
    for start in range(0, len(reference), query_chunk):
        idx = tree.query(reference[start:start+query_chunk], k=1, workers=workers)[1]
        result[start:start+query_chunk] = score[idx]
    return result
