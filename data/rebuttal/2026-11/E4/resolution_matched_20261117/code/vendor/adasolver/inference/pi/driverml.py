"""Geometry-only DriverML acquisition; pressure and Cd use separate priors."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from adasolver.inference.driverml_sampling import select_from_score


def geometry_prior(geometry, prior='normal_variation', k=16, workers=8):
    pos = geometry.pos.numpy()
    normal = geometry.normal.numpy()
    area = geometry.area.numpy()
    if prior == 'drag_sensitivity':
        return np.abs(normal[:, 0])*area
    if prior == 'normal_variation':
        tree = cKDTree(pos)
        result = np.empty(len(pos), dtype=np.float64)
        neighbors = min(k+1, len(pos))
        for start in range(0, len(pos), 65536):
            stop = min(start+65536, len(pos))
            idx = tree.query(pos[start:stop], k=neighbors, workers=workers)[1][:, 1:]
            result[start:stop] = ((normal[idx]-normal[start:stop, None])**2).sum(axis=-1).mean(axis=1)
        return result
    x = (pos[:, 0]-pos[:, 0].min())/max(float(np.ptp(pos[:, 0])), 1e-12)
    z = (pos[:, 2]-pos[:, 2].min())/max(float(np.ptp(pos[:, 2])), 1e-12)
    if prior == 'lower_front':
        return (1-z)*np.exp(-0.5*((x-0.25)/0.35)**2)
    if prior == 'front_rear':
        return np.maximum(np.clip((0.15-x)/0.15, 0, 1), np.clip((x-0.85)/0.15, 0, 1))*np.abs(normal[:, 0])
    raise ValueError(f'unknown DriverML geometry prior: {prior}')


def sample_driverml_pi(geometry, count, rng, amp=8.0, prior='drag_sensitivity', k=16, score=None):
    raw = geometry_prior(geometry, prior, k) if score is None else score
    return select_from_score(raw, count, rng, amp=amp,
                             metadata={'method': 'pi', 'prior': prior, 'truth_access': False})
