"""Spatial budgets, split identities and RNG streams for Elasticity refinement."""
from __future__ import annotations

import math
import numpy as np

NFULL = 972
BUDGETS = (0.375, 0.5, 0.75)
SPLITS = {
    "head_train": (0, 1000),
    "head_validation": (1000, 1100),
    "development": (1100, 1400),
    "confirmation": (1400, 1600),
    "independent_holdout": (1600, 1800),
    "historical_test": (1800, 2000),
}


def coarse_count(nfull=NFULL, fraction=.10):
    if not math.isfinite(fraction) or not 0 < fraction <= .10:
        raise ValueError("Elasticity coarse fraction must be in (0, 0.10]")
    count = math.floor(nfull * fraction)
    if count < 2:
        raise ValueError("coarse input needs at least two points")
    return count


def validate_coarse_indices(indices, nfull=NFULL):
    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("coarse indices must be a one-dimensional integer array")
    if not 2 <= len(values) <= math.floor(.10 * nfull):
        raise ValueError("coarse input exceeds the 10% cap or has fewer than two points")
    if len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= nfull):
        raise ValueError("coarse indices must be unique and within the full grid")
    return values.astype(np.int64, copy=False)


def sample_rng(sample_id, seed, *, refinement=False):
    sid, seed = int(sample_id), int(seed)
    if sid < 0 or seed < 0:
        raise ValueError("sample ID and seed must be nonnegative")
    namespace = 7_000_000 if refinement else 0
    if 1800 <= sid < 2000:
        return np.random.default_rng(namespace + 1000 * seed + sid - 1800)
    return np.random.default_rng(np.random.SeedSequence([1_000_000 + namespace, sid, seed]))


def split_ids(name, limit=None):
    start, end = SPLITS[name]
    if limit is not None:
        if not 1 <= limit <= end - start:
            raise ValueError("sample limit is outside the selected split")
        end = start + limit
    return list(range(start, end))
