from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def index_grid(shape: tuple[int, int]) -> np.ndarray:
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return np.stack((i.reshape(-1), j.reshape(-1)), axis=1).astype(np.float64)


def reconstruct_field(indices: np.ndarray, prediction: np.ndarray,
                      reference_coords: np.ndarray) -> tuple[np.ndarray, int]:
    indices = np.asarray(indices, dtype=np.int64)
    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    source = reference_coords[indices]
    reconstructed = griddata(source, prediction, reference_coords,
                             method="linear", fill_value=np.nan)
    missing = np.isnan(reconstructed).any(axis=1)
    fallback_count = int(missing.sum())
    if fallback_count:
        reconstructed[missing] = griddata(source, prediction, reference_coords[missing],
                                          method="nearest")
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("interpolation produced non-finite values after nearest fallback")
    return reconstructed, fallback_count
