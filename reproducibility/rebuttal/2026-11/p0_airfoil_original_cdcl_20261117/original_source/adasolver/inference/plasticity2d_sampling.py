"""Full20 spatial variation and shared refinement on the reference lattice."""
from __future__ import annotations
from functools import lru_cache
import time
import numpy as np
from scipy.spatial import cKDTree

from adasolver.inference import adaptive, adaptive_fast
from adasolver.inference.base import SamplingResult
from adasolver.inference.plasticity2d_contract import N_FULL, N_COARSE, validate_indices
from adasolver.utils.interpolation import index_grid

REFERENCE = index_grid((101, 31))


def variation_features(positions, prediction, k=16):
    """Weighted local quadratic fit; raw units and physical spacing for all 40 outputs."""
    xy = np.asarray(positions, np.float64) * .5
    values = np.asarray(prediction, np.float64)
    if values.ndim != 2 or values.shape[1] != 40 or xy.shape != (len(values), 2):
        raise ValueError("variation requires aligned positions and 40 physical displacement outputs")
    if not np.isfinite(values).all() or not np.isfinite(xy).all() or len(xy) < 7:
        raise ValueError("variation requires finite fields and at least 7 points")
    distances, neighbors = cKDTree(xy).query(xy, k=min(k + 1, len(xy)))
    neighbors, distances = neighbors[:, 1:], distances[:, 1:]
    dx = xy[neighbors] - xy[:, None]
    scale = np.maximum(distances.max(axis=1), 1e-12)
    dx /= scale[:, None, None]
    x, y = dx[..., 0], dx[..., 1]
    design = np.stack((x, y, .5*x*x, x*y, .5*y*y), axis=-1)
    weight = 1. / np.maximum(distances / scale[:, None], .1)
    lhs = np.einsum("nki,nk,nkj->nij", design, weight, design)
    rhs = np.einsum("nki,nk,nkc->nic", design, weight, values[neighbors] - values[:, None])
    coef = np.linalg.solve(lhs + 1e-8*np.eye(5)[None], rhs)
    gradient = coef[:, :2] / scale[:, None, None]
    laplacian = (coef[:, 2] + coef[:, 4]) / scale[:, None]**2
    increment = np.diff(gradient.reshape(len(xy), 2, 20, 2), axis=2)
    return {
        "g": np.sqrt(np.square(gradient).sum(axis=(1, 2))),
        "l": np.sqrt(np.square(laplacian).sum(axis=1)),
        "increment": np.sqrt(np.square(increment).sum(axis=(1, 2, 3))),
    }


def normalize(values):
    score = np.asarray(values, np.float64)
    if not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("acquisition score must be finite and nonnegative")
    return score / max(float(score.mean()), 1e-12)


def density_from_score(score, gamma=.8, power=1.):
    if not 0 <= gamma < 1 or power <= 0:
        raise ValueError("gamma must be in [0,1) and power must be positive")
    scaled = normalize(score)
    scaled = normalize(scaled**power)
    if scaled.max() == 0:
        return np.ones_like(scaled)
    return (1-gamma) + gamma*scaled


def solution_score(features, config):
    kind = config.get("score", "g")
    if kind == "gl":
        mix = config.get("laplacian_weight", .5)
        return (1-mix)*normalize(features["g"]) + mix*normalize(features["l"])
    return features[kind]


def lift_score(coarse, scores, backend="nearest"):
    tree = cKDTree(REFERENCE[coarse])
    if backend == "nearest":
        return np.asarray(scores)[tree.query(REFERENCE)[1]]
    if backend != "idw":
        raise ValueError(f"unknown score lift {backend}")
    dist, neighbors = tree.query(REFERENCE, k=4)
    weights = 1. / np.maximum(dist, 1e-12)**2
    return (weights * np.asarray(scores)[neighbors]).sum(1) / weights.sum(1)


@lru_cache(maxsize=12)
def patches(count):
    ids, mass = adaptive.build_patches(REFERENCE, count, seed=0)
    return ids, mass, adaptive_fast.build_patch_cache(REFERENCE, ids, count)


def finish(indices, density, beta, metadata):
    started = time.perf_counter()
    compensation = adaptive.comp_weights_for_subset(indices, REFERENCE, N_FULL, beta)
    metadata = dict(metadata, compensation_type="final_subset_voronoi_mass", beta=beta,
                    compensation_seconds=time.perf_counter()-started)
    result = SamplingResult(np.asarray(indices, np.int64), np.asarray(density)[indices], compensation, metadata)
    result.validate(N_FULL, len(indices))
    return result


def sample_pi_full20(count, rng, profile, config):
    ell = config.get("ell", .1)
    if ell <= 0:
        raise ValueError("PI contact length must be positive")
    score = np.exp(-REFERENCE[:, 1]/30/ell)
    weight = config.get("curvature_weight", 0.)
    if weight:
        profile = np.asarray(profile).reshape(101)
        curvature = np.abs(np.gradient(np.gradient(profile, .5), .5))
        score *= np.repeat(1 + weight*normalize(curvature), 31)
    density = density_from_score(score, config.get("gamma", .8), config.get("power", 1.))
    indices = rng.choice(N_FULL, count, replace=False, p=density/density.sum())
    return finish(indices, density, config.get("beta", 1.), {"method": "pi", "coarse_count": 0})


def refine_full20(coarse, scores, count, seed, config):
    validate_indices(coarse, N_COARSE)
    density = density_from_score(lift_score(coarse, scores, config.get("lift", "nearest")),
                                 config.get("gamma", .8), config.get("power", 1.))
    k = config.get("patch_count", 24)
    pid, mass, members = patches(k)
    patch_scores = np.bincount(pid, weights=density, minlength=k) / np.bincount(pid, minlength=k)
    indices, info = adaptive_fast.refine_once(
        coarse, density, REFERENCE, pid, mass, count, kappa=config.get("kappa", 4.),
        seed=seed, patch_cache=members, placement=config.get("placement", "fps"),
        patch_scores=patch_scores,
    )
    validate_indices(indices, count, coarse)
    if not info["kappa_satisfied"]:
        raise ValueError(f"requested kappa was violated: {info['dens_ratio']}")
    return finish(indices, density, config.get("beta", 1.), dict(info, coarse_count=N_COARSE,
                  patch_score_source="full_lifted_density", patch_count=k,
                  zero_score_fallback=bool(not np.any(np.asarray(scores) > 0))))
