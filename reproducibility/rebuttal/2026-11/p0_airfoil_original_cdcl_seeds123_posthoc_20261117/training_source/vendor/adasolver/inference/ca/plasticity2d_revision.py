"""CA full20 acquisition-to-density experiments; the original sampler stays frozen."""
from __future__ import annotations

import numpy as np

from adasolver.inference import adaptive_fast
from adasolver.inference.plasticity2d_contract import (
    N_COARSE, N_FULL, validate_indices, validate_method_config,
)
from adasolver.inference.plasticity2d_sampling import (
    REFERENCE, density_from_score, finish, lift_score, normalize, patches,
)

REVISION_KEYS = {"revision", "allocation", "score_control", "prior_mix", "prior_ell", "prior_curvature"}


def validate_config(config):
    if config.get("revision") != "ca_v2" or config.get("method") != "ca":
        raise ValueError("expected a ca_v2 configuration")
    validate_method_config({k: v for k, v in config.items() if k not in REVISION_KEYS})
    if config.get("allocation", "greedy") not in ("greedy", "proportional", "weighted"):
        raise ValueError("unknown CA allocation")
    if config.get("score_control", "real") not in ("real", "constant", "shuffle", "prior"):
        raise ValueError("unknown acquisition control")
    if not 0 <= config.get("prior_mix", 0.) <= 1:
        raise ValueError("prior_mix must be in [0,1]")
    if config.get("prior_ell", .1) <= 0 or config.get("prior_curvature", .5) < 0:
        raise ValueError("invalid geometric prior")


def prior_score(profile, ell=.1, curvature_weight=.5):
    curvature = np.abs(np.gradient(np.gradient(np.asarray(profile).reshape(101), .5), .5))
    return np.exp(-REFERENCE[:, 1]/30/ell) * np.repeat(1+curvature_weight*normalize(curvature), 31)


def proportional_counts(capacity, coarse_counts, patch_density, count, kappa):
    """Water-fill final quotas, retaining coarse points and a bounded density ratio.

    A common density floor t=max(r/sqrt(kappa), max(coarse density)/kappa)
    defines integer bounds ceil(t*n_k) <= q_k <= floor(kappa*t*n_k).
    Within them q_k is proportional to n_k*density_k. Largest remainders give
    an exact integer budget. Infeasible integer bounds are explicit failures.
    """
    capacity = np.asarray(capacity, np.int64)
    coarse_counts = np.asarray(coarse_counts, np.int64)
    density = np.asarray(patch_density, np.float64)
    if (capacity.ndim != 1 or capacity.shape != coarse_counts.shape or capacity.shape != density.shape
            or np.any(capacity <= 0) or np.any(coarse_counts < 0) or np.any(coarse_counts > capacity)
            or not np.isfinite(density).all() or np.any(density < 0) or kappa < 1
            or not coarse_counts.sum() <= count <= capacity.sum()):
        raise ValueError("invalid proportional allocation inputs")
    floor = max(count/capacity.sum()/np.sqrt(kappa), float((coarse_counts/capacity).max())/kappa)
    lower = np.maximum(coarse_counts, np.ceil(floor*capacity-1e-10).astype(np.int64))
    upper = np.minimum(capacity, np.floor(kappa*floor*capacity+1e-10).astype(np.int64))
    if np.any(lower > upper) or lower.sum() > count or upper.sum() < count:
        raise ValueError("infeasible integer density bounds; increase kappa or reduce patch count")
    mass = capacity*np.maximum(density, 1e-12)
    lo, hi = 0., float(np.max(upper/mass))
    for _ in range(60):
        mid = (lo+hi)/2
        if np.clip(mid*mass, lower, upper).sum() < count:
            lo = mid
        else:
            hi = mid
    target = np.clip((lo+hi)/2*mass, lower, upper)
    quotas = np.maximum(lower, np.floor(target+1e-10).astype(np.int64))
    remaining = int(count-quotas.sum())
    eligible = np.flatnonzero(quotas < upper)
    order = eligible[np.argsort(-(target-quotas)[eligible], kind="stable")]
    if remaining < 0 or remaining > len(order):
        raise RuntimeError("invalid water-fill rounding")
    quotas[order[:remaining]] += 1
    return quotas


def refine_ca(coarse, scores, profile, count, seed, config):
    validate_config(config)
    validate_indices(coarse, N_COARSE)
    coarse = np.asarray(coarse, np.int64)
    scores = np.asarray(scores, np.float64)
    if scores.shape != (N_COARSE,) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("expected 313 finite nonnegative acquisition scores")
    control = config.get("score_control", "real")
    if control == "shuffle":
        # Independent stream: preserve the score histogram and the placement RNG.
        scores = np.random.default_rng(np.random.SeedSequence([seed, 73])).permutation(scores)
    if control == "constant":
        scores = np.ones_like(scores)
    full_score = lift_score(coarse, scores, config.get("lift", "nearest"))
    mix = config.get("prior_mix", 0.)
    if mix or control == "prior":
        prior = prior_score(profile, config.get("prior_ell", .1), config.get("prior_curvature", .5))
        full_score = prior if control == "prior" else (1-mix)*normalize(full_score)+mix*normalize(prior)
    density = density_from_score(full_score, config.get("gamma", .8), config.get("power", 1.))
    allocation = config.get("allocation", "greedy")
    info = {"allocation": allocation, "score_control": control, "prior_mix": mix}
    if allocation == "weighted":
        available = np.setdiff1d(np.arange(N_FULL), coarse, assume_unique=True)
        probability = density[available]/density[available].sum()
        new = np.random.default_rng(seed).choice(available, count-N_COARSE, replace=False, p=probability)
        indices = np.concatenate((coarse, new))
    else:
        k = config.get("patch_count", 24)
        pid, mass, members = patches(k)
        capacity = np.bincount(pid, minlength=k)
        patch_density = np.bincount(pid, weights=density, minlength=k)/capacity
        base_counts = np.bincount(pid[coarse], minlength=k)
        if allocation == "greedy":
            indices, legacy_info = adaptive_fast.refine_once(
                coarse, density, REFERENCE, pid, mass, count, kappa=config.get("kappa", 4.),
                seed=seed, patch_cache=members, placement=config.get("placement", "fps"),
                patch_scores=patch_density,
            )
            info.update(legacy_info)
            if not info["kappa_satisfied"]:
                raise ValueError("greedy allocator violated kappa")
        else:
            quotas = proportional_counts(capacity, base_counts, patch_density, count, config.get("kappa", 4.))
            rng = np.random.default_rng(seed)
            coarse_mask = np.zeros(N_FULL, bool)
            coarse_mask[coarse] = True
            additions = []
            for p, cells in enumerate(members):
                candidates = cells[~coarse_mask[cells]]
                n = int(quotas[p]-base_counts[p])
                chosen = (adaptive_fast._fps_fast(candidates, REFERENCE, coarse, n, rng)
                          if config.get("placement", "fps") == "fps"
                          else rng.choice(candidates, n, replace=False))
                additions.append(chosen)
            indices = np.concatenate([coarse, *additions])
        actual = np.bincount(pid[indices], minlength=k)
        realized_density = actual/capacity
        ratio = float(realized_density.max()/realized_density.min())
        if ratio > config.get("kappa", 4.)+1e-8:
            raise ValueError("final patch density ratio exceeds kappa")
        info.update(patch_count=k, patch_capacity=capacity.tolist(), patch_density=patch_density.tolist(),
                    patch_coarse_counts=base_counts.tolist(), patch_final_counts=actual.tolist(),
                    dens_ratio=ratio, kappa=config.get("kappa", 4.), kappa_satisfied=True)
    validate_indices(indices, count, coarse)
    info.update(coarse_count=N_COARSE, n_add=count-N_COARSE,
                zero_score_fallback=bool(not np.any(scores > 0)))
    return finish(indices, density, config.get("beta", 1.), info)
