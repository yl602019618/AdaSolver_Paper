"""Sparse scalar acquisition and explicit density allocation for Elasticity.

No target field is accepted by this module. Full coordinates supply geometry only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree

from .adaptive import build_patches, fps_within, subset_mass
from .base import SamplingResult
from .elasticity_contract import coarse_count, validate_coarse_indices


class InfeasibleAllocation(ValueError):
    pass


@dataclass(frozen=True)
class DensityConfig:
    coarse_fraction: float = .10
    score: str = "difference"
    neighbors: int = 8
    transfer: str = "idw4"
    quantile: float = .95
    exponent: float = 1.
    gamma: float = .9
    sampler: str = "patch"
    patches: int = 16
    aggregation: str = "mean"
    kappa: float = 8.
    placement: str = "fps"
    beta: float = .75

    def __post_init__(self):
        coarse_count(972, self.coarse_fraction)
        if self.score not in {"difference", "gradient", "residual"}:
            raise ValueError("unknown scalar acquisition")
        if self.neighbors < 2 or self.transfer not in {"nearest", "idw4", "idw8"}:
            raise ValueError("invalid neighborhood/transfer")
        if not 0 < self.quantile <= 1 or self.exponent <= 0 or not 0 <= self.gamma < 1:
            raise ValueError("invalid density transformation")
        if self.sampler not in {"patch", "weighted"} or self.placement not in {"random", "fps"}:
            raise ValueError("invalid final sampler")
        if self.aggregation not in {"mean", "q75", "max"}:
            raise ValueError("invalid patch aggregation")
        if not 1 <= self.patches <= 972 or self.kappa < 1 or not 0 <= self.beta <= 1:
            raise ValueError("invalid patch constraints/compensation")
        if not all(np.isfinite(v) for v in (self.quantile, self.exponent, self.gamma, self.kappa, self.beta)):
            raise ValueError("density parameters must be finite")


def scalar_score(coords, values, kind="difference", neighbors=8):
    x = np.asarray(coords, dtype=np.float64)
    u = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.shape != (len(u), 2) or len(u) < 2 or not np.isfinite(u).all():
        raise ValueError("finite scalar predictions must align with coarse coordinates")
    k = min(neighbors + 1, len(u))
    distance, nn = cKDTree(x).query(x, k=k)
    dx = x[nn[:, 1:]] - x[:, None, :]
    du = u[nn[:, 1:]] - u[:, None]
    d2 = np.square(distance[:, 1:]).clip(1e-12)
    if kind == "difference":
        return np.mean(du * du / d2, axis=1)
    # Relative local coordinates improve conditioning without changing physical derivatives.
    scale = np.sqrt(np.mean(d2, axis=1)).clip(1e-8)
    design = dx / scale[:, None, None]
    weight = 1. / (d2 / scale[:, None] ** 2 + .05)
    gram = np.einsum("nki,nk,nkj->nij", design, weight, design)
    rhs = np.einsum("nki,nk,nk->ni", design, weight, du)
    gram += 1e-8 * np.trace(gram, axis1=1, axis2=2)[:, None, None] * np.eye(2)
    coef = np.linalg.solve(gram + 1e-14 * np.eye(2), rhs[..., None])[..., 0]
    if kind == "gradient":
        return np.sum((coef / scale[:, None]) ** 2, axis=1)
    if kind == "residual":
        residual = du - np.einsum("nki,ni->nk", design, coef)
        return np.mean(residual ** 2 / d2, axis=1)
    raise ValueError("unknown acquisition kind")


def transfer_score(coords, coarse, score, kind):
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (len(coarse),) or not np.isfinite(score).all() or np.any(score < 0):
        raise ValueError("acquisition must have one finite nonnegative score per coarse point")
    k = {"nearest": 1, "idw4": 4, "idw8": 8}[kind]
    dist, nn = cKDTree(coords[coarse]).query(coords, k=min(k, len(coarse)))
    if k == 1:
        result = score[nn]
    else:
        weight = 1. / np.maximum(dist, 1e-12) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        result = np.sum(weight * score[nn], axis=1)
    result[coarse] = score
    return result


def density_probabilities(score, config):
    s = np.asarray(score, dtype=np.float64).clip(0)
    s = np.minimum(s, np.quantile(s, config.quantile))
    scale = s.mean()
    if scale <= 1e-30:
        return np.full(len(s), 1. / len(s))
    s = (s / scale) ** config.exponent
    return (1. - config.gamma) / len(s) + config.gamma * s / s.sum()


def integer_quotas(base, capacity, probability, total, kappa):
    """Bounded integer allocation with a verified global patch-density ratio.

    A feasible final set has a minimum density equal to n_k / A_k for some k.
    Enumerating these breakpoints gives exact integer feasibility, including
    sparse coarse sets and unequal patch capacities.
    """
    base, capacity = np.asarray(base, int), np.asarray(capacity, int)
    if np.any(capacity <= 0) or np.any(base < 0) or np.any(base > capacity):
        raise ValueError("invalid patch capacities")
    if not base.sum() <= total <= capacity.sum() or kappa < 1:
        raise InfeasibleAllocation("budget cannot contain the coarse set")
    mass = capacity / capacity.sum()
    target = total * np.asarray(probability) / np.sum(probability)
    ts = np.unique(np.concatenate([np.arange(1, c + 1) / a for c, a in zip(capacity, mass)]))
    ts = ts[(ts >= total / kappa - 1e-8) & (ts <= total + 1e-8)]
    low = np.maximum(base, np.ceil(ts[:, None] * mass - 1e-8).astype(int))
    high = np.minimum(capacity, np.floor(kappa * ts[:, None] * mass + 1e-8).astype(int))
    feasible = (low.sum(1) <= total) & (high.sum(1) >= total) & (low <= high).all(1)
    choices = np.flatnonzero(feasible)
    if not len(choices):
        raise InfeasibleAllocation(f"no integer allocation satisfies kappa={kappa}")
    preferred = np.min(target / mass)
    choice = choices[np.argmin(np.abs(ts[choices] - preferred))]
    lo, hi = low[choice], high[choice]
    counts = np.clip(np.floor(target).astype(int), lo, hi)
    while counts.sum() < total:
        priority = np.where(counts < hi, target - counts, -np.inf)
        counts[np.argmax(priority)] += 1
    while counts.sum() > total:
        priority = np.where(counts > lo, counts - target, -np.inf)
        counts[np.argmax(priority)] -= 1
    ratio = float(np.max(counts / mass) / np.min(counts / mass))
    if ratio > kappa + 1e-7:
        raise RuntimeError("integer allocator violated its density constraint")
    return counts, ratio


class ElasticityDensity:
    def __init__(self, coords):
        self.coords = np.asarray(coords, np.float64)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2 or not np.isfinite(self.coords).all():
            raise ValueError("Elasticity coordinates must be finite (N,2)")
        self.patch_cache = {}

    def patches(self, count):
        if count not in self.patch_cache:
            labels, _ = build_patches(self.coords, count, seed=0)
            # KMeans can return fewer occupied labels than requested.
            _, labels = np.unique(labels, return_inverse=True)
            members = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
            self.patch_cache[count] = (labels, members)
        return self.patch_cache[count]

    def refine(self, coarse, score, final_count, config, rng):
        c = config if isinstance(config, DensityConfig) else DensityConfig(**config)
        coarse = validate_coarse_indices(coarse, len(self.coords))
        if len(coarse) != coarse_count(len(self.coords), c.coarse_fraction):
            raise ValueError("coarse set does not match the configured fraction")
        if not len(coarse) <= final_count <= len(self.coords):
            raise ValueError("invalid final count")
        full_score = transfer_score(self.coords, coarse, score, c.transfer)
        p = density_probabilities(full_score, c)
        free = np.ones(len(p), dtype=bool)
        free[coarse] = False
        info = {"coarse_count": len(coarse), "coarse_fraction": c.coarse_fraction,
                "density_config": asdict(c), "compensation_kind": "final_voronoi"}
        if c.sampler == "weighted":
            available = np.flatnonzero(free)
            pp = p[available] / p[available].sum()
            delta = rng.choice(available, final_count - len(coarse), replace=False, p=pp)
            selected = np.concatenate((coarse, delta))
        else:
            labels, members = self.patches(c.patches)
            capacity = np.array([len(x) for x in members])
            base = np.bincount(labels[coarse], minlength=len(members))
            if c.aggregation == "mean":
                pp = np.array([p[m].sum() for m in members])
            else:
                aggregate = np.max if c.aggregation == "max" else lambda v: np.quantile(v, .75)
                # q75/max aggregate transferred scores first, then normalize on
                # patch mass. The mean path integrates the pointwise density.
                values = np.array([aggregate(full_score[m]) for m in members])
                values = np.minimum(values, np.quantile(values, c.quantile))
                values = (values / max(values.mean(), 1e-30)) ** c.exponent
                mass = capacity / capacity.sum()
                pp = ((1-c.gamma)*mass + c.gamma*mass*values / np.sum(mass*values)
                      if np.any(values > 0) else mass)
            counts, ratio = integer_quotas(base, capacity, pp, final_count, c.kappa)
            selected = coarse.copy()
            for group, count, used in zip(members, counts, base):
                available = group[free[group]]
                add = int(count - used)
                if c.placement == "fps":
                    new = fps_within(available, self.coords, selected, add,
                                     seed=int(rng.integers(1 << 30)))
                else:
                    new = rng.choice(available, add, replace=False)
                selected = np.concatenate((selected, new))
            info.update(dens_ratio=ratio, kappa=c.kappa, kappa_satisfied=True,
                        empty_coarse_patches=int((base == 0).sum()), patch_counts=counts.tolist(),
                        patch_capacity=capacity.tolist(), effective_patch_count=len(members))
        mass = subset_mass(selected, self.coords, len(p))
        relative_density = mass.mean() / mass
        compensation = (mass / mass.mean()) ** c.beta
        result = SamplingResult(selected, relative_density, compensation.astype(np.float32), info)
        result.validate(len(p), final_count)
        if not np.isin(coarse, selected).all():
            raise RuntimeError("final set lost a coarse point")
        return result
