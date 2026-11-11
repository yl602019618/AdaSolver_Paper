"""GPU execution of the frozen Elasticity density and sequential patch FPS."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch
import triton
import triton.language as tl

from inference_speed.gpu_fps import GPUSparseFPS
from inference_speed.gpu_geometry import quantiles
from inference_speed.sampling import SparseFPSPlan


@triton.jit
def _integer_counts(target, lower, upper, output, K: tl.constexpr,
                    B: tl.constexpr, TOTAL: tl.constexpr):
    lane = tl.arange(0, B)
    t = tl.load(target + lane, lane < K, other=0.)
    lo = tl.load(lower + lane, lane < K, other=0)
    hi = tl.load(upper + lane, lane < K, other=0)
    count = tl.minimum(tl.maximum(tl.floor(t).to(tl.int64), lo), hi)
    total = tl.sum(count, 0)
    while total < TOTAL:
        priority = tl.where((lane < K) & (count < hi), t - count, -float("inf"))
        maximum = tl.max(priority, 0)
        best = tl.min(tl.where(priority == maximum, lane, 2147483647), 0)
        count += (lane == best).to(tl.int64)
        total += 1
    while total > TOTAL:
        priority = tl.where((lane < K) & (count > lo), count - t, -float("inf"))
        maximum = tl.max(priority, 0)
        best = tl.min(tl.where(priority == maximum, lane, 2147483647), 0)
        count -= (lane == best).to(tl.int64)
        total -= 1
    tl.store(output + lane, count, lane < K)


class Elasticity10Plan:
    def __init__(self, case, prepared_mass=None):
        self.case = case
        self.device = case.executor.device
        self.config = c = case.gpu_spec["density"]
        if c.score != "difference" or c.transfer != "idw8" or c.placement != "fps":
            raise ValueError("Expected the frozen stress-difference / IDW8 / FPS policy")
        if c.sampler != "patch" or c.aggregation not in {"mean", "max"}:
            raise ValueError("Expected the frozen regional density")
        if len(case.coarse) != 97 or len(case.reference) != 972:
            raise ValueError("Elasticity acquisition must use 97 of 972 points")
        self.count = round(972 * case.budget)
        self.mean = float(case.gpu_spec["output_mean"])
        self.std = float(case.gpu_spec["output_std"])
        self.coarse = torch.as_tensor(case.coarse, device=self.device)
        # The CUDA FPS kernel addresses coordinates as a contiguous (N, 2) array.
        coords = np.ascontiguousarray(case.reference, dtype=np.float64)

        distance, neighbors = cKDTree(coords[case.coarse]).query(coords[case.coarse], k=9)
        self.neighbors = torch.as_tensor(neighbors[:, 1:], device=self.device)
        self.distance2 = torch.as_tensor(np.square(distance[:, 1:]).clip(1e-12), device=self.device)
        dist, rows = cKDTree(coords[case.coarse]).query(coords, k=8)
        weights = 1. / np.maximum(dist, 1e-12) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        self.transfer_rows = torch.as_tensor(rows, device=self.device)
        self.transfer_weights = torch.as_tensor(weights, device=self.device)

        labels, members = case.gpu_spec["geometry"].patches(c.patches)
        capacity = np.array([len(group) for group in members])
        base = np.bincount(labels[case.coarse], minlength=len(members))
        mass = capacity / capacity.sum()
        self.k = len(members)
        self.mass_per_patch = torch.as_tensor(mass, device=self.device)
        self.base = torch.as_tensor(base, device=self.device)
        self.capacity = torch.as_tensor(capacity, device=self.device)
        padded = np.zeros((self.k, capacity.max()), np.int64)
        mask = np.zeros_like(padded, dtype=bool)
        for i, group in enumerate(members):
            padded[i, :len(group)] = group
            mask[i, :len(group)] = True
        self.members = torch.as_tensor(padded, device=self.device)
        self.member_mask = torch.as_tensor(mask, device=self.device)

        # These feasibility intervals depend only on geometry, coarse points and budget.
        # The interval choice and integer counts are recomputed from each prediction.
        ts = np.unique(np.concatenate([np.arange(1, n + 1) / a for n, a in zip(capacity, mass)]))
        ts = ts[(ts >= self.count / c.kappa - 1e-8) & (ts <= self.count + 1e-8)]
        lower = np.maximum(base, np.ceil(ts[:, None] * mass - 1e-8).astype(int))
        upper = np.minimum(capacity, np.floor(c.kappa * ts[:, None] * mass + 1e-8).astype(int))
        valid = ((lower.sum(1) <= self.count) & (upper.sum(1) >= self.count)
                 & (lower <= upper).all(1))
        if not valid.any():
            raise ValueError("No feasible integer patch allocation")
        self.intervals = torch.as_tensor(ts[valid], device=self.device)
        self.lower = torch.as_tensor(lower[valid], device=self.device)
        self.upper = torch.as_tensor(upper[valid], device=self.device)

        free = np.ones(972, dtype=bool)
        free[case.coarse] = False
        candidates = [group[free[group]] for group in members]
        sizes = np.array([len(group) for group in candidates])
        self.selector = SimpleNamespace(
            k=self.k, n0=97, max_capacity=int(sizes.max()),
            orders=torch.as_tensor(np.concatenate(candidates), device=self.device),
            offset=torch.as_tensor(np.r_[0, sizes.cumsum()], device=self.device),
        )
        self.fps = GPUSparseFPS(coords, SparseFPSPlan(coords, case.coarse), self.device)
        self.policy = ("frozen 97-point stress-difference density; exact bounded integer quotas; "
                       "sequential cross-patch FPS; fresh coarse and final predictions")

    def patch_probabilities(self, raw):
        field = (raw[0].reshape(-1) * self.std + self.mean).double()
        difference = field[self.neighbors] - field[:, None]
        score = (difference.square() / self.distance2).mean(1)
        full = (score[self.transfer_rows] * self.transfer_weights).sum(1)
        full[self.coarse] = score
        c = self.config
        if c.aggregation == "mean":
            clipped = torch.minimum(full, quantiles(full, [c.quantile])[0])
            scale = clipped.mean()
            transformed = (clipped / scale.clamp_min(1e-30)).pow(c.exponent)
            p = ((1. - c.gamma) / 972
                 + c.gamma * transformed / transformed.sum().clamp_min(1e-30))
            p = torch.where(scale <= 1e-30, torch.full_like(p, 1. / 972), p)
            return torch.where(self.member_mask, p[self.members], 0.).sum(1)
        values = torch.where(self.member_mask, full[self.members], 0.).max(1).values
        values = torch.minimum(values, quantiles(values, [c.quantile])[0])
        values = (values / values.mean().clamp_min(1e-30)).pow(c.exponent)
        weighted = self.mass_per_patch * values
        p = ((1. - c.gamma) * self.mass_per_patch
             + c.gamma * weighted / weighted.sum().clamp_min(1e-30))
        return torch.where(values.max() > 0, p, self.mass_per_patch)

    def select_indices(self, raw):
        probability = self.patch_probabilities(raw)
        target = self.count * probability / probability.sum()
        preferred = (target / self.mass_per_patch).min()
        choice = (self.intervals - preferred).abs().argmin().reshape(1)
        lower = self.lower.index_select(0, choice)[0]
        upper = self.upper.index_select(0, choice)[0]
        counts = torch.empty(self.k, dtype=torch.int64, device=self.device)
        _integer_counts[(1,)](target, lower, upper, counts, self.k,
                             triton.next_power_of_2(self.k), self.count,
                             num_warps=4, enable_fp_fusion=False)
        added = counts - self.base
        before = added.cumsum(0) - added
        selected = torch.empty(self.count, dtype=torch.int64, device=self.device)
        selected[:97] = self.coarse
        self.fps(self.selector, added, before, selected)
        return selected
