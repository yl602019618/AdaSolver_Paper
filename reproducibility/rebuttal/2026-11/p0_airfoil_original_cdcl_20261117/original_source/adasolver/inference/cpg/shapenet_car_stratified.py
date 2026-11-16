"""Surface-preserving refinement with pressure-conditioned volume strata.

All surface points fit in the prescribed Car budgets. The remaining budget
follows wall distance and the one coarse pressure signal. The legacy count
correction assigns equal mass to retained points within each stratum; this
is quadrature, not exact pointwise inclusion probability under FPS. Optional
Voronoi corrections use the realized reference-cell masses instead.
"""
from __future__ import annotations

import numpy as np

from adasolver.inference.base import SamplingResult
from adasolver.inference.pi.shapenet_car import car_surface_distance
from adasolver.inference.shapenet_car_contract import coarse_count, validate_indices


def allocate_counts(mass, lower, priority, total):
    """Capped proportional allocation with exact integer sum and retained S0."""
    mass = np.asarray(mass, dtype=np.int64)
    lower = np.asarray(lower, dtype=np.int64)
    priority = np.asarray(priority, dtype=np.float64)
    if (np.any(lower < 0) or np.any(lower > mass) or
            not lower.sum() <= total <= mass.sum() or
            np.any(priority <= 0) or not np.isfinite(priority).all()):
        raise ValueError("invalid stratified allocation")
    lo, hi = 0., float(np.max(mass / priority)) + 1.
    for _ in range(60):
        mid = (lo + hi) / 2
        count = np.clip(mid * priority, lower, mass)
        if count.sum() < total:
            lo = mid
        else:
            hi = mid
    real = np.clip(hi * priority, lower, mass)
    counts = np.floor(real).astype(np.int64)
    remaining = int(total - counts.sum())
    available = np.flatnonzero(counts < mass)
    order = available[np.argsort(-(real - counts)[available], kind="stable")]
    counts[order[:remaining]] += 1
    if counts.sum() != total:
        raise RuntimeError("stratified allocation failed to match budget")
    return counts


class ShapeNetCarStratified:
    def __init__(self, coords, surface_mask, *, wall_scale=.1, wall_amplitude=8.,
                 physics_strength=1., beta=1., placement="random", score_bins=4,
                 correction="stratum_count", wall_mask=None):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.mask = np.asarray(surface_mask, dtype=bool)
        if self.coords.shape != (len(self.mask), 3) or not self.mask.any():
            raise ValueError("invalid reference geometry")
        if wall_scale <= 0 or wall_amplitude < 0 or physics_strength < 0 or beta < 0:
            raise ValueError("invalid stratified density parameters")
        if placement not in {"random", "fps"} or score_bins < 1:
            raise ValueError("invalid stratified placement")
        if correction not in {"stratum_count", "voronoi", "phase_voronoi"}:
            raise ValueError("unknown stratified correction")
        self.distance = car_surface_distance(self.coords, self.mask, wall_mask=wall_mask)
        self.wall_mask = self.mask if wall_mask is None else np.asarray(wall_mask, dtype=bool)
        self.wall_reference = "tagged_surface" if wall_mask is None else "physical_body"
        self.scale, self.amplitude = wall_scale, wall_amplitude
        self.strength, self.beta = physics_strength, beta
        self.placement, self.bins = placement, int(score_bins)
        self.correction = correction

    def refine_score(self, coarse, score, final_count, rng):
        n = len(self.coords)
        coarse = validate_indices(coarse, n, coarse_count(n))
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (n,) or not np.isfinite(score).all() or np.any(score < 0):
            raise ValueError("score must be finite and nonnegative on the reference cloud")
        selected = np.zeros(n, dtype=bool)
        selected[coarse] = True
        surface_extra = np.flatnonzero(self.mask & ~selected)
        anchors = np.concatenate((coarse, surface_extra))
        if not len(anchors) <= final_count <= n:
            raise ValueError("budget cannot retain the coarse set and full surface")
        selected[surface_extra] = True
        # Surface is its own stratum. Volume strata resolve distance and the
        # coarse pressure score separately; zero-strength is a geometry ablation.
        strata = np.zeros(n, dtype=np.int64)
        shell = np.searchsorted(self.scale * np.array([.25, .5, 1., 2., 4., 8., 16.]),
                                self.distance)
        next_id = 1
        for band in np.unique(shell[~self.mask]):
            ids = np.flatnonzero((shell == band) & ~self.mask)
            if self.strength > 0 and self.bins > 1:
                cuts = np.unique(np.quantile(score[ids], np.arange(1, self.bins) / self.bins))
                group = np.searchsorted(cuts, score[ids], side="left")
            else:
                group = np.zeros(len(ids), dtype=int)
            for rank in np.unique(group):
                strata[ids[group == rank]] = next_id
                next_id += 1
        mass = np.bincount(strata, minlength=next_id)
        lower = np.bincount(strata[anchors], minlength=next_id)
        scale = max(float(np.quantile(score[self.mask], .95)), 1e-12)
        physics = np.clip(score / scale, 0., 1.)
        weights = ((1. + self.amplitude * np.exp(-self.distance / self.scale)) *
                   (1. + self.strength * physics))
        priority = np.bincount(strata, weights=weights, minlength=next_id)
        counts = allocate_counts(mass, lower, priority, final_count)
        extra = []
        for group in range(1, next_id):
            candidates = np.flatnonzero((strata == group) & ~selected)
            count = int(counts[group] - lower[group])
            if count == 0:
                continue
            if self.placement == "random":
                chosen = rng.choice(candidates, count, replace=False)
            else:
                from adasolver.inference.adaptive import fps_within
                chosen = fps_within(candidates, self.coords, np.flatnonzero(selected), count,
                                    seed=int(rng.integers(1 << 30)), distance_backend="native")
            extra.append(chosen)
            selected[chosen] = True
        indices = np.concatenate([anchors, *extra])
        density = counts[strata[indices]] / mass[strata[indices]]
        estimator = "stratum_count_ratio"
        if self.correction != "stratum_count":
            from adasolver.inference.finite_measure import reference_cell_weights
            groups = self.mask if self.correction == "phase_voronoi" else None
            cell_correction = reference_cell_weights(self.coords, indices, groups=groups)
            density = 1. / cell_correction
            estimator = ("reference_cell_surface_volume" if groups is not None
                         else "reference_cell_global")
        compensation = np.power(1. / density, self.beta).astype(np.float32)
        result = SamplingResult(indices, density, compensation, {
            "method": "cpg", "strategy": "surface_stratified", "truth_access": False,
            "coarse_count": len(coarse), "coarse_fraction": .1,
            "surface_coverage": 1., "surface_selected_count": int(self.mask.sum()),
            "density_estimator": estimator, "placement": self.placement,
            "physics_strength": self.strength, "wall_scale": self.scale,
            "wall_reference": self.wall_reference, "wall_point_count": int(self.wall_mask.sum()),
            "wall_amplitude": self.amplitude, "beta": self.beta,
            "stratum_reference_counts": mass.tolist(), "stratum_selected_counts": counts.tolist(),
        })
        result.validate(n, final_count)
        return result
