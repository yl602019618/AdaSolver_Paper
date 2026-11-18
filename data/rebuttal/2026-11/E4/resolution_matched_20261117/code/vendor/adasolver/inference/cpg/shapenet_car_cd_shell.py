"""One-pressure-signal correction to the Car near-wall sampling policy.

The coarse set is the first 10% of the same uniform permutation used for the
final query's base. Turning off guidance reproduces the corresponding PI
indices and shell-mass compensation exactly, including its fallback policy.
"""
from __future__ import annotations

import numpy as np

from adasolver.inference.base import SamplingResult
from adasolver.inference.pi.shapenet_car import car_surface_distance
from adasolver.inference.shapenet_car_contract import coarse_count, validate_indices


class ShapeNetCarCdShell:
    def __init__(self, coords, surface_mask, *, shell_radius=.1, frac_extra=.375,
                 physics_strength=1., beta=.5, compensation="shell_mass"):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.mask = np.asarray(surface_mask, dtype=bool)
        parameters = np.array([shell_radius, frac_extra, physics_strength, beta])
        if (not np.isfinite(parameters).all() or shell_radius <= 0 or
                not 0 <= frac_extra < 1 or physics_strength < 0 or beta < 0):
            raise ValueError("invalid Cd shell parameters")
        if compensation not in {"shell_mass", "voronoi"}:
            raise ValueError("unsupported Cd shell compensation")
        self.distance = car_surface_distance(self.coords, self.mask)
        self.radius, self.frac_extra = shell_radius, frac_extra
        self.strength, self.beta, self.compensation = physics_strength, beta, compensation

    def prepare(self, final_count, rng):
        n = len(self.coords)
        n_extra = int(round(final_count * self.frac_extra))
        n_base = final_count - n_extra
        if not coarse_count(n) <= n_base <= final_count <= n:
            raise ValueError("uniform base must contain the complete 10% coarse set")
        return rng.permutation(n)[:n_base].astype(np.int64, copy=False)

    def refine_from_base(self, base, score, final_count, rng):
        n = len(self.coords)
        n_extra = int(round(final_count * self.frac_extra))
        base = validate_indices(base, n, final_count - n_extra)
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (n,) or not np.isfinite(score).all() or np.any(score < 0):
            raise ValueError("Cd acquisition score must be finite, nonnegative, and align with the cloud")
        low = float(score.min())
        scale = max(float(np.quantile(score[self.mask], .95)) - low, 1e-12)
        phi = np.clip((score - low) / scale, 0., 1.)
        selected = np.zeros(n, dtype=bool)
        selected[base] = True
        shell_mask = self.distance < self.radius
        available = np.flatnonzero(shell_mask & ~selected)
        if len(available) >= n_extra:
            probability = None
            weights = 1. + self.strength * phi[available]
            if self.strength > 0 and np.ptp(weights) > 0:
                probability = weights / weights.sum()
            extra = rng.choice(available, n_extra, replace=False, p=probability)
            fallback = 0
        else:
            fallback = n_extra - len(available)
            rest = np.flatnonzero(~shell_mask & ~selected)
            effective_distance = self.distance[rest] / (1. + self.strength * phi[rest])
            # The default argsort is intentionally identical to historical PI
            # when strength=0; score only changes the fallback priority.
            extra = np.concatenate((available, rest[np.argsort(effective_distance)[:fallback]]))
        indices = np.concatenate((base, extra))
        if self.compensation == "shell_mass":
            shell_fraction = max(float(shell_mask.mean()), 1e-12)
            base_fraction = len(base) / n
            shell_weight = (base_fraction + n_extra / n / shell_fraction) / max(base_fraction, 1e-12)
            density = np.ones(final_count)
            density[shell_mask[indices]] = shell_weight
            if fallback:
                density[len(base) + len(available):] = shell_weight
            weights = np.power(1. / density, self.beta).astype(np.float32)
            density /= density.sum()
        else:
            from adasolver.inference.adaptive import comp_weights_for_subset
            weights = comp_weights_for_subset(indices, self.coords, n, beta=self.beta)
            density = (np.power(np.maximum(weights, 1e-12), -1. / self.beta)
                       if self.beta else np.ones(len(indices)))
        result = SamplingResult(indices, density, weights, {
            "strategy": "cd_shell", "truth_access": False, "single_predicted_scalar": "pressure",
            "coarse_count": coarse_count(n), "coarse_fraction": .1,
            "coarse_policy": "uniform_permutation_prefix_per_budget",
            "base_count": len(base), "extra_count": n_extra, "fallback_count": fallback,
            "shell_count": int(shell_mask.sum()), "shell_radius": self.radius,
            "frac_extra": self.frac_extra, "physics_strength": self.strength,
            "density_estimator": self.compensation, "beta": self.beta,
            "surface_coverage": float(self.mask[indices].sum() / self.mask.sum()),
        })
        result.validate(n, final_count)
        return result
