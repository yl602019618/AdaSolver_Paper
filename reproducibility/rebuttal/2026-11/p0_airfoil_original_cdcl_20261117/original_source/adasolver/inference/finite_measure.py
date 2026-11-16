"""Finite-population inclusion probabilities and reference-cell quadrature.

These are different constructions. Inclusion probabilities average over a
declared random sampling design. Cell masses describe one realized point set
and give a deterministic quadrature approximation. Neither makes a finite
self-normalized token, or a complete neural network, an unbiased estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import betabinom, hypergeom


@dataclass(frozen=True)
class TwoStageDesign:
    """Uniform B-subset followed by E shell/nearest-outside additions.

    Outside points must have a fixed total priority independent of the random
    base. The formulas are derived and exhaustively checked in the project's
    beta_ablation/theory_correction records (2026-11-12).
    """

    n_total: int
    n_shell: int
    n_base: int
    n_extra: int

    def __post_init__(self):
        values = (self.n_total, self.n_shell, self.n_base, self.n_extra)
        if not all(isinstance(v, (int, np.integer)) for v in values):
            raise TypeError("design counts must be integers")
        if not (self.n_total >= 1 and 0 <= self.n_shell <= self.n_total
                and 1 <= self.n_base <= self.n_total
                and 0 <= self.n_extra <= self.n_total - self.n_base):
            raise ValueError("require N>=1, 0<=M<=N, 1<=B<=N, 0<=E<=N-B")

    @property
    def n_final(self):
        return self.n_base + self.n_extra

    def shell_inclusion(self):
        N, M, B, E = self.n_total, self.n_shell, self.n_base, self.n_extra
        if M == 0:
            return None
        if B + E == N or E >= M:
            return 1.0
        if E == 0:
            return B / N
        if min(B, M) + E <= M:
            return B / N + E / M
        h = np.arange(max(0, B - (N - M)), min(B, M) + 1)
        return float(np.dot(np.minimum(M, h + E), hypergeom.pmf(h, N, M, B)) / M)

    def outside_inclusion(self, ranks=None):
        N, M, B, E = self.n_total, self.n_shell, self.n_base, self.n_extra
        if ranks is None:
            ranks = np.arange(1, N - M + 1, dtype=np.int64)
        else:
            ranks = np.asarray(ranks)
            if (ranks.ndim != 1 or not np.issubdtype(ranks.dtype, np.integer)
                    or np.any(ranks < 1) or np.any(ranks > N - M)):
                raise ValueError("outside ranks must be integers in [1,N-M]")
        if B + E == N:
            return np.ones(len(ranks), dtype=np.float64)
        if E == 0:
            return np.full(len(ranks), B / N, dtype=np.float64)
        threshold = M + ranks - E
        tail = np.zeros(len(ranks), dtype=np.float64)
        tail[threshold <= 0] = 1.0
        active = (threshold > 0) & (threshold <= B)
        if np.any(active):
            # Conditional on a point being absent from the base, let X be
            # the number of base points preceding the E-th remaining point
            # in a fixed ordering of the other N-1 IDs. Then
            # X ~ BetaBinomial(B, E, N-B-E), and adding outside rank r is
            # equivalent to X >= M+r-E. One PMF and reverse cumulative sum
            # evaluate every rank, instead of a separate hypergeometric SF.
            pmf = betabinom.pmf(np.arange(B + 1), B, E, N - B - E)
            pmf /= pmf.sum()
            survival = np.cumsum(pmf[::-1])[::-1]
            tail[active] = np.clip(survival[threshold[active]], 0.0, 1.0)
        return B / N + (1.0 - B / N) * tail

    def inclusion_vector(self):
        prefix = np.full(self.n_shell, self.shell_inclusion(), dtype=np.float64)
        return np.concatenate((prefix, self.outside_inclusion()))

    def compensation_vector(self):
        return self.n_final / (self.n_total * self.inclusion_vector())


def two_stage_probabilities(distance, radius, base_count, extra_count):
    """Return full-population inclusion probabilities in original point order."""
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 1 or not len(distance) or not np.isfinite(distance).all():
        raise ValueError("distance must be a nonempty finite vector")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    shell = np.flatnonzero(distance < radius)
    outside = np.flatnonzero(distance >= radius)
    outside = outside[np.lexsort((outside, distance[outside]))]
    design = TwoStageDesign(len(distance), len(shell), base_count, extra_count)
    probabilities = np.empty(len(distance), dtype=np.float64)
    probabilities[np.concatenate((shell, outside))] = design.inclusion_vector()
    return probabilities


def reference_cell_weights(coords, indices, *, groups=None):
    """Voronoi masses for the uniform empirical measure on the original IDs.

    Each retained ID represents itself, including colocated distinct IDs.
    Every other reference ID is assigned to a nearest retained point. Thus
    mass is conserved, every weight is positive, and full sampling gives one.
    The result is K*m_i, where m_i is a cell's fraction of reference IDs.
    Optional discrete groups restrict assignments to the same group, for
    example to preserve the surface and volume reference masses separately.
    """
    coords = np.asarray(coords, dtype=np.float64)
    indices = np.asarray(indices)
    if (coords.ndim != 2 or not len(coords) or not np.isfinite(coords).all()
            or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or not len(indices) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(coords))):
        raise ValueError("expected finite coordinates and unique valid selected IDs")
    if groups is None:
        _, representatives = cKDTree(coords[indices]).query(coords, k=1)
        representatives[indices] = np.arange(len(indices))
        masses = np.bincount(representatives, minlength=len(indices)).astype(np.float64)
    else:
        groups = np.asarray(groups)
        if groups.shape != (len(coords),):
            raise ValueError("groups must align with the reference coordinates")
        if not (np.issubdtype(groups.dtype, np.integer) or groups.dtype == np.bool_):
            raise ValueError("groups must contain discrete integer or boolean labels")
        masses = np.zeros(len(indices), dtype=np.float64)
        for label in np.unique(groups):
            reference = np.flatnonzero(groups == label)
            selected_rows = np.flatnonzero(groups[indices] == label)
            if not len(selected_rows):
                raise ValueError("every reference group needs a selected representative")
            selected = indices[selected_rows]
            _, representative = cKDTree(coords[selected]).query(coords[reference], k=1)
            representative[np.searchsorted(reference, selected)] = np.arange(len(selected))
            masses[selected_rows] = np.bincount(representative, minlength=len(selected))
    return len(indices) * masses / len(coords)


def temper_shell_fraction(shell_fraction, extra_fraction, alpha):
    """Mixture fraction whose density equals the normalized alpha power.

    For lambda=(1-gamma)+gamma*1_shell/s, the new shell/outside ratio
    is (1+gamma/((1-gamma)*s))**alpha. This changes the proposal, not
    the exponent on the eventual inclusion-probability correction.
    """
    s, gamma = float(shell_fraction), float(extra_fraction)
    if (not all(np.isfinite(x) for x in (s, gamma, alpha))
            or not 0 < s <= 1 or not 0 <= gamma < 1 or not 0 <= alpha <= 1):
        raise ValueError("require 0<s<=1, 0<=gamma<1, and 0<=alpha<=1")
    if alpha == 1:
        return gamma  # Preserve half-integer budget rounding exactly.
    if alpha == 0 or gamma == 0:
        return 0.0
    log_ratio = np.log1p(gamma / ((1-gamma)*s))
    contrast = np.expm1(alpha * log_ratio)
    return float(s * contrast / (1+s*contrast))
