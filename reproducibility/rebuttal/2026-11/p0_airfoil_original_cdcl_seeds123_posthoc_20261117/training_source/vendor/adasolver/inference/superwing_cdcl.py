"""Small-channel goal-oriented density signals for Cd/Cl.

Only geometry and coarse predictions enter these functions. The evaluation
population filter is deliberately absent from the acquisition interface.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from adasolver.inference.superwing_contract import GRID, NFULL
from adasolver.inference.superwing_cd import local_variation
from adasolver.inference.superwing_sampling import normalize_score
from adasolver.metrics.superwing_cdcl import signed_floor


def estimated_ratio(prediction, geometry, cl_floor=.05, ratio_cap=.5):
    """Stabilize acquisition only; evaluation always uses raw integrated values."""
    x = np.asarray(prediction)
    coef = np.einsum('knc,nc->k', geometry.weights[:2, :, :x.shape[1]], x)
    return float(np.clip(coef[1]/signed_floor(coef[0], cl_floor), -ratio_cap, ratio_cap))


def ratio_weights(geometry, ratio, channels=3):
    return geometry.weights[1, :, :channels] - ratio*geometry.weights[0, :, :channels]


def physics_score(geometry, policy):
    prior = policy['prior']
    wd, wl = geometry.weights[1], geometry.weights[0]
    scale = float(policy.get('lift_scale', .1))
    if prior == 'drag_lift_sensitivity':
        score = np.sqrt(np.sum(wd*wd + scale*scale*wl*wl, axis=1))
    elif prior in ('lift_sensitivity', 'lift_curvature'):
        score = np.linalg.norm(wl, axis=1)
        if prior == 'lift_curvature':
            change = np.gradient(geometry.surface.normals, axis=(0, 1))
            score *= np.sqrt(sum(np.sum(d*d, axis=-1) for d in change)).reshape(-1)
    elif prior == 'fixed_ratio_sensitivity':
        # AoA is an input condition, not a reference coefficient.
        ratio = np.copysign(scale, geometry.aoa if geometry.aoa else 1.)
        score = np.linalg.norm(ratio_weights(geometry, ratio), axis=1)
    else:
        from adasolver.inference.pi.superwing import physics_score as original
        return original(geometry, prior)
    return score


def ratio_score(prediction, geometry, policy):
    signal = policy.get('ratio_signal', 'weighted_lap')
    if signal == 'direct_score':
        return normalize_score(np.maximum(np.asarray(prediction)[:, 0], 0))
    x = np.asarray(prediction, dtype=np.float64)
    ratio = estimated_ratio(x, geometry, float(policy.get('acquisition_cl_floor', .05)),
                            float(policy.get('acquisition_ratio_cap', .5)))
    if policy.get('ratio_channels', 'pressure') == 'pressure':
        x = x[:, :1]
    channels = x.shape[1]
    sigma = float(policy.get('smooth_sigma', 0))
    if sigma:
        x = gaussian_filter(x.reshape(*GRID, channels), (sigma, sigma, 0), mode='nearest').reshape(NFULL, channels)
    wr = ratio_weights(geometry, ratio, channels)
    if signal in ('field_grad','field_lap'):
        raw=np.linalg.norm(local_variation(x,signal.split('_')[1]),axis=1)
    elif signal in ('weighted_grad', 'weighted_lap'):
        raw = np.sum(np.abs(wr)*local_variation(x, signal.split('_')[1]), axis=1)
    elif signal in ('contribution_grad', 'contribution_lap'):
        raw = local_variation(np.sum(wr*x, axis=1), signal.split('_')[1])[:, 0]
    elif signal in ('separate_grad', 'separate_lap'):
        weights = np.abs(geometry.weights[1, :, :channels])+abs(ratio)*np.abs(geometry.weights[0, :, :channels])
        raw = np.sum(weights*local_variation(x, signal.split('_')[1]), axis=1)
    elif signal == 'sensitivity':
        raw = np.linalg.norm(wr, axis=1)
    else:
        raise ValueError('unknown Cd/Cl acquisition signal')
    geometry_exponent=policy.get('ratio_geometry_power',.5) if policy.get('density_revision')=='v2' else policy.get('geometry_power',.5)
    base = normalize_score(np.linalg.norm(wr, axis=1))**float(geometry_exponent)
    score = normalize_score(raw)**float(policy.get('variation_power', 1))
    mix = float(policy.get('prediction_mix', 1))
    if not 0 <= mix <= 1:
        raise ValueError('prediction mix must be within [0,1]')
    return (1-mix)*base+mix*score


def supervision(field, geometry, structured):
    """One normalized scalar from GT, used only in fit/head_val head banks."""
    ratio = float(np.clip(structured[1]/signed_floor(structured[0], .05), -.5, .5))
    cp = gaussian_filter(np.asarray(field[:, 0]).reshape(GRID), 2, mode='nearest').reshape(NFULL, 1)
    raw = np.abs(ratio_weights(geometry, ratio, 1)[:, 0])*local_variation(cp, 'lap')[:, 0]
    return np.log1p(normalize_score(raw))
