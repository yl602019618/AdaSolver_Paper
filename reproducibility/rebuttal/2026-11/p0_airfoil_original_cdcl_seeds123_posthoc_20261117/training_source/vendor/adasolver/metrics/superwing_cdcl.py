"""Cd/Cl evaluation from the canonical reconstructed-field coefficients."""
from __future__ import annotations

import numpy as np

PROTOCOL = 'superwing_lattice_cdcl_budget10_v1'
EPSILON = 1e-12


def signed_floor(value, floor=EPSILON):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value < 0, -1., 1.) * np.maximum(np.abs(value), floor)


def ratio_metrics(cd_pred, cl_pred, cd_true, cl_true):
    values = np.asarray([cd_pred, cl_pred, cd_true, cl_true], dtype=np.float64)
    if not np.isfinite(values).all() or abs(cl_true) < EPSILON or abs(cd_true) < EPSILON:
        raise ValueError('Cd/Cl requires finite coefficients and nonzero reference coefficients')
    pred = float(cd_pred / signed_floor(cl_pred))
    true = float(cd_true / cl_true)
    return dict(CdCl_pred=pred, CdCl_gt_structured=true,
                cdcl_relative_error=abs(pred-true)/abs(true),
                cdcl_absolute_error=abs(pred-true),
                cdcl_prediction_denominator_floored=bool(abs(cl_pred) < EPSILON))


def from_record(record):
    return ratio_metrics(record['Cd_pred'], record['Cl_pred'],
                         record['Cd_gt_structured'], record['Cl_gt_structured'])
