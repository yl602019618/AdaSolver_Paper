"""DriverML CPG: scores and predictions share one immutable coarse point order."""
from __future__ import annotations
import numpy as np
from adasolver.inference.driverml_sampling import CoarsePrediction, local_variation, lift_score, select_from_score


def pressure_gradient_score(coords, pressure, k=8):
    return local_variation(coords, pressure, kind='grad_sq', k=k)


def coarse_acquisition(geometry, prediction: CoarsePrediction, *, feature='lap_proxy',
                       sensitivity=False, k=16, wss_weight=0.0, workers=8):
    if prediction.nfull != geometry.point_count:
        raise ValueError('coarse prediction belongs to a different reference surface')
    coords = geometry.pos.numpy()
    coarse = prediction.indices
    coarse_score = local_variation(coords[coarse], prediction.pressure, kind=feature, k=k, workers=workers)
    score = lift_score(coords, coords[coarse], coarse_score, workers=workers)
    if sensitivity:
        score *= np.abs(geometry.normal[:, 0].numpy())*geometry.area.numpy()
    if wss_weight:
        if prediction.wss is None:
            raise ValueError('WSS acquisition requires a prediction on the same 10% coarse set')
        viscous = local_variation(coords[coarse], prediction.wss[:, 0], kind=feature, k=k, workers=workers)
        score += wss_weight*geometry.area.numpy()*lift_score(coords, coords[coarse], viscous, workers=workers)
    return score


def sample_driverml_cpg(geometry, prediction: CoarsePrediction, count, rng, *, amp=8.0,
                         feature='lap_proxy', sensitivity=False, k=16, score=None):
    raw = coarse_acquisition(geometry, prediction, feature=feature, sensitivity=sensitivity, k=k) if score is None else score
    return select_from_score(raw, count, rng, amp=amp, coarse=prediction.indices,
                             metadata={'method': 'cpg', 'feature': feature,
                                       'sensitivity': sensitivity, 'neighborhood': k, 'truth_access': False})
