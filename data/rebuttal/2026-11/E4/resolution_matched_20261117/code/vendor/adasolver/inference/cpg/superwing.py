"""Prediction-guided SuperWing acquisition from exactly 10% of the cells."""
from __future__ import annotations

import time

import numpy as np

from adasolver.inference.superwing_contract import GRID,NCOARSE,validate_indices
from adasolver.utils.interpolation import index_grid,reconstruct_field

REFERENCE=index_grid(GRID)


def variation(field,kind):
    values=np.asarray(field).reshape(*GRID,-1)
    if kind=='grad':
        di,dj=np.gradient(values,axis=(0,1))
        return np.sqrt(di*di+dj*dj)
    if kind=='lap':
        padded=np.pad(values,((1,1),(1,1),(0,0)),mode='edge')
        return np.abs(padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:]-4*values)
    raise ValueError('variation must be grad or lap')


def score_from_prediction(prediction,geometry,policy):
    feature=policy['feature']
    if feature not in ('cp_grad','cp_lap','joint_grad','joint_lap'):
        raise ValueError('unsupported SuperWing CPG feature')
    joint=feature.startswith('joint')
    if joint and prediction.shape[1]!=3:
        raise ValueError('joint variation requires all three coarse predicted channels')
    field=prediction if joint else prediction[:,:1]
    change=variation(field,feature.rsplit('_',1)[-1]).reshape(-1,field.shape[1])
    if policy['target']=='cd':
        weights=np.abs(geometry.weights[1] if joint else geometry.weights[1,:,:1])
        score=np.sum(weights*change,axis=1)
    else:
        score=np.linalg.norm(change,axis=1)
    if not np.isfinite(score).all():
        raise ValueError('non-finite acquisition score')
    return score


def acquire_score(runtime,geometry,s0,policy,acquisition,heads):
    validate_indices(s0,NCOARSE)
    method=policy['method']
    cache=None
    if method=='cpg':
        influence=policy.get('cd_signal')=='density_influence'
        key='cpg_influence' if influence else 'cpg'
        hit=key in acquisition
        if not hit:
            if influence:
                from adasolver.inference.superwing_influence import density_influence
                values,coarse_seconds=density_influence(runtime,geometry,s0)
                prediction=values[:,None]
            else:prediction,coarse_seconds=runtime.predict(geometry.features(s0))
            started=time.perf_counter()
            dense,fallback=reconstruct_field(s0,prediction,REFERENCE)
            acquisition[key]=dict(dense=dense,coarse_or_block_seconds=coarse_seconds,head_seconds=0.,
                                  reconstruction_seconds=time.perf_counter()-started,fallback_count=fallback,sampled=prediction)
    elif method=='ca':
        from adasolver.inference.ca.superwing import acquire_scalar
        key='ca:'+policy['head_sha256']
        hit=key in acquisition
        if not hit:
            prediction,cache,block_seconds,head_seconds=acquire_scalar(runtime,geometry,s0,policy,heads,acquisition)
            started=time.perf_counter()
            dense,fallback=reconstruct_field(s0,prediction[:,None],REFERENCE)
            acquisition[key]=dict(dense=dense,cache=cache,coarse_or_block_seconds=block_seconds,head_seconds=head_seconds,
                                  reconstruction_seconds=time.perf_counter()-started,fallback_count=fallback,
                                  head_target=heads[policy['head_sha256']][1]['target'],sampled=prediction[:,None])
        if acquisition[key]['head_target']!=policy.get('head_target','Cp_GT'):
            raise ValueError('cached acquisition belongs to another scalar target')
        cache=acquisition[key]['cache']
    else:
        raise ValueError('acquisition requires CPG or CA')
    source=acquisition[key]
    started=time.perf_counter()
    if policy['target']=='cdcl':
        from adasolver.inference.superwing_cdcl import ratio_score
        score=ratio_score(source['dense'],geometry,policy)
    elif policy.get('cd_revision')=='v2':
        from adasolver.inference.superwing_cd import cd_score
        prediction=source['dense'][:,:1] if policy.get('cd_channels')=='pressure' else source['dense']
        sampled=source['sampled'][:,:1] if policy.get('cd_channels')=='pressure' else source['sampled']
        score=cd_score(prediction,geometry,policy,coarse=(s0,sampled))
    else:
        score=(np.maximum(source['dense'][:,0],0) if policy['feature']=='direct_score'
               else score_from_prediction(source['dense'],geometry,policy))
    acquisition['_last_cache_hit']=hit
    acquisition['_last_fallback_count']=source['fallback_count']
    timing=dict(coarse_or_block_seconds=source['coarse_or_block_seconds'],head_seconds=source['head_seconds'],
                score_seconds=time.perf_counter()-started+source['reconstruction_seconds'])
    return score,cache,timing
