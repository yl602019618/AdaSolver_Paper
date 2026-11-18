"""Cd-oriented scores and conditional weighted designs for SuperWing v2."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from adasolver.inference.superwing_contract import GRID, NFULL
from adasolver.inference.superwing_sampling import normalize_score
from adasolver.utils.interpolation import index_grid,reconstruct_field


def local_variation(values, kind):
    x=np.asarray(values).reshape(*GRID,-1)
    if kind=='grad':
        di,dj=np.gradient(x,axis=(0,1))
        return np.sqrt(di*di+dj*dj).reshape(NFULL,-1)
    if kind=='lap':
        p=np.pad(x,((1,1),(1,1),(0,0)),mode='edge')
        return np.abs(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*x).reshape(NFULL,-1)
    raise ValueError('unknown variation')


def cd_score(prediction, geometry, policy, *, coarse=None):
    """Return an explicitly normalized density signal, using predictions only."""
    base=normalize_score(np.linalg.norm(geometry.weights[1],axis=-1))**float(policy.get('geometry_power',.5))
    signal=policy.get('cd_signal','weighted_lap')
    channels=prediction.shape[1]
    x=np.asarray(prediction,dtype=np.float64)
    sigma=float(policy.get('smooth_sigma',0))
    if sigma:
        x=gaussian_filter(x.reshape(*GRID,channels),(sigma,sigma,0),mode='nearest').reshape(NFULL,channels)
    w=geometry.weights[1,:,:channels]
    if signal=='geometry':
        return base
    if signal=='reconstruction_disagreement':
        if coarse is None:raise ValueError('reconstruction disagreement needs sampled predictions')
        indices,values=coarse
        reference=index_grid(GRID)
        left,_=reconstruct_field(indices[::2],values[::2],reference)
        right,_=reconstruct_field(indices[1::2],values[1::2],reference)
        difference=np.abs(left-right)
        tolerance=16*np.finfo(np.float64).eps*max(1.,float(np.abs(values).max()))
        difference[difference<tolerance]=0
        raw=np.sum(np.abs(geometry.weights[1,:,:values.shape[1]])*difference,axis=1)
    elif signal in ('learned_variation','learned_weighted_variation','density_influence'):
        raw=np.maximum(x[:,0],0)
        if signal=='learned_weighted_variation':raw=raw*np.linalg.norm(geometry.weights[1],axis=1)
    elif signal in ('weighted_lap','weighted_grad'):
        raw=np.sum(np.abs(w)*local_variation(x,signal.split('_')[1]),axis=1)
    elif signal in ('contribution_lap','contribution_grad'):
        # Retain the sign until taking spatial variation of the net contribution.
        contribution=np.sum(w*x,axis=1)
        raw=local_variation(contribution,signal.split('_')[1])[:,0]
    elif signal=='contribution_magnitude':
        raw=np.abs(np.sum(w*x,axis=1))
    else:
        raise ValueError('unknown Cd signal')
    variation=normalize_score(raw)**float(policy.get('variation_power',.5))
    mix=float(policy.get('prediction_mix',.5))
    if not 0<=mix<=1:raise ValueError('prediction mix must be within [0,1]')
    if policy.get('combine','blend')=='blend':
        return (1-mix)*base+mix*variation
    if policy['combine']=='modulate':
        return base*((1-mix)+mix*variation)
    raise ValueError('unknown geometry/prediction combination')


def inclusion_mass(weights, count):
    """Exponential-race approximation to PPSWOR inclusion probabilities."""
    weights=np.asarray(weights,dtype=np.float64)
    if count<0 or count>len(weights) or np.any(weights<=0):raise ValueError('invalid remaining population')
    if count==len(weights):return np.ones_like(weights)
    if count==0:return np.zeros_like(weights)
    lo,hi=0.,1./weights.min()
    while (-np.expm1(-hi*weights)).sum()<count:hi*=2
    for _ in range(42):
        mid=(lo+hi)/2
        if (-np.expm1(-mid*weights)).sum()<count:lo=mid
        else:hi=mid
    return -np.expm1(-((lo+hi)/2)*weights)


def stratified_additions(candidates,weights,count,strata,rng):
    """Fixed stratum quotas, capacity bounded, followed by weighted sampling."""
    groups=np.unique(strata)
    mass=inclusion_mass(weights,count)
    expected=np.array([mass[strata==g].sum() for g in groups])
    capacity=np.array([np.sum(strata==g) for g in groups])
    quota=np.minimum(np.floor(expected).astype(int),capacity)
    remaining=count-int(quota.sum())
    for i in np.argsort(-(expected-quota),kind='stable'):
        if remaining and quota[i]<capacity[i]:quota[i]+=1;remaining-=1
    if remaining or quota.sum()!=count:raise ValueError('stratum allocation failed to meet exact budget')
    selected=[]
    for g,n in zip(groups,quota):
        mask=strata==g
        if n:
            w=weights[mask];selected.append(rng.choice(candidates[mask],int(n),replace=False,p=w/w.sum()))
    return np.concatenate(selected),dict(stratum_count=len(groups),stratum_quotas=quota.tolist())
