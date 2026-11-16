"""Exact-budget sampling and common lattice compensation for SuperWing."""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from adasolver.inference.adaptive import build_patches, comp_weights_for_subset
from adasolver.inference.adaptive_fast import build_patch_cache, refine_once
from adasolver.inference.superwing_contract import GRID, NFULL, NCOARSE, rng_for, validate_indices
from adasolver.utils.interpolation import index_grid

REFERENCE=index_grid(GRID)


def normalize_score(score):
    score=np.asarray(score,dtype=np.float64)
    if score.shape!=(NFULL,) or not np.isfinite(score).all() or np.any(score<0):
        raise ValueError('SuperWing acquisition score must be finite, nonnegative, full-reference length')
    low,high=np.quantile(score,[.01,.99])
    if high<=low+1e-20:
        return np.zeros_like(score)
    return np.clip((score-low)/(high-low),0,1)


@lru_cache(maxsize=8)
def patches(k):
    ids,mass=build_patches(REFERENCE,k,seed=0)
    return ids,mass,build_patch_cache(REFERENCE,ids,k)


def select(score,count,policy,sample_id,seed,budget_index,coarse=None,*,geometry=None):
    if policy.get('budget_parameters') is not None:
        from adasolver.inference.superwing_density import for_budget
        policy=for_budget(policy,budget_index)
    if policy.get('density_revision')=='v2':
        from adasolver.inference.superwing_density import select_mixed
        if coarse is None or geometry is None:raise ValueError('mixed density needs geometry and retained S0')
        return select_mixed(score,count,policy,sample_id,seed,budget_index,coarse,geometry)
    start=time.perf_counter()
    score=np.asarray(score,dtype=np.float64)
    power=float(policy.get('power',1))
    beta=float(policy.get('beta',1))
    amp=float(policy.get('amp',8))
    if not (power>0 and amp>=0 and 0<=beta<=1):
        raise ValueError('invalid density power/amplitude/compensation beta')
    if policy.get('score_normalized',False):
        if score.shape!=(NFULL,) or not np.isfinite(score).all() or np.any(score<0) or np.any(score>1+1e-12):
            raise ValueError('normalized score must be finite in [0,1]')
        normalized=np.clip(score,0,1)**power
    else:
        normalized=normalize_score(score)**power
    rng=rng_for(sample_id,seed,30,budget_index)
    info={}
    sampler=policy.get('sampler','patch')
    if sampler in ('weighted_remaining','stratified_weighted'):
        if coarse is None:raise ValueError('conditional weighted design requires an explicit retained S0')
        validate_indices(coarse,NCOARSE)
        available=np.ones(NFULL,dtype=bool);available[coarse]=False
        candidates=np.flatnonzero(available)
        weights=1+amp*normalized[candidates]
        nadd=count-len(coarse)
        if sampler=='weighted_remaining':
            added=rng.choice(candidates,nadd,replace=False,p=weights/weights.sum())
        else:
            if geometry is None:raise ValueError('signed force stratification requires geometry')
            from adasolver.inference.superwing_cd import stratified_additions
            bands=int(policy.get('span_bands',8))
            if bands not in (4,8,16):raise ValueError('unsupported span strata count')
            span=(candidates//GRID[1])*bands//GRID[0]
            sign=(geometry.weights[1,candidates,0]>=0).astype(int)
            strata=span*2+sign
            added,details=stratified_additions(candidates,weights,nadd,strata,rng)
            info.update(details)
        selected=np.concatenate([coarse,added])
        validate_indices(selected,count)
        sampling_seconds=time.perf_counter()-start
        start=time.perf_counter()
        compensation=comp_weights_for_subset(selected,REFERENCE,NFULL,beta=beta)
        return selected,compensation,dict(score_support='full',sampler=sampler,
            sampling_seconds=sampling_seconds,compensation_seconds=time.perf_counter()-start,**info)
    if sampler!='patch':raise ValueError('unknown SuperWing sampler')
    if coarse is None:
        weights=1+amp*normalized
        selected=rng.choice(NFULL,count,replace=False,p=weights/weights.sum())
        info.update(score_support='full',requested_density_ratio=float(weights.max()/weights.min()))
    else:
        validate_indices(coarse,NCOARSE)
        if not np.any(normalized):
            # With no acquisition information, S0 plus uniform additions has the
            # same unconditional uniform inclusion probabilities as the baseline.
            available=np.ones(NFULL,dtype=bool);available[coarse]=False
            added=rng.choice(np.flatnonzero(available),count-NCOARSE,replace=False)
            selected=np.concatenate([coarse,added])
            validate_indices(selected,count)
            return selected,np.ones(count,dtype=np.float32),dict(score_support=policy.get('score_support','coarse'),
                score_uniform_fallback=True,sampling_seconds=time.perf_counter()-start,compensation_seconds=0.)
        k=int(policy.get('K',64))
        kappa=float(policy.get('kappa',4))
        placement=policy.get('placement','random')
        support=policy.get('score_support','coarse')
        if k not in (32,64,128) or kappa<1 or placement not in ('random','fps') or support not in ('coarse','full'):
            raise ValueError('unsupported SuperWing patch strategy')
        pid,mass,pcache=patches(k)
        # A full-support policy explicitly aggregates all candidate scores.
        if support=='full':
            totals=np.bincount(pid,weights=normalized,minlength=k)
            normalized=(totals/np.bincount(pid,minlength=k))[pid]
        selected,allocation=refine_once(coarse,normalized,REFERENCE,pid,mass,count,kappa=kappa,
                                        seed=int(rng.integers(2**32-1)),patch_cache=pcache,placement=placement)
        info.update(score_support=support,patch_count=k,**allocation)
    validate_indices(selected,count)
    sampling_seconds=time.perf_counter()-start
    start=time.perf_counter()
    compensation=comp_weights_for_subset(selected,REFERENCE,NFULL,beta=beta)
    if not np.isfinite(compensation).all() or np.any(compensation<=0):
        raise ValueError('invalid discrete density compensation')
    info.update(sampling_seconds=sampling_seconds,compensation_seconds=time.perf_counter()-start)
    return selected,compensation,info
