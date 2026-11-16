"""Magnitude-aware mixed densities and exact-size retained-context designs.

The neural input is one scalar acquisition. Geometry supplies a second scalar.
Compensation remains the common lattice nearest-cell mass estimator.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from adasolver.inference.superwing_contract import GRID,NFULL,NCOARSE,rng_for,validate_indices

SAMPLING_KEYS=frozenset(('density_mix','density_mode','geometry_amp','prediction_amp','geometry_power',
    'prediction_power','uniform_mix','density_cap','density_design','coarse_correction','systematic_order',
    'K','placement','beta','power','amp','kappa','score_support'))


def for_budget(policy,budget_index):
    overrides=policy.get('budget_parameters')
    if overrides is None:return policy
    if len(overrides)!=3 or any(set(p)-SAMPLING_KEYS for p in overrides):
        raise ValueError('budget overrides may change only the three sets of sampling parameters')
    result=dict(policy);result.pop('budget_parameters')
    result.update(overrides[budget_index])
    return result


def bounded_mass(weights,total,lower=None,upper=None):
    """Scale positive weights and project onto bounds with an exact total."""
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('positive finite one-dimensional allocation weights required')
    low=np.zeros_like(w) if lower is None else np.broadcast_to(np.asarray(lower,dtype=float),w.shape)
    high=np.ones_like(w) if upper is None else np.broadcast_to(np.asarray(upper,dtype=float),w.shape)
    if np.any(low<0) or np.any(high<low) or not low.sum()-1e-8<=total<=high.sum()+1e-8:
        raise ValueError('infeasible bounded allocation')
    if abs(total-low.sum())<1e-10:return low.copy()
    if abs(total-high.sum())<1e-10:return high.copy()
    lo=0.;hi=max(1.,float(total/w.sum()))
    while np.clip(hi*w,low,high).sum()<total:hi*=2
    for _ in range(48):
        mid=(lo+hi)/2
        if np.clip(mid*w,low,high).sum()<total:lo=mid
        else:hi=mid
    result=np.clip((lo+hi)/2*w,low,high)
    # Remove sub-ulp sum residual on a genuinely free cell.
    free=np.flatnonzero((result>low+1e-8)&(result<high-1e-8))
    if len(free):result[free]+=(total-result.sum())/len(free)
    if np.any(result<low-1e-10) or np.any(result>high+1e-10) or abs(result.sum()-total)>1e-7:
        raise ValueError('bounded allocation failed to converge')
    return result


def integer_quota(expected,total,lower,upper):
    lower=np.asarray(lower,dtype=np.int64);upper=np.asarray(upper,dtype=np.int64)
    quota=np.maximum(lower,np.minimum(np.floor(expected).astype(np.int64),upper))
    remaining=int(total-quota.sum())
    order=np.argsort(-(expected-quota),kind='stable')
    for i in order:
        if remaining and quota[i]<upper[i]:quota[i]+=1;remaining-=1
    if remaining or quota.sum()!=total:raise ValueError('integer allocation did not meet the budget')
    return quota


def mixed_density(score,geometry,policy):
    from adasolver.inference.pi.superwing import physics_score
    from adasolver.inference.superwing_sampling import normalize_score
    p=normalize_score(score) if not policy.get('score_normalized') else np.asarray(score,dtype=float)
    if p.shape!=(NFULL,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1+1e-10):
        raise ValueError('scalar prediction score must be normalized to [0,1]')
    g=normalize_score(physics_score(geometry,'normal_variation'))
    ga=float(policy.get('geometry_amp',8));pa=float(policy.get('prediction_amp',8))
    gp=float(policy.get('geometry_power',.5));pp=float(policy.get('prediction_power',.5))
    mix=float(policy.get('density_mix',.2));uniform=float(policy.get('uniform_mix',0))
    cap=float(policy.get('density_cap',8))
    if min(ga,pa)<0 or min(gp,pp)<=0 or not 0<=mix<=1 or not 0<=uniform<=1 or cap<1:
        raise ValueError('invalid mixed density parameters')
    dg=1+ga*g**gp;dg/=dg.mean()
    dp=1+pa*np.clip(p,0,1)**pp;dp/=dp.mean()
    mode=policy.get('density_mode','blend')
    if mode=='blend':density=(1-mix)*dg+mix*dp
    elif mode=='product':density=dg**(1-mix)*dp**mix
    elif mode=='modulate':density=dg*((1-mix)+mix*dp)
    else:raise ValueError('unknown density combination')
    density/=density.mean()
    density=(1-uniform)*density+uniform
    density=np.maximum(density,density.max()/cap)
    return density/density.mean()


@lru_cache(maxsize=1)
def morton_keys():
    i,j=np.divmod(np.arange(NFULL,dtype=np.uint32),GRID[1])
    key=np.zeros(NFULL,np.uint32)
    for b in range(8):
        key|=((j>>b)&1)<<(2*b)
        key|=((i>>b)&1)<<(2*b+1)
    return key


def systematic_sample(candidates,probabilities,count,rng,order='morton'):
    candidates=np.asarray(candidates)
    pi=np.asarray(probabilities,dtype=float)
    if len(pi)!=len(candidates) or np.any(pi<0) or np.any(pi>1+1e-10) or abs(pi.sum()-count)>1e-7:
        raise ValueError('systematic sampling needs valid inclusion probabilities summing to count')
    if count==0:return np.empty(0,np.int64)
    if order=='random':permutation=rng.permutation(len(candidates))
    elif order=='morton':
        permutation=np.argsort(morton_keys()[candidates],kind='stable')
        if rng.integers(2):permutation=permutation[::-1]
        permutation=np.roll(permutation,int(rng.integers(len(permutation))))
    else:raise ValueError('unknown systematic ordering')
    cumulative=np.cumsum(pi[permutation]);cumulative[-1]=count
    positions=np.searchsorted(cumulative,rng.random()+np.arange(count),side='right')
    result=candidates[permutation[positions]]
    if len(np.unique(result))!=count:raise ValueError('systematic design selected a cell twice')
    return result


def conditional_probabilities(density,count,coarse,correction):
    available=np.ones(NFULL,bool);available[coarse]=False
    candidates=np.flatnonzero(available)
    if correction=='none':weights=density[candidates]
    elif correction=='residual':
        full=bounded_mass(density,count)
        f0=len(coarse)/NFULL
        weights=np.maximum((full[candidates]-f0)/(1-f0),1e-12)
    else:raise ValueError('unknown retained-S0 density correction')
    return candidates,bounded_mass(weights,count-len(coarse))


def select_mixed(score,count,policy,sample_id,seed,budget_index,coarse,geometry):
    from adasolver.inference.superwing_sampling import patches,REFERENCE
    from adasolver.inference.adaptive import comp_weights_for_subset
    from adasolver.inference.adaptive_fast import _fps_fast
    start=time.perf_counter()
    validate_indices(coarse,NCOARSE)
    density=mixed_density(score,geometry,policy)
    rng=rng_for(sample_id,seed,30,budget_index)
    design=policy.get('density_design','proportional_patch')
    info=dict(density_design=design,density_mode=policy.get('density_mode','blend'),
              density_mix=float(policy.get('density_mix',.2)),proposal_density_ratio=float(density.max()/density.min()),
              score_support='full',sampler='mixed_density_v2')
    if design in ('weighted','systematic'):
        correction=policy.get('coarse_correction','none')
        candidates,pi=conditional_probabilities(density,count,coarse,correction)
        nadd=count-len(coarse)
        if design=='systematic':
            added=systematic_sample(candidates,pi,nadd,rng,policy.get('systematic_order','morton'))
        else:
            weights=(density[candidates] if correction=='none' else -np.log1p(-np.clip(pi,0,1-1e-12)))
            added=rng.choice(candidates,nadd,replace=False,p=weights/weights.sum())
        info.update(coarse_correction=correction,conditional_probability_min=float(pi.min()),
                    conditional_probability_max=float(pi.max()),
                    inclusion_probability_role='exact_conditional' if design=='systematic' else 'design_proxy')
    elif design=='proportional_patch':
        k=int(policy.get('K',64))
        if k not in (16,32,64,128,256,512):raise ValueError('unsupported density patch count')
        pid,mass,groups=patches(k)
        capacity=np.bincount(pid,minlength=k)
        base=np.bincount(pid[coarse],minlength=k)
        demand=np.bincount(pid,weights=density,minlength=k)
        lower=np.maximum(base,(capacity>0).astype(np.int64))
        expected=bounded_mass(np.maximum(demand,1e-20),count,lower,capacity)
        quota=integer_quota(expected,count,lower,capacity)
        present=np.zeros(NFULL,bool);present[coarse]=True
        pieces=[]
        placement=policy.get('placement','random')
        for j,members in enumerate(groups):
            candidates=members[~present[members]];nadd=int(quota[j]-base[j])
            if not nadd:continue
            if placement=='fps':piece=_fps_fast(candidates,REFERENCE,coarse,nadd,rng)
            elif placement=='random':piece=rng.choice(candidates,nadd,replace=False)
            elif placement=='weighted':
                w=density[candidates];piece=rng.choice(candidates,nadd,replace=False,p=w/w.sum())
            elif placement=='systematic':
                pi=bounded_mass(density[candidates],nadd)
                piece=systematic_sample(candidates,pi,nadd,rng,policy.get('systematic_order','morton'))
            else:raise ValueError('unknown mixed-density patch placement')
            pieces.append(piece)
        added=np.concatenate(pieces)
        fractions=quota[capacity>0]/capacity[capacity>0]
        info.update(patch_count=k,patch_quotas=quota.tolist(),
                    realized_patch_density_ratio=float(fractions.max()/fractions.min()),
                    coarse_bound_active_patches=int(np.sum(expected<=base+1e-8)))
    else:raise ValueError('unknown mixed-density design')
    selected=np.concatenate((coarse,added))
    validate_indices(selected,count)
    sampling_seconds=time.perf_counter()-start
    start=time.perf_counter()
    beta=float(policy.get('beta',1))
    compensation=comp_weights_for_subset(selected,REFERENCE,NFULL,beta=beta)
    info.update(sampling_seconds=sampling_seconds,compensation_seconds=time.perf_counter()-start)
    return selected,compensation,info
