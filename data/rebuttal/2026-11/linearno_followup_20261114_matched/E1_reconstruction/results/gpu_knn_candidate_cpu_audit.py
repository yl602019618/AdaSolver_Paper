"""Independent CPU oracle audit of the proposed exhaustive GPU kNN mapper.

This deliberately does not initialize CUDA or change the candidate implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.spatial import cKDTree

OUT=Path(__file__).resolve().parent
CODE=OUT.parent/'code'
SOURCE=CODE/'gpu_knn_candidate.py'
sys.path.insert(0,str(CODE))
import gpu_knn_candidate as candidate
from reconstruct_api import native_idw


def oracle(source,values,query,k):
    """Direct selected-tree rule, matching the frozen native IDW convention."""
    source=np.asarray(source,dtype=np.float64)
    query=np.asarray(query,dtype=np.float64)
    values=np.asarray(values)
    if values.ndim==1:values=values[:,None]
    effective=min(k,len(source))
    distance,indices=cKDTree(source).query(query,k=effective)
    if effective==1:distance=distance[:,None];indices=indices[:,None]
    if effective==1:return values[indices[:,0]].astype(np.float64),distance,indices
    weights=1/(distance+1e-12)
    gathered=values[indices]
    result=np.sum(gathered*weights[:,:,None],axis=1)/weights.sum(axis=1)[:,None]
    for row in range(len(query)):
        hit=np.flatnonzero(distance[row]<=1e-12)
        if len(hit):result[row]=gathered[row,hit[0]]
    return result,distance,indices


def valid_case(name,source,query,values,k=4,max_pairs=8388608,supply_cpu=True):
    source=np.asarray(source,dtype=np.float64);query=np.asarray(query,dtype=np.float64)
    values=np.asarray(values)
    x=torch.from_numpy(source);q=torch.from_numpy(query);v=torch.from_numpy(values)
    assert x.device.type==q.device.type==v.device.type=='cpu'
    calls=[];original=candidate.torch.cdist
    def observed(a,b,*args,**kwargs):
        calls.append(dict(query=a.detach().numpy().copy(),source_rows=len(b),same_source_object=b is x,
                          mode=kwargs.get('compute_mode')))
        return original(a,b,*args,**kwargs)
    candidate.torch.cdist=observed
    try:
        mapping=candidate.build_exact_gpu_knn(x,q,k=k,max_pairs=max_pairs,
            source_cpu=source if supply_cpu else None,query_cpu=query if supply_cpu else None)
    finally:candidate.torch.cdist=original
    result=mapping.apply(v).detach().numpy()
    expected,distance,indices=oracle(source,values,query,k)
    old=native_idw(source,values,query,k=k)
    np.testing.assert_allclose(expected,old,rtol=0,atol=0)
    assert result.shape==expected.shape
    np.testing.assert_allclose(result,expected,rtol=5e-13,atol=2e-12)
    np.testing.assert_array_equal(mapping.indices.numpy(),indices)
    np.testing.assert_allclose(mapping.distances.numpy(),distance,rtol=5e-14,atol=1e-14)
    np.testing.assert_array_equal(np.concatenate([c['query'] for c in calls]),query)
    assert all(c['source_rows']==len(source) and c['same_source_object'] and c['mode']=='donot_use_mm_for_euclid_dist' for c in calls)
    assert max(len(c['query'])*c['source_rows'] for c in calls)<=max_pairs
    diff=np.abs(result-expected)
    return dict(name=name,source_points=len(source),query_points=len(query),dimensions=source.shape[1],
                value_shape=list(values.shape),output_shape=list(result.shape),output_dtype=str(result.dtype),
                effective_k=mapping.effective_k,cdist_calls=len(calls),all_source_points_visited_per_block=True,
                each_query_visited_once=True,max_distance_pairs=mapping.metadata['max_distance_pairs'],
                requested_max_pairs=max_pairs,tie_fallback_rows=mapping.metadata['legacy_tie_rows'],
                max_abs_field_difference=float(diff.max()),
                relative_field_difference=float(np.linalg.norm(result-expected)/(np.linalg.norm(expected)+1e-30)),
                default_field_tolerance_violations=int(np.sum(diff>1e-6+1e-5*np.abs(expected))),
                native_indices_identical=True,passed=True)


def probe_failure_inputs():
    issues=[];rng=np.random.default_rng(782)
    source=rng.normal(size=(11,3));query=rng.normal(size=(5,3));values=rng.normal(size=(11,2))
    x=torch.from_numpy(source);q=torch.from_numpy(query)
    # A small requested budget exposes the missing source-axis bound without
    # allocating a large tensor or creating an artificial OOM.
    for limit in [10,0]:
        m=candidate.build_exact_gpu_knn(x,q,k=4,max_pairs=limit)
        issues.append(dict(id='PAIR_BOUND' if limit else 'ZERO_PAIR_BUDGET',
                           requested_max_pairs=limit,actual_max_pairs=m.metadata['max_distance_pairs'],
                           query_block=m.metadata['query_block'],accepted=True,
                           bound_violated=m.metadata['max_distance_pairs']>limit))
    # Side copies must not be allowed to override the actual tensor geometry.
    source=np.array([[0.,0.],[0.,0.],[1.,0.],[0.,1.],[2.,0.]])
    query=np.array([[0.,0.]])
    values=np.array([[1.],[9.],[20.],[30.],[40.]])
    expected,_,_=oracle(source,values,query,4)
    for name,source_cpu,query_cpu in [
        ('REORDERED_SOURCE_CPU',source[::-1].copy(),query),
        ('STALE_QUERY_CPU',source,np.array([[2.,0.]])),
    ]:
        m=candidate.build_exact_gpu_knn(torch.from_numpy(source),torch.from_numpy(query),k=4,
                                         source_cpu=source_cpu,query_cpu=query_cpu)
        actual=m.apply(torch.from_numpy(values)).numpy()
        issues.append(dict(id=name,expected=expected.tolist(),actual=actual.tolist(),
                           max_abs_difference=float(np.max(np.abs(actual-expected))),
                           fallback_rows=m.metadata['legacy_tie_rows'],silently_accepted=True))
    for k in [0,-1]:
        result=dict(id='INVALID_K',k=k)
        try:
            m=candidate.build_exact_gpu_knn(torch.from_numpy(source),torch.from_numpy(query),k=k)
            result['builder_returned_map']=True
            try:m.apply(torch.from_numpy(values));result['apply_raised']=None
            except Exception as exc:result['apply_raised']=type(exc).__name__+': '+str(exc)
        except Exception as exc:
            result['builder_returned_map']=False;result['builder_raised']=type(exc).__name__+': '+str(exc)
        issues.append(result)
    # Native cKDTree rejects nonfinite geometry; check that GPU-style topk
    # does not silently discard a NaN source coordinate.
    source=rng.normal(size=(9,3));source[0,0]=np.nan;query=rng.normal(size=(2,3))
    result=dict(id='NONFINITE_SOURCE')
    try:cKDTree(source);result['native_raised']=None
    except Exception as exc:result['native_raised']=type(exc).__name__
    try:
        m=candidate.build_exact_gpu_knn(torch.from_numpy(source),torch.from_numpy(query),k=4)
        result['candidate_returned_map']=True
        result['returned_distances_finite']=bool(torch.isfinite(m.distances).all())
    except Exception as exc:result['candidate_returned_map']=False;result['candidate_raised']=type(exc).__name__
    issues.append(result)
    return issues


def main():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    rng=np.random.default_rng(20261114);cases=[]
    cases.append(valid_case('random_2d_scalar_blocked',rng.normal(size=(37,2)),rng.normal(size=(23,2)),rng.normal(size=37),max_pairs=73))
    cases.append(valid_case('random_3d_four_channels_blocked',rng.normal(size=(41,3)),rng.normal(size=(17,3)),rng.normal(size=(41,4)).astype(np.float32),max_pairs=89))
    source=np.array([[0.,0.],[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,0.]])
    query=np.array([[0.,0.],[.5,0.],[.5,.5],[1.,1.]])
    values=np.arange(18,dtype=np.float64).reshape(6,3)
    cases.append(valid_case('duplicates_exact_hits',source,query,values,supply_cpu=False,max_pairs=17))
    order=np.array([5,3,1,4,0,2])
    cases.append(valid_case('permuted_duplicate_source_order',source[order],query,values[order],max_pairs=17))
    source=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
    cases.append(valid_case('cutoff_tie_group',source,np.array([[0.,0.],[.5,.5]]),np.arange(8,dtype=float),k=3,max_pairs=8))
    cases.append(valid_case('k1_tie_vector',source,np.array([[0.,0.],[.3,.7]]),rng.normal(size=(8,4)).astype(np.float32),k=1,max_pairs=16))
    cases.append(valid_case('source_smaller_than_k',np.array([[0.,0.,0.],[2.,0.,0.]]),np.array([[0.,0.,0.],[1.,0.,0.],[3.,0.,0.]]),np.array([[2.,4.],[3.,8.]],dtype=np.float32),k=4,max_pairs=4))
    cases.append(valid_case('single_source_scalar',np.array([[2.,3.]]),np.array([[2.,3.],[8.,9.]]),np.array([17.],dtype=np.float32),k=4,max_pairs=2))
    source=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    eps=1e-12;query=np.array([[eps*(1-1e-9),0.],[eps,0.],[eps*(1+1e-9),0.]])
    cases.append(valid_case('epsilon_boundary',source,query,np.array([2.,3.,4.,5.]),max_pairs=8))
    base=1e12
    source=np.array([[base+.001,base],[base+.003,base],[base,base+.004],[base+.005,base+.007],[base-.003,base-.001]])
    query=np.array([[base+.0015,base+.0005],[base+.004,base+.006]])
    cases.append(valid_case('large_offset_close_points',source,query,np.array([1.,3.,5.,7.,9.]),max_pairs=10))
    xx,yy=np.meshgrid(np.arange(13.),np.arange(13.),indexing='ij');grid=np.stack((xx.ravel(),yy.ravel()),axis=1)
    ids=np.sort(rng.choice(len(grid),67,replace=False))
    cases.append(valid_case('integer_grid_tie_frequency',grid[ids],grid,rng.normal(size=(67,2)),max_pairs=256))
    source=rng.normal(size=(23,6))[:,::2];query=rng.normal(size=(13,6))[:,::2]
    assert not source.flags.c_contiguous
    cases.append(valid_case('noncontiguous_coordinates',source,query,rng.normal(size=(23,3)),max_pairs=52))
    probes=probe_failure_inputs()
    after=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if after!=before:raise RuntimeError('Candidate source changed during audit; do not certify mixed source versions')
    bounds=[]
    for m,n in [(2709,7225),(8453,11271),(24140,32186),(32186,32186)]:
        block=max(1,min(512,8388608//m))
        bounds.append(dict(source_points=m,query_points=n,query_block=block,distance_tile_pairs=min(n,block)*m,
                           distance_tile_mib=min(n,block)*m*8/(1024**2)))
    result=dict(audit_status='valid_domain_cpu_checks_passed_with_candidate_issues',
                checked_at_utc=datetime.now(timezone.utc).isoformat(),
                test_device='cpu',cuda_validation_performed=False,performance_measurement_performed=False,
                source=str(SOURCE),source_sha256=before,source_unchanged=True,
                environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
                valid_cases=cases,valid_case_count=len(cases),
                maximum_absolute_field_difference=max(c['max_abs_field_difference'] for c in cases),
                default_field_tolerance_violations=sum(c['default_field_tolerance_violations'] for c in cases),
                issue_probes=probes,configured_dimension_tile_bounds=bounds,
                approval='CPU correctness evidence only. Keep candidate outside accepted GPU protocol until issues and CUDA/end-to-end validation are resolved.')
    (OUT/'gpu_knn_candidate_cpu_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['valid_cases','issue_probes']},indent=2))
    print('issue_probes',json.dumps(probes,indent=2))


if __name__=='__main__':main()
