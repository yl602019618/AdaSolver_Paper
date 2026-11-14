"""Separate native CPU stage probes; never used by direct request timing.

SciPy linear interpolation fuses simplex search, barycentric weights and
application in compiled code. That fused interval is reported honestly; the
accepted explicit-map implementation separates those operations.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from reconstruct_api import native_reconstruct,build_plan
from validate_contract import difference


def native_stages(dataset,coords,ids,values,mask=None):
    rows=[]
    def stage(component,fn):
        started=time.perf_counter();value=fn();rows.append(dict(component=component,ms=(time.perf_counter()-started)*1000));return value
    def idw(source,value,query,prefix):
        tree=stage(prefix+'_build_index_ms',lambda:cKDTree(source));k=min(4,len(source))
        distance,neighbor=stage(prefix+'_neighbor_query_ms',lambda:tree.query(query,k=k))
        if k==1:return stage(prefix+'_gather_apply_ms',lambda:value[neighbor])
        weights=stage(prefix+'_weights_ms',lambda:1/(distance+1e-12))
        gathered=stage(prefix+'_gather_ms',lambda:value[neighbor])
        out=stage(prefix+'_weighted_application_ms',lambda:(gathered*weights[...,None]).sum(axis=1)/weights.sum(axis=1)[:,None])
        def exact():
            exact=distance<=1e-12;active=np.flatnonzero(exact.any(axis=1));out[active]=gathered[active,np.argmax(exact[active],axis=1)]
        stage(prefix+'_zero_distance_override_ms',exact);return out
    if dataset=='shapenet_car':
        source,active,surface=stage('coordinate_index_prepare_ms',lambda:(coords[ids],mask[ids],np.flatnonzero(mask)))
        velocity=idw(source,values[:,:3],coords,'velocity')
        pressure=idw(source[active],values[active,3:4],coords[surface],'pressure')
        def output():
            out=np.zeros((len(coords),4),np.float64);out[:,:3]=velocity;out[surface,3]=pressure[:,0];return out
        result=stage('output_container_scatter_ms',output)
    else:
        source=stage('coordinate_index_prepare_ms',lambda:coords[ids])
        interpolator=stage('triangulation_interpolator_build_ms',lambda:LinearNDInterpolator(source,values,fill_value=np.nan))
        result=stage('fused_simplex_query_weights_apply_ms',lambda:interpolator(coords))
        missing=stage('fallback_mask_ms',lambda:np.isnan(result).any(axis=1))
        if missing.any():
            nearest=stage('fallback_tree_build_ms',lambda:NearestNDInterpolator(source,values))
            def fallback():result[missing]=nearest(coords[missing])
            stage('fallback_query_apply_ms',fallback)
        stage('output_finite_validation_ms',lambda:np.isfinite(result).all() or (_ for _ in ()).throw(ValueError('nonfinite')))
    return result,rows


def profile_fixture(dataset,coords,ids,values,mask=None,repeats=5):
    reference=native_reconstruct(dataset,ids,values,coords,mask);probe,stages=native_stages(dataset,coords,ids,values,mask)
    assert difference(reference,probe)['violations']==0
    rows=[]
    for repeat in range(repeats):
        old,parts=native_stages(dataset,coords,ids,values,mask)
        rows.extend(dict(implementation='native_stage_probe',profile_repeat=repeat,**r) for r in parts)
        plan=build_plan(dataset,coords,ids,surface_mask=mask,channels=values.shape[1],workers=4,profile=True)
        started=time.perf_counter();new=plan.apply_numpy(values);apply=(time.perf_counter()-started)*1000
        assert difference(reference,new)['violations']==0
        rows.extend(dict(implementation='accepted_cpu_plan',profile_repeat=repeat,map_index=i,component=k,ms=v)
                    for i,m in enumerate(plan.timings['maps']) for k,v in m.items() if k.endswith('_ms'))
        rows.extend([dict(implementation='accepted_cpu_plan',profile_repeat=repeat,component='coordinate_index_prepare_ms',ms=plan.timings['coordinate_index_prepare_ms']),
                     dict(implementation='accepted_cpu_plan',profile_repeat=repeat,component='cpu_plan_total_ms',ms=plan.timings['cpu_plan_total_ms']),
                     dict(implementation='accepted_cpu_plan',profile_repeat=repeat,component='apply_output_scatter_ms',ms=apply)])
    return rows,difference(reference,probe)
