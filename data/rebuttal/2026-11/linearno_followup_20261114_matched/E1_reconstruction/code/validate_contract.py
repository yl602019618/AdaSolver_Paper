"""Independent old/new reconstruction checks before test-set benchmarking."""
from __future__ import annotations
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
CURRENT=HERE.parents[1]
sys.path.insert(0,str(CURRENT/'AdaSolver_official'))
from adasolver.metrics.shapenet_car import reconstruct_car_fields
from adasolver.utils.interpolation import reconstruct_field,index_grid
from reconstruct_api import build_plan,native_reconstruct,PlanCache


def difference(old,new):
    error=np.abs(new-old);tolerance=1e-6+1e-5*np.abs(old)
    return dict(max_abs=float(error.max()),relative_norm=float(np.linalg.norm(new-old)/max(np.linalg.norm(old),1e-30)),
                q50=float(np.quantile(error,.5)),q95=float(np.quantile(error,.95)),q99=float(np.quantile(error,.99)),
                violations=int(np.sum(error>tolerance)),elements=error.size)


def main():
    torch.set_num_threads(4);rng=np.random.default_rng(20261114);rows=[]
    cases=[]
    coords=rng.normal(size=(96,3));coords[3]=coords[2];coords[44]=coords[45]
    mask=np.arange(96)<24;selected=np.unique(np.r_[np.arange(12),np.arange(36,60)])
    cases.append(('car_duplicates','shapenet_car',coords,mask,selected,rng.normal(size=(len(selected),4)).astype(np.float32)))
    coords=np.vstack((np.zeros((1,3)),np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])))
    mask=np.ones(7,dtype=bool);selected=np.arange(1,7)
    cases.append(('car_equidistant','shapenet_car',coords,mask,selected,np.arange(24,dtype=np.float32).reshape(6,4)))
    coords=rng.normal(size=(48,3));coords[1]=coords[0]+[2e-9,0,0];mask=np.arange(48)<16;selected=np.arange(0,48,2)
    cases.append(('car_float64_geometry','shapenet_car',coords,mask,selected,(100*rng.normal(size=(len(selected),4))).astype(np.float32)))
    for channels in [1,4]:
        coords=index_grid((17,23));selected=np.array([3*23+3,3*23+19,13*23+3,13*23+19,8*23+11])
        cases.append((f'grid_sparse_boundary_c{channels}','airfoil',coords,None,selected,rng.normal(size=(len(selected),channels)).astype(np.float32)))
    coords=index_grid((17,17))/16;coords=coords.astype(np.float32).astype(np.float64);selected=np.sort(rng.choice(len(coords),105,False))
    cases.append(('grid_quantized_shared_edges','darcy',coords,None,selected,rng.normal(size=(len(selected),1)).astype(np.float32)))
    for name,dataset,coords,mask,selected,values in cases:
        if dataset=='shapenet_car':
            velocity,pressure,surface=reconstruct_car_fields(selected,values,coords,mask)
            old=np.zeros((len(coords),4));old[:,:3]=velocity;old[surface,3]=pressure
        else:old=reconstruct_field(selected,values,coords)[0]
        native=native_reconstruct(dataset,selected,values,coords,mask)
        np.testing.assert_array_equal(native,old)
        plan=build_plan(dataset,coords,selected,surface_mask=mask,channels=values.shape[1],workers=4,device='cuda:0')
        cpu=plan.apply_numpy(values);gpu=plan.apply_tensor(torch.tensor(values,device='cuda:0')).cpu().numpy()
        checks={label:difference(old,prediction) for label,prediction in [('cpu_sparse',cpu),('gpu_sparse',gpu)]}
        assert all(c['violations']==0 for c in checks.values()),(name,checks)
        rows.append(dict(case=name,dataset=dataset,source_points=len(selected),query_points=len(coords),channels=values.shape[1],
                         reference_dtype=str(coords.dtype),old_dtype=str(old.dtype),new_dtype=str(gpu.dtype),checks=checks,
                         plan_timings=plan.timings,device_map_bytes=plan.device_bytes()))
    cache=PlanCache();coords=index_grid((5,5));ids=np.array([0,4,20,24]);v=np.arange(4.)[:,None]
    first,hit=cache.get('airfoil',coords,ids,geometry_identity='physical_geometry_A');assert not hit
    same,hit=cache.get('airfoil',coords,ids,geometry_identity='physical_geometry_A');assert hit and same is first
    other,hit=cache.get('airfoil',coords,ids,geometry_identity='physical_geometry_B');assert not hit
    permuted,hit=cache.get('airfoil',coords,ids[::-1],geometry_identity='physical_geometry_A');assert not hit
    np.testing.assert_allclose(first.apply_numpy(v*2),first.apply_numpy(v)*2)
    result=dict(status='passed',scope='mathematical-rule contracts; real validation cases are a separate gate',
                tolerance='abs(new-old)<=1e-6+1e-5*abs(old)',cases=rows,
                cache_check=dict(geometry_identity_invalidates=True,selected_order_invalidates=True,predictions_not_cached=True),
                torch=torch.__version__,numpy=np.__version__)
    (HERE/'results/contract_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','cases':len(rows),'max_abs':max(c['checks']['gpu_sparse']['max_abs'] for c in rows)},indent=2))


if __name__=='__main__':main()
