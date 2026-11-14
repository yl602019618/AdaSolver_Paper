"""One predeclared real Car fixture: CPU candidate coverage/full-field check.

No model is rerun. Velocity and surface pressure use their original separate
source sets and fixed physical predictions. This does not certify CUDA or
request timing, and does not promote the candidate to accepted API v1.1.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from gpu_knn_candidate import build_exact_gpu_knn,CANDIDATE_VERSION
from reconstruct_api import native_reconstruct
from validate_contract import difference

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    fixtures=json.loads((ROOT/'records/reconstruction_microprofile/fixtures.json').read_text())
    fixture=next(f for f in fixtures if f['dataset']=='shapenet_car' and f['backbone']=='Transolver3')
    data=np.load(ROOT/fixture['fixture_file']);coords=data['reference_coordinates'];selected=data['selected'];values=data['values'];mask=data['surface_mask']
    # Promotion preserves the exact values of the original float32 coordinates.
    coordinate64=np.asarray(coords,dtype=np.float64);full=np.zeros((len(coords),4),np.float64);checks=[]
    selected_surface=np.flatnonzero(mask[selected]);surface_rows=np.flatnonzero(mask)
    for name,source_rows,target_rows,columns in [('velocity',np.arange(len(selected)),np.arange(len(coords)),[0,1,2]),
                                                ('surface_pressure',selected_surface,surface_rows,[3])]:
        source=coordinate64[selected[source_rows]];query=coordinate64[target_rows]
        sx=torch.from_numpy(source);qx=torch.from_numpy(query);calls=[];original=torch.cdist
        def observed(a,b,*args,**kwargs):
            assert b is sx
            calls.append(dict(query_rows=len(a),source_rows=len(b),pair_count=len(a)*len(b)))
            return original(a,b,*args,**kwargs)
        torch.cdist=observed;started=time.perf_counter()
        try:mapping=build_exact_gpu_knn(sx,qx,source_cpu=source,query_cpu=query)
        finally:torch.cdist=original
        physical=torch.from_numpy(values[np.ix_(source_rows,columns)])
        result=mapping.apply(physical).numpy();duration=time.perf_counter()-started
        full[np.ix_(target_rows,columns)]=result
        distance,index=cKDTree(source).query(query,k=min(4,len(source)))
        np.testing.assert_array_equal(mapping.indices.numpy(),index)
        np.testing.assert_allclose(mapping.distances.numpy(),distance,rtol=5e-14,atol=1e-14)
        assert sum(r['query_rows'] for r in calls)==len(query)
        assert max(r['pair_count'] for r in calls)<=mapping.metadata['max_pairs_budget']
        checks.append(dict(output=name,source_points=len(source),query_points=len(query),channels=len(columns),
            all_sources_visited_each_query=True,query_rows_visited_once=sum(r['query_rows'] for r in calls),
            exact_native_neighbor_indices=True,cdist_call_count=len(calls),observed_max_tile_pairs=max(r['pair_count'] for r in calls),
            candidate_metadata=mapping.metadata,diagnostic_execution_seconds=duration,
            diagnostic_time_scope='one CPU correctness check; not formal latency',cdist_blocks=calls))
        print(name,'checked',len(source),'sources',len(query),'queries',round(duration,3),'CPU seconds',flush=True)
    native=native_reconstruct('shapenet_car',selected,values,coords,mask);delta=difference(native,full)
    assert delta['violations']==0 and np.all(full[~mask,3]==0)
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',candidate_version=CANDIDATE_VERSION,
        candidate_source_sha256=hashlib.sha256((ROOT/'code/gpu_knn_candidate.py').read_bytes()).hexdigest(),fixture=fixture,
        original_coordinate_dtype=str(coords.dtype),candidate_coordinate_dtype='float64 (value-preserving promotion)',
        output_dtype=str(full.dtype),checks=checks,full_field_difference=delta,test_device='CPU',cuda_validation_performed=False,
        acceptance='One real fixed-prediction CPU Car reconstruction check only; no CUDA, fresh model end-to-end, speed or peak-GPU-memory claim')
    (ROOT/'results/gpu_knn_candidate_real_car_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Real Car candidate CPU full-field check:',delta,flush=True)


if __name__=='__main__':main()
