"""Fixed-checkpoint real validation cases for the reconstruction-only gate."""
from __future__ import annotations
from pathlib import Path
import sys,json,time,hashlib,csv
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1];CURRENT=OUT.parents[1]
E3=CURRENT/'rebuttal_20261112/experiments/E3_backbones'
sys.path[:0]=[str(E3/'code'),str(E3/'code/vendor'),str(CURRENT/'AdaSolver_official')]
from reconstruct_api import build_plan,native_reconstruct,CAR_NATIVE,CAR_PLAN,GRID_NATIVE,GRID_PLAN
from validate_contract import difference
from adasolver.data.shapenet_car_runtime import CarRuntime
from evaluate_linearno import LinearNORuntime
from linearno_adapter import install_linear_compensation
from adasolver.layers import install_compensation
from fullfield_request import LinearNOFullFieldRequest
from adasolver.metrics.shapenet_car import dual_field_metrics
from adasolver.metrics.car_cd import CarCdGeometry
from adasolver.models.checkpoint import load_darcy_operator,load_airfoil2d_operator,sha256_file
from adasolver.data.pipe import UnitTransformer
from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.utils.interpolation import index_grid
from adasolver.evaluation.airfoil2d import pressure_surface_force


def primitive_errors(old,new,truth):
    denom=np.linalg.norm(truth)
    a=float(np.linalg.norm(old-truth)/denom);b=float(np.linalg.norm(new-truth)/denom)
    return {'old_field_error':a,'new_field_error':b,'field_error_delta':abs(a-b)}


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if (OUT/'results/real_validation.json').exists():raise FileExistsError('real validation exists')
    shared=json.loads((CURRENT/'linearno_followup_20261114/standard_benchmarks/split_manifest.json').read_text())
    val=shared['validation_ids'];standard_ids=[val[0],val[len(val)//2],val[-1]]
    split=json.loads((E3/'config/splits_seed0.json').read_text());cv=split['operator_validation'];car_ids=[cv[0],cv[len(cv)//2],cv[-1]]
    protocol=dict(validation_cases={'darcy':standard_ids,'airfoil':standard_ids,'shapenet_car':car_ids},
                  budgets=[.375,.5,.75],sampling_seeds=[0,1,2],
                  tolerance='abs(new-old)<=1e-6+1e-5*abs(old)',metric_tolerance='absolute downstream error change <=1e-8 +1e-5*abs(old metric)',
                  checkpoint_scope='published frozen checkpoints supply physical prediction values for implementation validation, not new-cohort accuracy',
                  final_new_models='must additionally compare reconstruction and metrics on their own validation cases before final tests',
                  gpu=7,workers=4)
    (OUT/'config/real_validation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    def check(dataset,case,backbone,seed,budget,ids,values,coords,truth,mask=None,geometry=None,checkpoint=None,model_inputs=None):
        t=time.perf_counter();old=native_reconstruct(dataset,ids,values,coords,mask);old_ms=(time.perf_counter()-t)*1000
        plan=build_plan(dataset,coords,ids,surface_mask=mask,channels=values.shape[1],workers=4,device='cuda:0')
        new=plan.apply_tensor(torch.tensor(values,device='cuda:0')).cpu().numpy()
        stats=difference(old,new);assert stats['violations']==0,(dataset,case,stats)
        metrics=primitive_errors(old,new,truth)
        if dataset=='shapenet_car':
            old_metrics=dual_field_metrics(old,truth,mask);new_metrics=dual_field_metrics(new,truth,mask)
            metrics.update(old_field_error=old_metrics.field_error,new_field_error=new_metrics.field_error,
                           field_error_delta=abs(old_metrics.field_error-new_metrics.field_error),
                           pressure_error_delta=abs(old_metrics.pressure_rel_l2-new_metrics.pressure_rel_l2),
                           velocity_error_delta=abs(old_metrics.velocity_rel_l2-new_metrics.velocity_rel_l2))
            cd0=float(geometry.compute(old[mask,3],old[mask,:3]));cd1=float(geometry.compute(new[mask,3],new[mask,:3]))
            metrics.update(old_cd=cd0,new_cd=cd1,cd_delta=abs(cd0-cd1))
        elif dataset=='airfoil':
            f0=pressure_surface_force(model_inputs,old);f1=pressure_surface_force(model_inputs,new)
            metrics.update(cd_delta=abs(f0.drag_coefficient-f1.drag_coefficient),cl_delta=abs(f0.lift_coefficient-f1.lift_coefficient))
        assert metrics['field_error_delta']<=1e-8+1e-5*abs(metrics['old_field_error'])
        for key in ['cd_delta','cl_delta','pressure_error_delta','velocity_error_delta']:
            if key in metrics:assert metrics[key]<1e-8
        row=dict(dataset=dataset,case_id=str(case),backbone=backbone,sampling_seed=seed,budget=budget,
                 retained_points=len(ids),checkpoint_sha256=checkpoint,indices_sha256=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),
                 coordinates_dtype=str(coords.dtype),values_dtype=str(values.dtype),output_dtype=str(new.dtype),
                 native_reconstruction_ms=old_ms,plan=plan.timings,**stats,**metrics)
        records.append(row)
        with (OUT/'records/real_validation.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    # Car values from both already-trained backbones; fixed uniform point sets.
    p=json.loads((E3/'config/latency_protocol.json').read_text())
    for name,cls,key in [('Transolver3',CarRuntime,'t3_manifest'),('LinearNO',LinearNORuntime,'linearno_manifest')]:
        runtime=cls(p[key]);model=runtime.load_operator('cuda:0')
        context,_=install_compensation(model) if name=='Transolver3' else install_linear_compensation(model)
        for case in car_ids:
            item=runtime.sample(case);coords=item.pos.numpy();mask=item.surface_mask.numpy();truth=item.target_phys.numpy()
            geometry=CarCdGeometry.load(runtime.data_path/'training_data'/case/'quadpress_smpl.vtk',OUT/'records/car_geometry_cache')
            for budget in protocol['budgets']:
                for seed in protocol['sampling_seeds']:
                    rng=np.random.default_rng(np.random.SeedSequence([20261114,seed,len(coords),round(budget*10000)]))
                    ids=rng.choice(len(coords),round(budget*len(coords)),replace=False)
                    context.clear();pred=model(item.pos[ids][None].to('cuda:0'),item.x_norm[ids][None].to('cuda:0'))[0].cpu()
                    values=runtime.output_normalizer.decode(pred).numpy()
                    check('shapenet_car',case,name,seed,budget,ids,values,coords,truth,mask,geometry,runtime.metadata['checkpoint_sha256'])
            print('Car validation',name,case,flush=True)
        del model,runtime;torch.cuda.empty_cache()

    # Published Darcy supplies values; local raw cache avoids re-reading large MATs.
    data=CURRENT/'linearno_followup_20261114/standard_benchmarks/darcy/data'
    coeff=np.load(data/'pool_coeff.npy',mmap_mode='r').reshape(1000,-1,1)
    truth_all=np.load(data/'pool_solution.npy',mmap_mode='r').reshape(1000,-1,1)
    pos=torch.from_numpy(np.load(data/'positions.npy'));coords=np.load(data/'reference_coordinates.npy')
    xn=UnitTransformer(torch.from_numpy(np.array(coeff)))
    checkpoint=CURRENT/'AdaSolver_official/checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_best.pt'
    manifest=checkpoint.with_name(checkpoint.name.replace('_best.pt','_manifest.json'))
    model=load_darcy_operator(checkpoint,manifest,'cuda:0');model.requires_grad_(False)
    yn=np.load(checkpoint.with_name(checkpoint.name.replace('_best.pt','_ynorm.npz')))
    for case in standard_ids:
        fx=xn.encode(torch.from_numpy(np.array(coeff[case:case+1])))[0]
        for budget in protocol['budgets']:
            for seed in protocol['sampling_seeds']:
                ids=np.random.default_rng(np.random.SeedSequence([20261114,case,seed,round(budget*10000)])).choice(len(coords),round(budget*len(coords)),replace=False)
                pred=model(pos[ids][None].to('cuda:0'),fx[ids][None].to('cuda:0'))[0].cpu().numpy()
                values=pred*np.asarray(yn['std']).reshape(1,1)+np.asarray(yn['mean']).reshape(1,1)
                check('darcy',case,'published_Transolver3',seed,budget,ids,values,coords,truth_all[case],checkpoint=sha256_file(checkpoint))
    del model;torch.cuda.empty_cache()
    checkpoint=CURRENT/'AdaSolver_official/checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_best.pt'
    manifest=checkpoint.with_name(checkpoint.name.replace('_best.pt','_manifest.json'))
    model=load_airfoil2d_operator(checkpoint,manifest,'cuda:0');model.requires_grad_(False)
    air=AirfoilRuntimeData('/mnt/ai-material-data/shuaibinqu/NACA/data/naca',manifest);coords=index_grid((221,51))
    for case in standard_ids:
        physical=air.coords(case);truth=air.target(case);available=np.setdiff1d(np.arange(len(coords)),air.mandatory)
        for budget in protocol['budgets']:
            for seed in protocol['sampling_seeds']:
                rng=np.random.default_rng(np.random.SeedSequence([20261114,case,seed,round(budget*10000)]))
                ids=np.r_[air.mandatory,rng.choice(available,round(budget*len(coords))-len(air.mandatory),False)]
                x=torch.tensor(physical[ids][None],device='cuda:0');values=air.decode(model(x,x))[0].cpu().numpy()
                check('airfoil',case,'published_Transolver3',seed,budget,ids,values,coords,truth,checkpoint=sha256_file(checkpoint),model_inputs=physical)
    result=dict(status='passed',records=len(records),protocol=protocol,max_abs=max(x['max_abs'] for x in records),
                max_relative_norm=max(x['relative_norm'] for x in records),max_field_metric_delta=max(x['field_error_delta'] for x in records))
    (OUT/'results/real_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
