"""All100 Car test cases; same retained prediction reconstructed old and new."""
from __future__ import annotations
import argparse,csv,json,hashlib,time,sys,os,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

from car_request import OUT,E3,configuration_bank,load_backbone,CarRequest
from reconstruct_api import build_plan,CAR_PLAN,CAR_NATIVE,IMPLEMENTATION_VERSION,IMPLEMENTATION_SOURCE_SHA256
from validate_contract import difference
from adasolver.metrics.shapenet_car import dual_field_metrics
from adasolver.metrics.car_cd import CarCdGeometry
from run_attempt import Attempt,atomic_json,preserve_existing,track_status,read_status


def metrics(value,truth,mask,geometry):
    result=asdict(dual_field_metrics(value,truth,mask));cd=float(geometry.compute(value[mask,3],value[mask,:3]))
    gt=float(geometry.compute(truth[mask,3],truth[mask,:3]));result.update(cd_pred=cd,cd_gt=gt,cd_relative_error=abs(cd-gt)/abs(gt))
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert IMPLEMENTATION_VERSION=='1.1'
    protocol=json.loads((OUT/'config/car_protocol.json').read_text());bank=configuration_bank()
    assert bank==protocol['configurations'],'Legacy strategy source no longer matches the locked configurations'
    stage=('car_pilot' if a.device=='cpu' else 'car_pilot_gpu') if a.pilot else 'car_accuracy';folder=OUT/'records'/stage;folder.mkdir(parents=True,exist_ok=True)
    status_path=OUT/'runtime'/f'{stage}.json'
    preserve_existing(status_path);previous=read_status(status_path)
    if previous.get('execution_status')=='completed' and 'source_hashes' not in previous:
        raise FileExistsError(f'{status_path}: historical completed record has no matching source receipt; kept unchanged')
    cases=protocol['validation_ids'][:1] if a.pilot else protocol['test_ids']
    budgets=[.375] if a.pilot else protocol['budgets'];seeds=[0] if a.pilot else protocol['sampling_seeds']
    rows=[];done=set();stream_path=folder/'pairs.jsonl'
    if stream_path.exists():
        for line_number,line in enumerate(stream_path.open(),1):
            try:r=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid accuracy record {stream_path}:{line_number}; raw file preserved, no append performed') from exc
            rows.append(r);done.add((r['backbone'],r['case_id'],r['variant'],r['sampling_seed'],r['budget_fraction']))
    source_hashes={}
    snapshots=folder/'source';snapshots.mkdir(exist_ok=True)
    for name in ['car_accuracy.py','car_request.py','reconstruct_api.py','native_car_reference.py','native_grid_reference.py','run_attempt.py']:
        source=OUT/'code'/name;digest=hashlib.sha256(source.read_bytes()).hexdigest();source_hashes[name]=digest
        dest=snapshots/name
        if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest,'Accuracy source changed; do not append to a different run'
        else:shutil.copy2(source,dest)
    expected=8 if a.pilot else protocol['expected_accuracy_requests']
    assert len(done)==len(rows),'Duplicate prior accuracy request keys'
    if previous.get('execution_status')=='completed':
        assert len(rows)==expected==previous['completed_pairs']
        assert previous['source_hashes']==source_hashes
        assert previous['protocol_sha256']==hashlib.sha256((OUT/'config/car_protocol.json').read_bytes()).hexdigest()
        print(stage,'already complete; unchanged source/protocol and raw request count verified',flush=True)
        return
    track_status(status_path)
    state=dict(execution_status='running',evidence_status='pending',pid=os.getpid(),device=a.device,scope='implementation pilot' if a.pilot else 'all100 locked test cases',
               implementation_version=IMPLEMENTATION_VERSION,implementation_source_sha256=IMPLEMENTATION_SOURCE_SHA256,
               source_hashes=source_hashes,protocol_sha256=hashlib.sha256((OUT/'config/car_protocol.json').read_bytes()).hexdigest(),
               cases=cases,completed_pairs=len(rows),expected_pairs=expected,failures=[])
    atomic_json(status_path,state)
    with stream_path.open('a') as stream:
        for architecture in ['Transolver3','LinearNO']:
            variants=protocol['accuracy_variants'][architecture]
            if a.pilot:variants=[v for v in variants if v!='cpg_beta0']
            runtime,model,context,head=load_backbone(architecture,a.device,with_ca=architecture=='Transolver3')
            for case in cases:
                item=runtime.sample(case);mask=item.surface_mask.numpy();truth=item.target_phys.numpy()
                geometry=CarCdGeometry.load(runtime.data_path/'training_data'/case/'quadpress_smpl.vtk',OUT/'records/geometry_cache')
                geometry.validate_surface(item.pos.numpy()[mask])
                for variant in variants:
                    for budget in ([1.] if variant=='full' else budgets):
                        for seed in ([None] if variant=='full' else seeds):
                            key=(architecture,case,variant,seed,budget)
                            if key in done:continue
                            request=CarRequest(model,context,item,runtime.output_normalizer,bank[variant],budget,seed,device=a.device,ca=head)
                            request.capture_values=True
                            old=request('old','A');ids=request.last_indices.copy();coarse_hash=request.last_coarse_hash
                            if variant=='full':new=old.copy();stats=difference(old,new)
                            else:
                                values=request.last_gpu_physical_values
                                plan=build_plan('shapenet_car',request.coords,ids,surface_mask=mask,workers=4)
                                new=plan.apply_tensor(torch.as_tensor(values,device=a.device)).cpu().numpy()
                                stats=difference(old,new)
                            assert stats['violations']==0,(key,stats)
                            old_metrics=metrics(old,truth,mask,geometry);new_metrics=metrics(new,truth,mask,geometry)
                            delta={name:abs(old_metrics[name]-new_metrics[name]) for name in ['field_error','pressure_rel_l2','velocity_rel_l2','cd_relative_error']}
                            assert all(value<=1e-8+1e-5*abs(old_metrics[name]) for name,value in delta.items()),(key,delta)
                            if a.pilot:
                                # Fresh complete request, not only a reconstruction function.
                                end_to_end=request('new','A')
                                np.testing.assert_array_equal(ids,request.last_indices)
                                assert coarse_hash==request.last_coarse_hash
                                e2e=difference(old,end_to_end);assert e2e['violations']==0
                            else:e2e=None
                            identity=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
                            ipath=folder/'indices'/f'{identity}.npz';ipath.parent.mkdir(exist_ok=True)
                            if not ipath.exists():
                                temporary=ipath.with_suffix(f'.{os.getpid()}.npz.tmp')
                                with temporary.open('wb') as index_stream:np.savez_compressed(index_stream,indices=ids)
                                temporary.replace(ipath)
                            row=dict(backbone=architecture,case_id=case,variant=variant,sampling_seed=seed,budget_fraction=budget,
                                training_seed=0,head_seed=0 if variant=='ca' else None,retained_points=len(ids),
                                beta=bank[variant].get('beta'),metric_target='field',operator_sha256=runtime.metadata['checkpoint_sha256'],
                                normalizer_sha256=runtime.metadata['normalizer_sha256'],selected_indices_sha256=identity,
                                coarse_prediction_sha256=coarse_hash,old_reconstruction_id=CAR_NATIVE,new_reconstruction_id=CAR_PLAN,
                                old_metrics=old_metrics,new_metrics=new_metrics,difference=stats,metric_differences=delta,end_to_end_check=e2e,
                                output_dtype='float64',device=a.device,implementation_source_sha256=IMPLEMENTATION_SOURCE_SHA256,
                                indices_file=str(ipath.relative_to(OUT)))
                            rows.append(row);done.add(key);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                            state['completed_pairs']=len(rows);atomic_json(status_path,state)
                print(architecture,case,len(rows),'pairs',flush=True)
            del model,runtime,head;torch.cuda.empty_cache()
    expected=sum(len(cases)*(1+(len(protocol['accuracy_variants'][arch])-1)*len(budgets)*len(seeds)) for arch in ['Transolver3','LinearNO'])
    if a.pilot:expected=8
    assert len(rows)==expected,(len(rows),expected)
    grouped={(r['backbone'],r['case_id'],r['sampling_seed'],r['budget_fraction'],r['variant']):r for r in rows}
    pairs=0
    if not a.pilot:
        for r in rows:
            if r['variant']!='cpg_beta0':continue
            other=grouped[(r['backbone'],r['case_id'],r['sampling_seed'],r['budget_fraction'],'cpg_beta1')]
            assert r['selected_indices_sha256']==other['selected_indices_sha256']
            assert r['coarse_prediction_sha256']==other['coarse_prediction_sha256'];pairs+=1
    state.update(execution_status='completed',evidence_status='pending',expected_pairs=expected,completed_pairs=len(rows),beta_pairs=pairs,
                 max_field_absolute_difference=max(r['difference']['max_abs'] for r in rows),max_field_metric_difference=max(r['metric_differences']['field_error'] for r in rows))
    atomic_json(status_path,state)
    print(json.dumps({k:v for k,v in state.items() if k!='cases'},indent=2))


if __name__=='__main__':
    with Attempt('car_accuracy'):main()
