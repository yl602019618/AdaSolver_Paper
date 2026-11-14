"""Fresh maps, GPU output and explicitly reused maps: paired full Car requests."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time,traceback,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

from car_request import OUT,configuration_bank,load_backbone,CarRequest
from reconstruct_api import IMPLEMENTATION_VERSION,IMPLEMENTATION_SOURCE_SHA256
from timing_helper import measure_paired,timed_request,isolated_memory
from validate_contract import difference
from adasolver.metrics.shapenet_car import dual_field_metrics
from adasolver.metrics.car_cd import CarCdGeometry
from inference_speed.gpu_monitor import ExclusiveGPU


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def encode_key(*values):return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:20]


def array(value):return value.cpu().numpy() if isinstance(value,torch.Tensor) else value


def run_scope(args,protocol):
    root=OUT/'records/car_timing'/args.scope;root.mkdir(parents=True,exist_ok=True)
    source_dir=root/'source';source_dir.mkdir(exist_ok=True)
    source_hashes={}
    for name in ['car_timing.py','car_request.py','timing_helper.py','reconstruct_api.py','native_car_reference.py','native_grid_reference.py']:
        source=OUT/'code'/name;digest=hashlib.sha256(source.read_bytes()).hexdigest();source_hashes[name]=digest
        destination=source_dir/name
        if destination.exists():assert hashlib.sha256(destination.read_bytes()).hexdigest()==digest,'Source changed; use new run ID'
        else:shutil.copy2(source,destination)
    write(root/'source_manifest.json',source_hashes)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    bank=configuration_bank();resources={};model_preparation=[]
    assert bank==protocol['configurations'],'Legacy strategy source no longer matches the locked configurations'
    expected={}
    source=OUT/'records/car_accuracy/pairs.jsonl'
    if not source.exists():raise RuntimeError('complete100-case accuracy must precede formal timing')
    for line in source.open():
        row=json.loads(line);expected[(row['backbone'],row['case_id'],row['variant'],row['sampling_seed'],row['budget_fraction'])]=row
    if len(expected)!=7400:raise RuntimeError('Car accuracy matrix is not complete')
    stage_path=OUT/'runtime'/f'car_timing_{args.scope}.json'
    state=dict(execution_status='running',evidence_status='pending',pid=os.getpid(),physical_gpu=args.gpu,
               scope=args.scope,completed_groups=[],expected_groups=30,unique_conditions=0,
               api_version=IMPLEMENTATION_VERSION,api_source_sha256=IMPLEMENTATION_SOURCE_SHA256)
    write(stage_path,state)
    with ExclusiveGPU(args.gpu,root/'gpu_telemetry.json',interval=1.) as monitor:
        from adasolver.inference.native_fps import load_native_fps
        started=time.perf_counter();load_native_fps();native_load_ms=(time.perf_counter()-started)*1000
        write(root/'runtime_environment.json',dict(physical_gpu=args.gpu,pid=os.getpid(),
            gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__,torch_cuda=torch.version.cuda,
            cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=torch.get_num_threads(),load_average=os.getloadavg(),
            native_fps_process_load_ms=native_load_ms,native_fps_cache=os.environ.get('ADASOLVER_NATIVE_CACHE'),
            fp32=True,allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
        # Admit before loading any CUDA model; the admission check requires an
        # idle card and would correctly reject even our own preloaded model.
        for arch in ['Transolver3','LinearNO']:
            t=time.perf_counter();runtime,model,context,head=load_backbone(arch,'cuda:0',with_ca=arch=='Transolver3')
            torch.cuda.synchronize();resources[arch]=(runtime,model,context,head)
            model_preparation.append(dict(backbone=arch,model_head_load_ms=(time.perf_counter()-t)*1000,
                checkpoint_sha256=runtime.metadata['checkpoint_sha256'],normalizer_sha256=runtime.metadata['normalizer_sha256']))
        write(root/'model_preparation.json',model_preparation)
        for case_index,case in enumerate(protocol['timing_cases']):
            first=resources['Transolver3'][0];item=first.sample(case);truth=item.target_phys.numpy();mask=item.surface_mask.numpy()
            geom=CarCdGeometry.load(first.data_path/'training_data'/case/'quadpress_smpl.vtk',OUT/'records/geometry_cache')
            gt=float(geom.compute(truth[mask,3],truth[mask,:3]))
            groups=[('full',None,1.)]+[(f'b{budget}_s{seed}',seed,budget) for budget in protocol['budgets'] for seed in protocol['sampling_seeds']]
            for tag,seed,budget in groups:
                cpu_start_load=os.getloadavg()
                key=f'case{case_index}_{tag}';directory=root/key;directory.mkdir(exist_ok=True)
                finished=directory/'complete.json'
                if finished.exists() and json.loads(finished.read_text()).get('execution_status')=='completed':
                    previous=json.loads(finished.read_text());assert previous['source_hashes']==source_hashes
                    state['completed_groups'].append(key);state['unique_conditions']+=previous['conditions'];continue
                requests={};labels={};preparation=[]
                for arch in ['Transolver3','LinearNO']:
                    runtime,model,context,head=resources[arch]
                    variants=['full'] if tag=='full' else [v for v in protocol['timing_variants'][arch] if v!='full']
                    for variant in variants:
                        request=CarRequest(model,context,item,runtime.output_normalizer,bank[variant],budget,seed,ca=head)
                        input_setup=request.prepare_scope(args.scope)
                        for impl in (['shared'] if variant=='full' else ['old','new']):
                            name=f'{arch}:{variant}:{impl}';requests[name]=(lambda r=request,i=impl:r('old' if i=='shared' else i,args.scope))
                            labels[name]=dict(backbone=arch,variant=variant,implementation=impl,request=request)
                        preparation.append(dict(backbone=arch,variant=variant,geometry_preparation_ms=request.preparation_ms,
                            geometry_preparation_scope='scorer/sampler and surface-distance construction are inside every request',resident_input_h2d_ms=input_setup))
                originals={};checks=[]
                def check(name,output):
                    info=labels[name];request=info['request'];out=array(output)
                    assert out.dtype==np.float64 and out.shape==(len(item),4) and np.isfinite(out).all()
                    assert np.all(out[~mask,3]==0)
                    values=asdict(dual_field_metrics(out,truth,mask));cd=float(geom.compute(out[mask,3],out[mask,:3]));values.update(cd_relative_error=abs(cd-gt)/abs(gt))
                    ref=expected[(info['backbone'],case,info['variant'],seed,budget)]
                    source_metrics=ref['old_metrics'] if info['implementation'] in ['old','shared'] else ref['new_metrics']
                    deltas={k:abs(values[k]-source_metrics[k]) for k in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error']}
                    assert max(deltas.values())<2e-6,(name,deltas)
                    identity=hashlib.sha256(np.asarray(request.last_indices,np.int64).tobytes()).hexdigest()
                    assert identity==ref['selected_indices_sha256'],(name,'selected points differ')
                    coarse_identity=None if request.last_coarse_values is None else hashlib.sha256(np.ascontiguousarray(request.last_coarse_values).tobytes()).hexdigest()
                    assert coarse_identity==ref['coarse_prediction_sha256'],(name,'coarse prediction differs')
                    originals[name]=out.copy()
                    comparison=None
                    if info['implementation']=='new':
                        comparison=difference(originals[name.replace(':new',':old')],out);assert comparison['violations']==0
                    entry=dict(condition=name,field_metrics=values,source_metric_deltas=deltas,selected_indices_sha256=identity,
                               coarse_prediction_sha256=coarse_identity,old_new_difference=comparison)
                    checks.append(entry);return entry
                def progress(kind,done,total):
                    write(root/'heartbeat.json',dict(pid=os.getpid(),group=key,kind=kind,repetition=done,total=total,time=time.time()))
                measured=measure_paired(requests,scope=args.scope,warmups=20,repeats=100,
                    order_seed=20261114+case_index*1000+int(budget*1000)+(seed or 0),check_output=check,
                    guard=monitor.assert_exclusive,progress=progress)
                # Ensure the final repeated sample set still matches the checked one.
                for entry in checks:
                    request=labels[entry['condition']]['request']
                    assert hashlib.sha256(np.asarray(request.last_indices,np.int64).tobytes()).hexdigest()==entry['selected_indices_sha256']
                profiles=[];cache_records=[]
                for name,info in labels.items():
                    req=info['request'];impl='old' if info['implementation']=='shared' else info['implementation']
                    if args.scope=='C':req.clear_caches()
                    for repeat in range(5):
                        _,stages,map_info=req.profile(impl,args.scope)
                        profiles.extend(dict(condition=name,profile_repeat=repeat,cache_hit=req.last_cache_hit,**stage) for stage in stages)
                        if map_info:cache_records.append(dict(condition=name,profile_repeat=repeat,cache_hit=req.last_cache_hit,**map_info))
                for filename,rows in [('timing_repeats.csv',measured['raw']),('summary.csv',measured['summary'])]:
                    with (directory/filename).open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
                write(directory/'profiles.json',profiles);write(directory/'cache.json',cache_records);write(directory/'first_requests.json',measured['first_requests'])
                write(directory/'alignment.json',dict(passed=True,checks=checks));write(directory/'order.json',measured['order']);write(directory/'preparation.json',preparation)
                write(finished,dict(execution_status='completed',evidence_status='pending',case_id=case,scope=args.scope,
                    sampling_seed=seed,budget_fraction=budget,conditions=len(requests),formal_repeats=100,warmups=20,
                    api_source_sha256=IMPLEMENTATION_SOURCE_SHA256,source_hashes=source_hashes,
                    cpu_load_average_before=cpu_start_load,cpu_load_average_after=os.getloadavg(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=torch.get_num_threads()))
                state['completed_groups'].append(key);state['unique_conditions']+=len(requests);write(stage_path,state)
                print(args.scope,key,'complete',len(state['completed_groups']),'/30',flush=True)
    assert len(state['completed_groups'])==30 and state['unique_conditions']==330,state
    state['execution_status']='completed';write(stage_path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['A','B','C'],required=True);p.add_argument('--gpu',type=int,default=6);args=p.parse_args()
    protocol=json.loads((OUT/'config/car_protocol.json').read_text())
    assert args.gpu==protocol['physical_gpu'];assert IMPLEMENTATION_VERSION=='1.1'
    run_scope(args,protocol)


if __name__=='__main__':main()
