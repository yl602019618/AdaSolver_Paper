"""Formal complete-CPU-field inference for one frozen policy target."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from common import ROOT,E3,WORKING_ROOT,activate_sources,sha256,write_json,verify_lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['field','cd'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--cpu-set',required=True);a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    cpus={int(x) for x in a.cpu_set.split(',')};assert cpus<=set(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus)
    activate_sources();verify_lock()
    import sys
    sys.path.append(str(WORKING_ROOT/'AdaSolver_official'))
    import numpy as np
    import torch
    from evaluate_linearno import LinearNORuntime
    from linearno_adapter import install_linear_compensation
    from linearno_ca import LinearNOCA
    from inference_request import LinearNOCarRequest
    from accuracy_tools import load_runs,validate_runs,expected_configs
    from adasolver.metrics.shapenet_car import dual_field_metrics
    from adasolver.metrics.car_cd import CarCdGeometry
    from inference_speed.gpu_monitor import ExclusiveGPU
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    head_acceptance=json.loads((ROOT/'head_acceptance.json').read_text());assert head_acceptance['passed']
    runs=load_runs();accuracy=validate_runs(runs);assert accuracy['passed'],accuracy['findings']
    protocol=json.loads((ROOT/'inference_timing_protocol.json').read_text());assert protocol['request_sha256']==sha256(ROOT/'inference_request.py')
    destination=ROOT/'records/formal_inference'/a.target;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'run_manifest.json').exists():raise FileExistsError(destination)
    model_manifest=E3/'checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json'
    runtime=LinearNORuntime(model_manifest)
    head_manifest=ROOT/'checkpoints/seed0/head/ca_pressure_linearno_trainingseed0_headseed0_manifest.json'
    state={'status':'running','target':a.target,'training_seed':0,'sampling_seeds':[0,1,2],'budgets':[.375,.5,.75],
           'warmups':20,'formal_repeats':100,'physical_gpu':a.gpu,'cpu_affinity':sorted(cpus),
           'checkpoint_sha256':runtime.metadata['checkpoint_sha256'],'head_sha256':head_acceptance['head_sha256'],
           'protocol_sha256':sha256(ROOT/'inference_timing_protocol.json'),'request_sha256':sha256(ROOT/'inference_request.py'),
           'source_sha256':sha256(__file__),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'complete_conditions':0,'complete_groups':0,'conditions_expected':111,'memory_claim':'none; all policies share the process'}
    write_json(destination/'run_manifest.json',state)
    reference={}
    configs=expected_configs();names=['uniform','pi_'+a.target,'cpg_'+a.target,'ca_'+a.target]
    for name in ['full',*names]:
        for row in runs[name]['rows']:
            reference[name,row['sample_id'],None if name=='full' else row['seed'],row['budget_fraction']]=row
    fields=['target','method','variant','case_id','sampling_seed','budget','phase','repeat','order_position','wall_ms','physical_gpu']
    summaries=[];alignment=[];raw_count=0
    digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    try:
        with ExclusiveGPU(a.gpu,destination/'gpu_telemetry.json',interval=1.) as monitor:
            model=runtime.load_operator('cuda:0');context,_=install_linear_compensation(model)
            ca=LinearNOCA(model,head_manifest,runtime.binding(),'cuda:0')
            state['environment']={'torch':str(torch.__version__),'cuda':torch.version.cuda,'numpy':np.__version__,'device':torch.cuda.get_device_name(),'fp32':True,'TF32':False}
            with (destination/'timing_repeats.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
                for case_index,case in enumerate(protocol['timing_cases']):
                    prep=time.perf_counter();item=runtime.sample(case)
                    geom=CarCdGeometry.load(runtime.data_path/'training_data'/case/'quadpress_smpl.vtk',ROOT/'records/geometry_cache')
                    geom.validate_surface(item.pos.numpy()[item.surface_mask.numpy()]);preparation_ms=(time.perf_counter()-prep)*1000
                    truth=item.target_phys.numpy();mask=item.surface_mask.numpy()
                    gt=float(geom.compute(truth[mask,3],truth[mask,:3]))
                    groups=[('full',None,1.)]+[(f's{s}_b{b}',s,b) for s in [0,1,2] for b in [.375,.5,.75]]
                    for tag,seed,budget in groups:
                        labels=['full'] if tag=='full' else names;requests={};identities={};latencies={name:[] for name in labels}
                        for name in labels:
                            request=LinearNOCarRequest(model,context,item,runtime.output_normalizer,configs[name],budget,seed,
                                     ca=ca,pressure_influence=geom.pressure_influence,device='cuda:0')
                            out=request();torch.cuda.synchronize()
                            assert out.shape==(len(item),4) and out.dtype==np.float64 and np.isfinite(out).all()
                            assert (out[~mask,3]==0).all()
                            identity=request.point_identity();ref=reference[name,case,seed,budget]
                            assert identity['selected_indices_sha256']==ref['selected_indices_sha256'],(name,case,'selected points')
                            assert identity['coarse_indices_sha256']==ref['coarse_indices_sha256'],(name,case,'coarse points')
                            metrics=vars(dual_field_metrics(out,truth,mask));cd=float(geom.compute(out[mask,3],out[mask,:3]));metrics['cd_relative_error']=abs(cd-gt)/max(abs(gt),1e-12)
                            delta={k:abs(metrics[k]-ref[k]) for k in metrics}
                            assert max(delta.values())<=2e-6,(name,case,delta)
                            alignment.append({'target':a.target,'variant':name,'case_id':case,'sampling_seed':seed,'budget':budget,
                                              'identity':identity,'native_metric_absolute_differences':delta,'native_metrics':metrics,
                                              'data_and_mesh_influence_preparation_ms':preparation_ms})
                            requests[name]=request;identities[name]=identity
                        order_rng=np.random.default_rng(202611170+case_index*10000+(seed or 0)*100+int(budget*1000)+(0 if a.target=='field' else 1))
                        for phase,repeats in [('warmup',20),('formal',100)]:
                            for repeat in range(repeats):
                                monitor.assert_exclusive()
                                for position,name in enumerate(order_rng.permutation(labels).tolist()):
                                    torch.cuda.synchronize();started=time.perf_counter();result=requests[name]();torch.cuda.synchronize();wall=(time.perf_counter()-started)*1000
                                    writer.writerow({'target':a.target,'method':configs[name]['method'],'variant':name,'case_id':case,
                                         'sampling_seed':'' if seed is None else seed,'budget':budget,'phase':phase,'repeat':repeat,
                                         'order_position':position,'wall_ms':wall,'physical_gpu':a.gpu});raw_count+=1
                                    if phase=='formal':latencies[name].append(wall)
                                if repeat%20==0:write_json(destination/'heartbeat.json',{'case_index':case_index,'group':tag,'phase':phase,'repeat':repeat})
                        for name in labels:
                            assert requests[name].point_identity()==identities[name]
                            values=latencies[name];assert len(values)==100
                            summaries.append({'target':a.target,'method':configs[name]['method'],'variant':name,'case_id':case,
                                'sampling_seed':'' if seed is None else seed,'budget':budget,'physical_gpu':a.gpu,'formal_repeats':100,
                                'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'mean_ms':float(np.mean(values)),
                                'repeat_sd_ms':float(np.std(values,ddof=1))})
                        stream.flush();state['complete_conditions']+=len(labels);state['complete_groups']+=1
                        write_json(destination/'run_manifest.json',state);write_json(destination/'alignment.json',{'passed':True,'conditions':alignment})
                        print(a.target,case_index,tag,'conditions',state['complete_conditions'],'/111',flush=True)
            assert state['complete_conditions']==111 and state['complete_groups']==30 and raw_count==13320
            with (destination/'latency_summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(summaries)
        state.update(status='completed',execution_status='completed')
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(destination/'run_manifest.json',state);raise
    state['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['total_observations']=raw_count
    write_json(destination/'run_manifest.json',state)


if __name__=='__main__':main()
