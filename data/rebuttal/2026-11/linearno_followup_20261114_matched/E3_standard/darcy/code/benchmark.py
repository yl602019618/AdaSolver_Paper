"""Controlled same-GPU A/B/C old/new requests for the newly trained Darcy cohort."""
from __future__ import annotations
import argparse,csv,json,os,sys,time,gc,subprocess,platform
from pathlib import Path
import numpy as np
import torch
from common import OUT,FOLLOWUP,read,write,digest
from inference import (Runtime,CompleteRequest,metric,array_hash,reconstruction_module,budget_count,
                        GRID_NATIVE,GRID_PLAN,build_plan,native_reconstruct)
from benchmark_core import run_group
from timing_helper import isolated_memory
from adasolver.inference import adaptive

CURRENT=FOLLOWUP.parent
sys.path.append(str(CURRENT/'AdaSolver_official'))
from inference_speed.gpu_monitor import ExclusiveGPU

SCOPES={'A':'A_cpu_cpu','B':'B_gpu_gpu','C':'C_cpu_cpu_cached_map'}


def identity(request):
    last=request.last
    return dict(selected_indices_hash=array_hash(np.asarray(last['indices'],np.int64)),
                coarse_indices_hash=None if last['coarse_indices'] is None else array_hash(np.asarray(last['coarse_indices'],np.int64)),
                compensation_weights_hash=None if last['weights'] is None else array_hash(np.asarray(last['weights'],np.float32)))


def metadata(runtime,sid,method,budget,seed,implementation,scope):
    rid='identity_full' if method=='full' else GRID_NATIVE if implementation=='old' else GRID_PLAN
    if scope=='C' and method!='full':rid='grid_qhull_cached_cpu_apply_v1p1' if implementation=='old' else 'grid_qhull_cached_gpu_apply_v1p1'
    return dict(dataset='darcy',backbone=runtime.label,model_variant=runtime.variant,case_id=sid,method=method,
                beta=None if method in ['full','uniform'] else .5 if method=='pi' else 1.,budget_fraction=budget,
                sampling_seed=seed,training_seed=0,head_seed=0 if method=='ca' else None,
                checkpoint_id=runtime.metadata['checkpoint_sha256'],normalizer_id=runtime.metadata['normalizer_sha256'],
                head_checkpoint_id=None if method!='ca' else runtime.head_metadata['checkpoint_sha256'],
                retained_points=7225 if method=='full' else budget_count(7225,budget),test_population='historical_smooth2_200',
                request_code_sha256=digest(__file__),inference_code_sha256=digest(Path(__file__).with_name('inference.py')),
                timing_core_sha256=digest(Path(__file__).with_name('benchmark_core.py')),
                implementation=implementation,scope=SCOPES[scope],reconstruction_id=rid,
                metric_target='field',output_device='cuda' if scope=='B' else 'cpu',output_dtype='float64',
                input_device='cuda' if scope=='B' else 'cpu',input_dtype='float32',
                input_features='model-ready xy and normalized coefficient; prediction physical decode included in request',
                first_preparation='model/load/kernel setup separate; first request measured before plan-cache warmup')


def reference_values():
    values={}
    for key in ['transolver3','linearno','linearno_conv']:
        p=OUT/'records'/key/'metrics_records.csv'
        for row in csv.DictReader(p.open()):
            k=(row['backbone'],int(row['case_id']),row['method'],float(row['budget_fraction']),row['sampling_seed'],row['reconstruction_id'])
            row['_arrays_path']=str(p.parent/row['arrays_path']);values[k]=row
    return values


def check_output(request,output,references):
    if torch.is_tensor(output):
        if output.dtype!=torch.float64 or output.device.type!='cuda':raise RuntimeError('B output device/dtype differs')
        output=output.cpu().numpy()
    else:
        if output.dtype!=np.float64:raise RuntimeError('CPU output dtype differs')
    rid=GRID_NATIVE if request.method=='full' or request.reconstruction=='old' else GRID_PLAN
    key=(request.runtime.label,request.sid,request.method,float(request.budget),'' if request.seed is None else str(request.seed),rid)
    row=references[key];current=metric(output,request.runtime.case(request.sid)['truth'])
    expected=float(row['value']);difference=abs(current-expected)
    if difference>1e-7+1e-5*abs(expected):raise RuntimeError('timed output differs from accepted accuracy')
    point=identity(request)
    if point['selected_indices_hash']!=row['selected_indices_hash']:raise RuntimeError('timed selected points differ')
    if (point['coarse_indices_hash'] or '')!=row['coarse_indices_hash']:raise RuntimeError('timed coarse points differ')
    if (point['compensation_weights_hash'] or '')!=row['compensation_weights_hash']:raise RuntimeError('timed compensation differs')
    with np.load(row['_arrays_path']) as saved:
        baseline=saved['prediction'].astype(np.float64) if request.method=='full' else native_reconstruct('darcy',saved['indices'],saved['prediction'],request.runtime.reference)
    absolute=np.abs(output-baseline)
    if np.any(absolute>1e-6+1e-5*np.abs(baseline)):raise RuntimeError('timed complete field differs from fixed accepted prediction')
    return dict(field_error=current,reference_field_error=expected,absolute_difference=difference,**point,
                full_field_max_absolute_difference=float(absolute.max()),
                full_field_relative_difference=float(np.linalg.norm(output-baseline)/max(np.linalg.norm(baseline),1e-300)),
                accepted_prediction_arrays=row['_arrays_path'],
                cached_map_cpu_bytes=request.last['plan_cpu_bytes'],cached_map_gpu_bytes=request.last['plan_device_bytes'],
                cache_hit=request.last['cache_hit'])


@torch.no_grad()
def profile_request(request,repeats=5):
    """Independent instrumented pass, never added into the direct total."""
    result=[];rt=request.runtime
    def measure(label,fn,repeat):
        torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
        result.append(dict(profile_repeat=repeat,component=label,ms=(time.perf_counter()-start)*1000,
                           timing_scope='separate instrumented pass'))
        return out
    if request.scope=='C' and request.method!='full':
        # Independent cold-map profile. This is outside the directly timed first request.
        sampled=rt.sample(request.sid,request.method,request.budget,request.seed,request.scope)
        cold=measure('first_cache_map_build',lambda:build_plan('darcy',rt.reference,sampled[0],channels=1,
                     reconstruction_id=GRID_PLAN,device=rt.device if request.reconstruction=='new' else None,profile=True),-1)
        result[-1].update(cache_cpu_bytes=cold.cpu_bytes,cache_gpu_bytes=cold.device_bytes(),
                          timing_scope='separate cold-map profile; includes construction and map transfer')
        del cold
    if request.method.startswith('cpg') or request.method=='ca':
        for repeat in range(repeats):
            measure('native_patch_geometry_build_probe',lambda:adaptive.build_patches(rt.reference,32,seed=0),repeat)
            result[-1]['timing_scope']='separate geometry probe; already included in sampling stage and direct total'
    for repeat in range(repeats):
        sampled=measure('sampling_coarse_score_selection_compensation',
            lambda:rt.sample(request.sid,request.method,request.budget,request.seed,request.scope),repeat)
        ids,weights,coarse,cp,meta=sampled
        if rt.context:measure('attention_weight_upload_and_transform',lambda:rt.context.set(None if weights is None else torch.tensor(weights,dtype=torch.float32,device=rt.device)),repeat)
        try:prediction=measure('final_forward_with_input_transfer',lambda:rt.forward(rt.case(request.sid),None if request.method=='full' else ids,request.scope),repeat)
        finally:
            if rt.context:rt.context.clear()
        pred=measure('physical_decode',lambda:prediction*rt.std[0]+rt.mean[0],repeat)
        if request.method=='full':
            measure('output_conversion_transfer',lambda:pred.to(torch.float64) if request.scope=='B' else pred.cpu().numpy().astype(np.float64),repeat)
            continue
        def reconstruction_complete_output():
            if request.scope=='C':
                plan,hit=request.cache.get('darcy',rt.reference,ids,geometry_identity=f'darcy_case_{request.sid}',
                    reconstruction_id=GRID_PLAN,channels=1,device=rt.device if request.reconstruction=='new' else None)
                return plan.apply_numpy(pred.cpu().numpy()) if request.reconstruction=='old' else plan.apply_tensor(pred).cpu().numpy()
            if request.reconstruction=='old':
                field=native_reconstruct('darcy',ids,pred.cpu().numpy(),rt.reference)
                return torch.as_tensor(field,dtype=torch.float64,device=rt.device) if request.scope=='B' else field
            plan=build_plan('darcy',rt.reference,ids,channels=1,reconstruction_id=GRID_PLAN,device=rt.device)
            field=plan.apply_tensor(pred)
            return field if request.scope=='B' else field.cpu().numpy()
        measure('reconstruction_complete_output',reconstruction_complete_output,repeat)
        if request.scope=='C':
            plan,hit=measure('cache_key_and_lookup',lambda:request.cache.get('darcy',rt.reference,ids,
                geometry_identity=f'darcy_case_{request.sid}',reconstruction_id=GRID_PLAN,channels=1,
                device=rt.device if request.reconstruction=='new' else None),repeat)
            if request.reconstruction=='old':
                cpu=measure('prediction_d2h',lambda:pred.cpu().numpy(),repeat)
                measure('cached_cpu_sparse_apply',lambda:plan.apply_numpy(cpu),repeat)
            else:
                field=measure('cached_gpu_sparse_apply',lambda:plan.apply_tensor(pred),repeat)
                measure('complete_field_d2h',lambda:field.cpu().numpy(),repeat)
            result.append(dict(profile_repeat=repeat,component='cache_hit',ms=None,timing_scope='cache metadata',cache_hit=hit))
        elif request.reconstruction=='old':
            cpu=measure('prediction_d2h',lambda:pred.cpu().numpy(),repeat)
            field=measure('native_reconstruction_build_query_apply',lambda:native_reconstruct('darcy',ids,cpu,rt.reference),repeat)
            if request.scope=='B':measure('complete_field_h2d',lambda:torch.tensor(field,dtype=torch.float64,device=rt.device),repeat)
        else:
            plan=measure('new_plan_build_and_map_transfer',lambda:build_plan('darcy',rt.reference,ids,
                channels=1,reconstruction_id=GRID_PLAN,device=rt.device,profile=True),repeat)
            for mapping in plan.timings.get('maps',[]):
                for name,value in mapping.items():
                    if name.endswith('_ms'):result.append(dict(profile_repeat=repeat,component='nested_'+name,ms=value,timing_scope='nested map profile, not additive to parent component'))
            field=measure('gpu_sparse_apply',lambda:plan.apply_tensor(pred),repeat)
            if request.scope=='A':measure('complete_field_d2h',lambda:field.cpu().numpy(),repeat)
    return result


def cuda_warmup(runtime):
    case=runtime.case(runtime.data.split['training_ids'][0]);start=time.perf_counter()
    with torch.no_grad():
        for count in ([7225] if runtime.conv else [722,2709,3612,5419,7225]):
            ids=np.arange(count)
            for _ in range(5):runtime.forward(case,None if count==7225 else ids)
    torch.cuda.synchronize();return time.perf_counter()-start


def paired(gpu):
    folder=OUT/'timing';folder.mkdir(exist_ok=True);references=reference_values();lock=read(OUT/'config/evaluation_lock.json')
    assert reconstruction_module.IMPLEMENTATION_VERSION==lock['implementation_version']=='1.1'
    assert reconstruction_module.IMPLEMENTATION_SOURCE_SHA256==lock['implementation_source_sha256']
    with ExclusiveGPU(gpu,folder/'gpu_telemetry.json',interval=1.) as monitor:
        model_start=time.perf_counter()
        models=[Runtime('Transolver3'),Runtime('LinearNO'),Runtime('LinearNO',conv=True)]
        setup=dict(model_loading_seconds=time.perf_counter()-model_start,kernel_warmup_seconds={m.label:cuda_warmup(m) for m in models},
                   source_files={name:digest(Path(__file__).with_name(name)) for name in ['benchmark.py','benchmark_core.py','inference.py','common.py']},
                   reconstruction_version=reconstruction_module.IMPLEMENTATION_VERSION,reconstruction_sha256=reconstruction_module.IMPLEMENTATION_SOURCE_SHA256,
                   timing_helper_sha256=digest(FOLLOWUP/'reconstruction/code/timing_helper.py'),environment=environment(gpu))
        write(folder/'setup.json',setup)
        input_preparations=[]
        jobs=[]
        for scope in ['A','B','C']:
            for sid in lock['timing_case_ids']:
                jobs.append((scope,sid,'full',1.,None))
                for budget in [.375,.5,.75]:
                    for seed in [0,1,2]:
                        for method in ['uniform','pi','cpg','ca']:
                            jobs.append((scope,sid,method,budget,seed))
        order=np.random.default_rng(20261114).permutation(len(jobs)).tolist()
        write(folder/'scheduled_order.json',dict(seed=20261114,groups=[jobs[i] for i in order],
                    rule='randomized group order; within group cyclic old/new and backbone ordering'))
        for sequence,i in enumerate(order):
            scope,sid,method,budget,seed=jobs[i]
            input_started=time.perf_counter()
            for runtime in models:runtime.cases.clear()
            if method=='full':
                requests={m.label:CompleteRequest(m,sid,method,budget,seed,'new',scope) for m in models}
                meta={name:metadata(r.runtime,sid,method,budget,seed,'full_shared',scope) for name,r in requests.items()}
                group=f'{scope}_case{sid}_full'
            else:
                requests={m.label+'_'+implementation:CompleteRequest(m,sid,method,budget,seed,implementation,scope)
                          for m in models[:2] for implementation in ['old','new']}
                meta={name:metadata(r.runtime,sid,method,budget,seed,r.reconstruction,scope) for name,r in requests.items()}
                group=f'{scope}_case{sid}_{method}_s{seed}_b{budget:.3f}'
            torch.cuda.synchronize()
            input_preparations.append(dict(group_id=group,setup_ms=(time.perf_counter()-input_started)*1000,
                scope='CPU array loading/normalization and request-object assembly, plus GPU input residency setup for B; outside model-ready request boundary',
                loaded_backbones=3 if method=='full' else 2))
            write(folder/'input_preparation.json',input_preparations)
            run_group(folder/'groups',group,requests,meta,scope=SCOPES[scope],
                check_output=lambda name,out:check_output(requests[name],out,references),identity_of=lambda name,r:identity(r),
                guard=monitor.assert_exclusive,extra_profiles=lambda reqs:{name:profile_request(r) for name,r in reqs.items()})
            print('completed',sequence+1,len(order),group,flush=True)
            del requests;gc.collect();torch.cuda.empty_cache()
    write(folder/'paired_complete.json',dict(execution_status='completed',groups=333,physical_conditions=1323,
        formal_repeats=132300,warmups=26460,Full_alias='old/new views share one actual Full series in each scope; convFull has no sampling variants'))


def memory(gpu,backbone,method,conv=False):
    folder=OUT/'timing/memory';folder.mkdir(parents=True,exist_ok=True)
    lock=read(OUT/'config/evaluation_lock.json');references=reference_values()
    label='LinearNO_conv_temp' if conv else 'LinearNO_point_temp' if backbone=='LinearNO' else 'Transolver3'
    key=label+'_'+method;file=folder/(key+'.json')
    if file.exists():raise FileExistsError(file)
    with ExclusiveGPU(gpu,folder/(key+'_gpu.json'),interval=1.) as monitor:
        runtime=Runtime(backbone,conv=conv,require_head=method=='ca',load_head=method=='ca');rows=[]
        for scope in ['A','B','C']:
            for sid in lock['timing_case_ids']:
                runtime.cases.clear()
                for budget in ([1.] if method=='full' else [.375,.5,.75]):
                    for seed in ([None] if method=='full' else [0,1,2]):
                        for implementation in (['full_shared'] if method=='full' else ['old','new']):
                            request=CompleteRequest(runtime,sid,method,budget,seed,'new' if implementation=='full_shared' else implementation,scope)
                            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                            first_output=request();torch.cuda.synchronize()
                            cold=dict(first_request_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      first_request_peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                            check_output(request,first_output,references);del first_output;monitor.assert_exclusive()
                            def nvml_sample():
                                result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                                return {'device_used_MiB':int(result.stdout.strip())} if result.returncode==0 else {'available':False}
                            measured=isolated_memory(request,warmups=2,repeats=2,nvml_post_sample=nvml_sample)
                            rows.append(dict(**metadata(runtime,sid,method,budget,seed,implementation,scope),**measured,**cold,
                                              cached_map_cpu_bytes=request.last['plan_cpu_bytes'],cached_map_gpu_bytes=request.last['plan_device_bytes']))
                            del request;gc.collect();torch.cuda.empty_cache()
                print('memory',key,scope,sid,flush=True)
    write(file,dict(execution_status='completed',model=runtime.label,method=method,rows=rows,
                    scope='single architecture/method process; PyTorch allocator peaks; CPU SciPy and transient external GPU/context peaks not claimed'))


def environment(gpu):
    import scipy
    def command(args):
        result=subprocess.run(args,capture_output=True,text=True);return result.stdout.strip() if result.returncode==0 else result.stderr.strip()
    return dict(python=platform.python_version(),torch=torch.__version__,scipy=scipy.__version__,numpy=np.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_index=gpu,
                gpu=command(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,driver_version,memory.total','--format=csv']),
                cpu=command(['lscpu']),cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=torch.get_num_threads(),
                cpu_load_average=os.getloadavg(),processes=command(['ps','-eo','pid,psr,pcpu,pmem,comm','--sort=-pcpu']),
                fp32_network=True,tf32=False,mode='eager',memory_statistics='separate per-model/method subprocess')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['paired','memory'],required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--backbone',default='Transolver3');p.add_argument('--method',default='full');p.add_argument('--conv-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=read(OUT/'config/evaluation_lock.json')
    if reconstruction_module.IMPLEMENTATION_VERSION!=lock['implementation_version'] or reconstruction_module.IMPLEMENTATION_SOURCE_SHA256!=lock['implementation_source_sha256']:
        raise RuntimeError('Reconstruction module differs from accepted protocol')
    if a.phase=='paired':paired(a.gpu)
    else:memory(a.gpu,a.backbone,a.method,a.conv_only)

if __name__=='__main__':main()
