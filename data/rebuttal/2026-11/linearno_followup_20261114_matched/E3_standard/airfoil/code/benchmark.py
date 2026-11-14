"""Same empty GPU, balanced old/new/backbone complete field timing and memory."""
import os
for _thread_variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_thread_variable]='4'
from common import *
import argparse,csv,gc,platform,subprocess,time,shutil
from evaluate import attach_reconstruction,metrics
from runtime import Runtime,configurations
from request import Request

api,gate=attach_reconstruction()
from benchmark_core import run_group
from timing_helper import isolated_memory,timed_request
sys.path.append(str(CURRENT/'AdaSolver_official'))
from inference_speed.gpu_monitor import ExclusiveGPU

LABELS={'t3':'Transolver3','linearno':'LinearNO_point_temp','linearno_conv':'LinearNO_conv_temp_fullgrid'}
SCOPES={'A':'A_cpu_cpu','B':'B_gpu_gpu','C':'C_cpu_cpu_cached_map'}
CASES=[1001,1094,1198]
CODE_SOURCE_BYTES={name:(OUT/'code'/name).read_bytes() for name in ['benchmark.py','benchmark_core.py','request.py','runtime.py','evaluate.py','common.py']}
CODE_IDENTITIES={name:hashlib.sha256(value).hexdigest() for name,value in CODE_SOURCE_BYTES.items()}
SOURCE_SNAPSHOT_ID=hashlib.sha256(json.dumps(CODE_IDENTITIES,sort_keys=True).encode()).hexdigest()


def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def preserve_previous(path):
    if path.exists():
        history=path.parent/'history';history.mkdir(exist_ok=True)
        shutil.copyfile(path,history/f'{path.stem}.{time.time_ns()}{path.suffix}')


def identity(request):
    coarse_prediction_hash=None
    if request.method in ['cpg','cpg_beta0']:
        key=(request.sid,request.seed,len(request.last_coarse),None,request.config['operator_stage1_cache'])
        coarse_prediction_hash=array_hash(request.runtime.coarse_cache[key][0])
    return dict(selected_indices_hash=array_hash(np.asarray(request.last_indices,np.int64)),
        coarse_indices_hash=None if request.last_coarse is None else array_hash(np.asarray(request.last_coarse,np.int64)),
        compensation_hash=None if request.last_compensation is None else array_hash(request.last_compensation),
        coarse_prediction_hash=coarse_prediction_hash)


def metadata(request):
    rt=request.runtime;method=request.method
    return dict(dataset='airfoil2d',backbone=LABELS[rt.architecture],case_id=request.sid,method=method,
        beta=None if request.config is None or method=='uniform' else request.config['beta'],
        budget_fraction=request.budget,sampling_seed=request.seed,training_seed=0,head_seed=0 if method=='ca' else None,
        checkpoint_id=rt.operator_hash,normalizer_id=rt.data.normalizer_sha256,
        head_checkpoint_sha256=rt.head(request.config)[1] if method=='ca' else None,
        implementation='full_shared' if method=='full' else request.implementation,scope=request.scope,
        reconstruction_id=request.reconstruction_id,metric_target='field',test_population='published_cl_gt_0p02_110',
        reconstruction_sha256=api.IMPLEMENTATION_SOURCE_SHA256,source_snapshot_id=SOURCE_SNAPSHOT_ID,
        request_code_sha256=CODE_IDENTITIES['request.py'],runtime_code_sha256=CODE_IDENTITIES['runtime.py'],
        benchmark_code_sha256=CODE_IDENTITIES['benchmark.py'],timing_core_sha256=CODE_IDENTITIES['benchmark_core.py'],
        output_device='cuda' if request.scope=='B_gpu_gpu' else 'cpu',output_dtype='float64',
        first_preparation='model/head loading and native kernel warmup separate; first request builds actual geometry/map',
        reuse_assumption='same physical geometry and exact selected order; only interpolation map cached; coarse/score/selection rerun' if request.scope=='C_cpu_cpu_cached_map' else 'none')


def reference_values():
    result={}
    for arch in LABELS:
        completion=read(OUT/f'results/{arch}_complete_completion.json')
        assert read(OUT/f'results/{arch}_accuracy_acceptance.json')['passed']
        for name in completion['source_files']:
            source=Path(name)
            for line in source.open():
                row=json.loads(line)
                if row['case_id'] not in CASES:continue
                result[(arch,row['case_id'],row['method'],row['sampling_seed'],row['budget_fraction'])]=(row,source.parent/row['arrays'])
    return result


def check_output(request,output,references):
    if request.scope=='B_gpu_gpu':
        assert torch.is_tensor(output) and output.dtype==torch.float64 and output.device.type=='cuda'
        observed=output.cpu().numpy()
    else:
        assert isinstance(output,np.ndarray) and output.dtype==np.float64
        observed=output
    assert observed.shape==(11271,1) and np.isfinite(observed).all()
    row,array_path=references[(request.runtime.architecture,request.sid,request.method,request.seed,request.budget)]
    assert row['checkpoint_sha256']==request.runtime.operator_hash
    assert row['normalizer_sha256']==request.runtime.data.normalizer_sha256
    if request.method=='ca':assert row['head_sha256']==request.runtime.head(request.config)[1]
    if request.method in ['cpg','cpg_beta0']:
        assert identity(request)['coarse_prediction_hash']==row['coarse_prediction_hash']
    with np.load(array_path) as arrays:
        if request.method=='full':expected=arrays['field']
        else:
            assert np.array_equal(request.last_indices,arrays['indices'])
            assert np.array_equal(np.asarray([] if request.last_coarse is None else request.last_coarse),arrays['coarse_indices'])
            if request.method!='uniform':assert np.array_equal(request.last_compensation,arrays['compensation'])
            expected=api.native_reconstruct('airfoil',arrays['indices'],arrays['prediction'],request.reference)
    delta=np.abs(observed-expected);allowed=1e-6+1e-5*np.abs(expected)
    if np.any(delta>allowed):raise RuntimeError('timed full field differs from accepted fixed prediction/index artifact')
    rt=request.runtime;field_error=metrics(observed,request.coords,rt.data.target(request.sid).astype(np.float64))['field_relative_l2']
    original=row['old' if request.implementation=='old' else 'new']['field_relative_l2']
    if abs(field_error-original)>1e-8+1e-5*abs(original):raise RuntimeError('timed downstream field metric differs')
    return dict(max_absolute_field_difference=float(delta.max()),relative_field_norm_difference=float(np.linalg.norm(delta)/np.linalg.norm(expected)),
        field_error=field_error,reference_field_error=original,retained_points=len(request.last_indices),
        cached_map_cpu_bytes=request.last_plan.cpu_bytes if request.last_plan is not None else 0,
        cached_map_gpu_bytes=request.last_plan.device_bytes() if request.last_plan is not None else 0,
        source_arrays=str(array_path),**identity(request))


@torch.no_grad()
def profile_request(request,repeats=5):
    """Separate decomposition with nested rows; never substituted for direct time."""
    rows=[]
    def measure(label,fn,repeat):
        value,ms=timed_request(fn);rows.append(dict(profile_repeat=repeat,component=label,ms=ms,timing_scope='separate instrumented pass'))
        return value
    if request.scope=='C_cpu_cpu_cached_map' and request.method!='full':
        request.runtime.clear_dynamic();sampled=request._prepare_sampling(request.coords)
        cold=measure('first_cache_map_build',lambda:api.build_plan('airfoil',request.reference,sampled[0].indices,
            channels=1,device=request.runtime.device if request.implementation=='new' else None,profile=True),-1)
        rows[-1].update(cache_cpu_bytes=cold.cpu_bytes,cache_gpu_bytes=cold.device_bytes(),
            timing_scope='separate cold map build and transfer, not part of steady cached application')
        del cold
    for repeat in range(repeats):
        rt=request.runtime;rt.clear_dynamic()
        if request.method=='full':
            pred=measure('full_forward_decode_input_transfer',lambda:request._prediction(np.arange(11271),None,request.coords),repeat)
            measure('output_conversion_transfer',lambda:pred.to(torch.float64) if request.scope=='B_gpu_gpu' else pred.cpu().numpy().astype(np.float64),repeat)
            rows.append(dict(profile_repeat=repeat,component='reconstruction_ms',ms=0.,timing_scope='separate instrumented pass'));continue
        coords=request.coords
        if request.scope=='B_gpu_gpu' and request.method!='uniform':coords=measure('geometry_d2h',lambda:request.gpu_coords.cpu().numpy(),repeat)
        (sampling,coarse,_)=measure('sampling_with_acquisition',lambda:request._prepare_sampling(coords,instrumented=True),repeat)
        for key in ['coarse_sampling_seconds','coarse_seconds','head_seconds','feature_seconds','refinement_seconds']:
            rows.append(dict(profile_repeat=repeat,component='nested_'+key.replace('_seconds','_ms'),ms=float(sampling.metadata.get(key,0))*1000,
                timing_scope='nested sampler profile, overlaps parent'))
        comp=None if request.method=='uniform' else sampling.compensation
        pred=measure('final_forward_decode_input_transfer_compensation',lambda:request._prediction(sampling.indices,comp,coords),repeat)
        measure('reconstruction_complete_output',lambda:request._reconstruct(sampling.indices,pred),repeat)
        if request.scope=='C_cpu_cpu_cached_map':
            plan,hit=measure('cache_key_lookup',lambda:request.cache.get('airfoil',request.reference,sampling.indices,
                geometry_identity=request.geometry_id,reconstruction_id=gate['new_id'],channels=1),repeat)
            if request.implementation=='old':
                cpu=measure('prediction_d2h',lambda:pred.cpu().numpy(),repeat)
                measure('cached_sparse_apply',lambda:plan.apply_numpy(cpu),repeat)
            else:
                field=measure('cached_sparse_apply',lambda:plan.apply_tensor(pred),repeat)
                measure('field_d2h',lambda:field.cpu().numpy(),repeat)
            rows.append(dict(profile_repeat=repeat,component='cache_hit',ms=None,cache_hit=hit,timing_scope='cache metadata'))
        elif request.implementation=='old':
            cpu=measure('prediction_d2h',lambda:pred.cpu().numpy(),repeat)
            field=measure('native_build_query_weights_apply',lambda:api.native_reconstruct('airfoil',sampling.indices,cpu,request.reference),repeat)
            if request.scope=='B_gpu_gpu':measure('field_h2d',lambda:torch.as_tensor(field,device=rt.device),repeat)
        else:
            plan=measure('new_plan_build_and_upload',lambda:api.build_plan('airfoil',request.reference,sampling.indices,
                channels=1,device=rt.device,profile=True),repeat)
            rows.append(dict(profile_repeat=repeat,component='coordinate_index_prepare',ms=plan.timings['coordinate_index_prepare_ms'],timing_scope='nested plan profile'))
            for mapping in plan.timings['maps']:
                for key,value in mapping.items():
                    if key.endswith('_ms'):rows.append(dict(profile_repeat=repeat,component='nested_'+key,ms=value,timing_scope='nested plan profile'))
            rows.append(dict(profile_repeat=repeat,component='map_h2d',ms=plan.timings['map_h2d_ms'],timing_scope='nested plan profile'))
            field=measure('gpu_sparse_apply',lambda:plan.apply_tensor(pred),repeat)
            if request.scope=='A_cpu_cpu':measure('field_d2h',lambda:field.cpu().numpy(),repeat)
    return rows


@torch.no_grad()
def cuda_warmup(runtime):
    start=time.perf_counter();coords=runtime.data.coords(read(SPLITS)['training_ids'][0])
    for count in ([11271] if runtime.architecture=='linearno_conv' else [1127,4227,5636,8453,11271]):
        pos=torch.as_tensor(coords[:count],device=runtime.device)[None]
        for _ in range(5):runtime.model(pos,pos)
    if runtime.architecture!='linearno_conv':
        cfg=next(c for c in configurations(runtime.architecture) if c['name']=='ca');head=runtime.head(cfg)[0]
        pos=torch.as_tensor(coords[:1127],device=runtime.device)[None]
        for _ in range(5):head.forward_with_cache(pos,pos)
    torch.cuda.synchronize();runtime.clear_dynamic();return time.perf_counter()-start


@torch.no_grad()
def reconstruction_warmup(runtime):
    """Warm libraries on discarded training-case maps; no timing-case map survives."""
    sid=read(SPLITS)['training_ids'][0];coords=runtime.data.coords(sid)
    indices=np.random.default_rng(20261114).choice(11271,4227,replace=False)
    pos=torch.as_tensor(coords[indices],device=runtime.device)[None]
    prediction=runtime.data.decode(runtime.model(pos,pos))[0]
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(3):
        api.native_reconstruct('airfoil',indices,prediction.cpu().numpy(),runtime.reference)
        plan=api.build_plan('airfoil',runtime.reference,indices,channels=1)
        output=plan.apply_tensor(prediction).cpu().numpy()
        del output,plan
    torch.cuda.synchronize();return time.perf_counter()-start


def paired(gpu):
    folder=OUT/'timing';folder.mkdir(exist_ok=True);references=reference_values()
    for name in ['gpu_telemetry.json','setup.json','scheduled_order.json']:preserve_previous(folder/name)
    with ExclusiveGPU(gpu,folder/'gpu_telemetry.json',interval=1.) as monitor:
        start=time.perf_counter();models=[Runtime(arch) for arch in LABELS]
        loading_seconds=time.perf_counter()-start;head_loading={}
        for runtime in models[:2]:
            start=time.perf_counter();runtime.head(next(c for c in configurations(runtime.architecture) if c['name']=='ca'))
            head_loading[runtime.architecture]=time.perf_counter()-start
        setup=dict(physical_gpu=gpu,model_loading_seconds=loading_seconds,head_loading_seconds=head_loading,
            kernel_warmup_seconds={r.architecture:cuda_warmup(r) for r in models},
            reconstruction_warmup_including_discarded_training_maps_seconds=reconstruction_warmup(models[0]),
            warmup_map_policy='three old/new warmups on an unused training geometry/set; all maps discarded; actual request maps built inside request',
            torch_version=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),
            cpu_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_name=torch.cuda.get_device_name(0),
            gpu_compute_capability=torch.cuda.get_device_capability(0),network_dtype='float32',tf32=False,
            reconstruction_version=api.IMPLEMENTATION_VERSION,reconstruction_sha256=api.IMPLEMENTATION_SOURCE_SHA256,
            sources=CODE_IDENTITIES,source_snapshot_id=SOURCE_SNAPSHOT_ID,
            timing_helper_sha256=sha(Path(gate['module']).parent/'timing_helper.py'),environment=environment(gpu))
        write(folder/'setup.json',setup)
        snapshot=folder/'sources'/SOURCE_SNAPSHOT_ID;snapshot.mkdir(parents=True,exist_ok=True)
        for name,value in CODE_SOURCE_BYTES.items():(snapshot/name).write_bytes(value)
        configs={r.architecture:{c['name']:c for c in configurations(r.architecture)} for r in models[:2]}
        jobs=[]
        for scope in SCOPES.values():
            for sid in CASES:
                jobs.append((scope,sid,'full',1.,None))
                jobs.extend((scope,sid,method,budget,seed) for budget in [.375,.5,.75] for seed in [0,1,2] for method in ['uniform','pi','cpg','ca'])
        order=np.random.default_rng(20261114).permutation(len(jobs)).tolist()
        write(folder/'scheduled_order.json',dict(seed=20261114,groups=[jobs[i] for i in order],
            rule='randomized group order; within group balanced cyclic old/new and backbone ordering'))
        for sequence,i in enumerate(order):
            scope,sid,method,budget,seed=jobs[i]
            if method=='full':
                requests={LABELS[r.architecture]:Request(r,api,gate,sid,None,1.,None,'new',scope) for r in models}
                group=f'{scope}_case{sid}_full'
            else:
                requests={LABELS[r.architecture]+'_'+mode:Request(r,api,gate,sid,seed,budget,configs[r.architecture][method],mode,scope)
                          for r in models[:2] for mode in ['old','new']}
                group=f'{scope}_case{sid}_{method}_s{seed}_b{budget:.3f}'
            run_group(folder/'groups',group,requests,{name:metadata(r) for name,r in requests.items()},scope=scope,
                check_output=lambda name,out:check_output(requests[name],out,references),identity_of=lambda name,r:identity(r),
                guard=monitor.assert_exclusive,extra_profiles=lambda reqs:{name:profile_request(r) for name,r in reqs.items()})
            write(folder/'progress.json',dict(completed_groups=sequence+1,expected_groups=333,last_group=group))
            print('completed',sequence+1,len(order),group,flush=True)
            del requests;gc.collect();torch.cuda.empty_cache()
    write(folder/'paired_complete.json',dict(execution_status='completed',groups=333,physical_conditions=1323,
        formal_repeats=132300,warmups=26460,Full_alias='old/new views share one actual Full series per scope; conv has Full only'))


def memory(gpu,architecture,method):
    folder=OUT/'timing/memory';folder.mkdir(parents=True,exist_ok=True);key=architecture+'_'+method;path=folder/(key+'.json')
    if path.exists():
        if read(path).get('execution_status')=='completed':return
        raise FileExistsError(path)
    references=reference_values();rows=[]
    preserve_previous(folder/(key+'_gpu.json'))
    with ExclusiveGPU(gpu,folder/(key+'_gpu.json'),interval=1.) as monitor:
        runtime=Runtime(architecture)
        cfg=None if method=='full' else next(c for c in configurations(architecture) if c['name']==method)
        if method=='ca':runtime.head(cfg)
        for scope in SCOPES.values():
            for sid in CASES:
                for budget in ([1.] if method=='full' else [.375,.5,.75]):
                    for seed in ([None] if method=='full' else [0,1,2]):
                        for mode in (['new'] if method=='full' else ['old','new']):
                            request=Request(runtime,api,gate,sid,seed,budget,cfg,mode,scope)
                            request.collect_memory=True
                            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                            first=request();torch.cuda.synchronize()
                            cold=dict(first_request_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                first_request_peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                            check_output(request,first,references);del first;monitor.assert_exclusive()
                            def sample():
                                state=monitor.snapshot();return dict(device_used_MiB=state['gpu']['memory.used'],processes=state['processes'])
                            measured=isolated_memory(request,warmups=2,repeats=2,nvml_post_sample=sample)
                            rows.append(dict(**metadata(request),**measured,**cold,
                                reconstruction_map_cpu_bytes=request.last_map_cpu_bytes,reconstruction_map_gpu_bytes=request.last_map_gpu_bytes,
                                cached_map_cpu_bytes=request.last_map_cpu_bytes if scope.startswith('C_') else 0,
                                cached_map_gpu_bytes=request.last_map_gpu_bytes if scope.startswith('C_') else 0,cache_hit=request.last_cache_hit,
                                map_byte_scope='numerical neighbor/weight/denominator/exact-slot/target-row arrays; Python and shared reference metadata excluded'))
                            del request;gc.collect();torch.cuda.empty_cache();monitor.assert_exclusive()
                print('memory',key,scope,sid,flush=True)
    write(path,dict(execution_status='completed',architecture=architecture,method=method,rows=rows,
        scope='independent architecture/method process; PyTorch allocator peaks and synchronized NVML observations; external transient peaks not certified'))


def environment(gpu):
    import scipy
    def command(args):
        result=subprocess.run(args,capture_output=True,text=True);return result.stdout.strip() if result.returncode==0 else result.stderr.strip()
    return dict(python=platform.python_version(),torch=torch.__version__,scipy=scipy.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_index=gpu,
        gpu=command(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,driver_version,memory.total','--format=csv']),
        cpu=command(['lscpu']),cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=torch.get_num_threads(),
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        cpu_load_average=os.getloadavg(),processes=command(['ps','-eo','pid,psr,pcpu,pmem,comm','--sort=-pcpu']),
        fp32_network=True,tf32=False,mode='native_eager',memory_statistics='independent per-model/method process')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['paired','memory'],required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--architecture',choices=list(LABELS),default='t3');p.add_argument('--method',default='full');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.phase=='paired':paired(a.gpu)
    else:memory(a.gpu,a.architecture,a.method)


if __name__=='__main__':main()
