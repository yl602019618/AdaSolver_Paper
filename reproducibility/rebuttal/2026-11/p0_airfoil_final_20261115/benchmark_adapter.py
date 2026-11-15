#!/usr/bin/env python3
"""P0 matrix adapter over immutable native Airfoil request/scoring code.

Only the job matrix, output destination and source receipts are new. This
adapter admits its own physical GPU lease, includes CPG beta0, and leaves
running training/evaluation sources untouched.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

METHODS=('full','uniform','pi','cpg_beta0','cpg','ca')
SCOPES=('A_cpu_cpu','B_gpu_gpu','C_cpu_cpu_cached_map')
CASES=(1001,1094,1198)
BUDGETS=(.375,.5,.75)
LABELS={'t3':'Transolver3','linearno':'LinearNO_point_temp'}


def jobs(methods):
    return [(scope,case,method,budget,sampling_seed)
        for scope in SCOPES for case in CASES for method in methods
        for budget in ((1.,) if method=='full' else BUDGETS)
        for sampling_seed in ((None,) if method=='full' else (0,1,2))]


def planned(methods):
    j=jobs(methods);conditions=sum(2 if method=='full' else 4 for _,_,method,_,_ in j)
    return dict(methods=list(methods),groups=len(j),physical_conditions=conditions,
        formal_repeats=conditions*100,warmups=conditions*20,training_replicates=1,
        full_sampling_seed=None,scopes=list(SCOPES),cases=list(CASES),budgets=list(BUDGETS))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def load_native(seed_root,output,seed):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(seed_root/'code'))
    spec=importlib.util.spec_from_file_location('p0_native_airfoil_benchmark',seed_root/'code/benchmark.py')
    native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    native.LABELS=dict(LABELS)
    source_bytes=dict(native.CODE_SOURCE_BYTES)
    source_bytes['p0_benchmark_adapter.py']=Path(__file__).read_bytes()
    source_bytes['p0_gpu_monitor.py']=Path(__file__).with_name('gpu_monitor.py').read_bytes()
    if (seed_root/'code/seed_context.py').exists():source_bytes['seed_context.py']=(seed_root/'code/seed_context.py').read_bytes()
    native.CODE_SOURCE_BYTES=source_bytes
    native.CODE_IDENTITIES={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    native.SOURCE_SNAPSHOT_ID=hashlib.sha256(json.dumps(native.CODE_IDENTITIES,sort_keys=True).encode()).hexdigest()
    native.OUT=output
    from gpu_monitor import ExclusiveGPU
    native.ExclusiveGPU=ExclusiveGPU
    def references():
        result={}
        for arch in LABELS:
            receipt=json.loads((seed_root/f'results/{arch}_accuracy_acceptance.json').read_text())
            assert receipt['passed']
            model=json.loads((seed_root/f'checkpoints/{arch}/manifest.json').read_text())
            assert model['training_seed']==seed
            assert model['checkpoint_sha256']==receipt['checkpoint_sha256']
            completion=json.loads((seed_root/f'results/{arch}_complete_completion.json').read_text())
            for source in completion['source_files']:
                path=Path(source)
                for line in path.open():
                    row=json.loads(line)
                    if row['case_id'] in CASES:
                        result[(arch,row['case_id'],row['method'],row['sampling_seed'],row['budget_fraction'])]=(row,path.parent/row['arrays'])
        return result
    native.reference_values=references
    original_meta=native.metadata
    def metadata(request):
        result=original_meta(request)
        assert result['training_seed']==seed
        result.update(p0_protocol='airfoil_three_training_seeds_beta0_timing_v1',
            p0_adapter_sha256=native.CODE_IDENTITIES['p0_benchmark_adapter.py'],
            source_seed_root=str(seed_root))
        return result
    native.metadata=metadata
    return native


def paired(native,gpu,methods):
    torch,np=native.torch,native.np
    folder=native.OUT/'timing';folder.mkdir(parents=True,exist_ok=True)
    expected=planned(methods);references=native.reference_values()
    for name in ['gpu_telemetry.json','setup.json','scheduled_order.json']:native.preserve_previous(folder/name)
    with native.ExclusiveGPU(gpu,folder/'gpu_telemetry.json',interval=1.) as monitor:
        start=time.perf_counter();models=[native.Runtime(arch) for arch in LABELS]
        loading=time.perf_counter()-start;head_loading={}
        for runtime in models:
            t=time.perf_counter();runtime.head(next(c for c in native.configurations(runtime.architecture) if c['name']=='ca'))
            head_loading[runtime.architecture]=time.perf_counter()-t
        setup=dict(physical_gpu=gpu,model_loading_seconds=loading,head_loading_seconds=head_loading,
            kernel_warmup_seconds={r.architecture:native.cuda_warmup(r) for r in models},
            reconstruction_warmup_including_discarded_training_maps_seconds=native.reconstruction_warmup(models[0]),
            network_dtype='float32',output_dtype='float64',tf32=False,cpu_threads=torch.get_num_threads(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),sources=native.CODE_IDENTITIES,
            source_snapshot_id=native.SOURCE_SNAPSHOT_ID,reconstruction_sha256=native.api.IMPLEMENTATION_SOURCE_SHA256,
            expected=expected,environment=native.environment(gpu))
        write(folder/'setup.json',setup)
        snapshot=folder/'sources'/native.SOURCE_SNAPSHOT_ID;snapshot.mkdir(parents=True,exist_ok=True)
        for name,content in native.CODE_SOURCE_BYTES.items():(snapshot/name).write_bytes(content)
        configs={r.architecture:{c['name']:c for c in native.configurations(r.architecture)} for r in models}
        queue=jobs(methods);order=np.random.default_rng(20261115).permutation(len(queue)).tolist()
        write(folder/'scheduled_order.json',dict(seed=20261115,groups=[queue[i] for i in order],rule='randomized P0 groups; native cyclic within-group order'))
        for sequence,i in enumerate(order):
            scope,sid,method,budget,sampling_seed=queue[i]
            if method=='full':
                requests={LABELS[r.architecture]:native.Request(r,native.api,native.gate,sid,None,1.,None,'new',scope) for r in models}
                group=f'{scope}_case{sid}_full'
            else:
                requests={LABELS[r.architecture]+'_'+mode:native.Request(r,native.api,native.gate,sid,sampling_seed,budget,configs[r.architecture][method],mode,scope) for r in models for mode in ['old','new']}
                group=f'{scope}_case{sid}_{method}_s{sampling_seed}_b{budget:.3f}'
            native.run_group(folder/'groups',group,requests,{name:native.metadata(r) for name,r in requests.items()},scope=scope,
                check_output=lambda name,output:native.check_output(requests[name],output,references),
                identity_of=lambda name,request:native.identity(request),guard=monitor.assert_exclusive,
                extra_profiles=lambda reqs:{name:native.profile_request(r) for name,r in reqs.items()})
            write(folder/'progress.json',dict(completed_groups=sequence+1,expected_groups=len(queue),last_group=group))
            print('completed',sequence+1,len(queue),group,flush=True)
            del requests;native.gc.collect();torch.cuda.empty_cache()
    write(folder/'paired_complete.json',dict(execution_status='completed',**expected,Full_alias='one actual series per point model/case/scope; no synthetic sampling seed'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--phase',choices=['paired','memory'],default='paired');p.add_argument('--gpu',type=int)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--architecture',choices=list(LABELS),default='t3');p.add_argument('--method',choices=METHODS,default='full')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--probe-native',action='store_true')
    a=p.parse_args();seed_root=a.seed_root.resolve();output=a.output_dir.resolve()
    if a.plan_only:
        proof=planned(a.methods)
        if a.probe_native:
            os.environ['CUDA_VISIBLE_DEVICES']=''
            native=load_native(seed_root,output,a.seed)
            for arch in LABELS:
                configs={c['name']:c for c in native.configurations(arch)}
                assert set(METHODS)-{'full'}<=set(configs)
                assert configs['cpg_beta0']['beta']==0 and configs['cpg']['beta']==1
            proof.update(native_imports=True,cpg_beta0_config=True,CUDA_visible_devices='',request_source_sha256=native.CODE_IDENTITIES['request.py'],runtime_source_sha256=native.CODE_IDENTITIES['runtime.py'])
        write(output/'plan.json',proof);print(json.dumps(proof,indent=2));return
    if a.gpu is None:p.error('--gpu is required for execution')
    if len(set(a.methods))!=len(a.methods):p.error('duplicate method')
    # The controller does not hold this lease. This process owns it until exit.
    lease=open(f'/tmp/adasolver_profile_gpu_{a.gpu}.lock','a')
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(75)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ADASOLVER_PROFILE_LOCK_FD=str(lease.fileno()),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    native=load_native(seed_root,output,a.seed)
    native.torch.set_num_threads(4);native.torch.backends.cuda.matmul.allow_tf32=False;native.torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True,exist_ok=True)
    write(output/'adapter_binding.json',dict(seed=a.seed,seed_root=str(seed_root),phase=a.phase,methods=a.methods,architecture=a.architecture,method=a.method,
        expected=planned(a.methods),source_snapshot_id=native.SOURCE_SNAPSHOT_ID,sources=native.CODE_IDENTITIES,
        source_unchanged=True,lock_policy='adapter owns profile flock; child-local NVML monitor verifies inherited inode and exclusivity'))
    try:
        if a.phase=='paired':paired(native,a.gpu,a.methods)
        else:native.memory(a.gpu,a.architecture,a.method)
    finally:
        native.torch.cuda.empty_cache();lease.close()


if __name__=='__main__':main()
