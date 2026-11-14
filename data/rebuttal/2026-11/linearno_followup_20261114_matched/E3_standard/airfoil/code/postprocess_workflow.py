"""Durable accepted-accuracy -> empty-GPU timing -> isolated memory -> local report."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
OUT=Path(__file__).resolve().parents[1];CODE=OUT/'code'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {}


def write(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def wait_for_gpu(state,path,gpu,stage):
    idle_since=None
    while True:
        try:
            r=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            pids=[int(line.strip()) for line in r.stdout.splitlines() if line.strip().isdigit()];known=r.returncode==0
        except (OSError,subprocess.TimeoutExpired):pids=[];known=False
        profile_lock_available=False
        with open(f'/tmp/adasolver_profile_gpu_{gpu}.lock','a') as lease:
            try:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);profile_lock_available=True
                fcntl.flock(lease,fcntl.LOCK_UN)
            except BlockingIOError:pass
        if known and not pids and profile_lock_available:
            idle_since=idle_since or time.monotonic()
            if time.monotonic()-idle_since>=30:return
        else:idle_since=None
        state.update(execution_status='waiting_resource',active_stage=stage,observed_gpu_pids=pids,
                     profile_lock_available=profile_lock_available,heartbeat=time.time())
        write(path,state);time.sleep(5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=2);args=parser.parse_args()
    lock=(OUT/'runtime/latency_workflow.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'runtime/latency_workflow.json';state=read(path) or dict(stages={})
    state.update(pid=os.getpid(),execution_status='waiting_dependency',evidence_status='pending',physical_gpu=args.gpu)
    while True:
        dependency={arch:read(OUT/f'results/{arch}_accuracy_acceptance.json').get('passed',False) for arch in ['t3','linearno','linearno_conv']}
        if all(dependency.values()):
            state.update(dependencies=dependency);write(path,state);break
        state.update(dependencies=dependency,heartbeat=time.time());write(path,state);time.sleep(10)
    stages=[('accuracy_aggregation','aggregate.py',[],False),('paired_latency','benchmark.py',['--phase','paired','--gpu',str(args.gpu)],True)]
    for arch in ['t3','linearno']:
        for method in ['full','uniform','pi','cpg','ca']:
            stages.append((f'memory_{arch}_{method}','benchmark.py',['--phase','memory','--architecture',arch,'--method',method,'--gpu',str(args.gpu)],True))
    stages.extend([('memory_conv_full','benchmark.py',['--phase','memory','--architecture','linearno_conv','--method','full','--gpu',str(args.gpu)],True),
        ('latency_aggregation','finalize_latency.py',[],False),('local_report','report_results.py',[],False)])
    for name,script,arguments,gpu_stage in stages:
        if state['stages'].get(name,{}).get('execution_status')=='completed':continue
        if not (CODE/script).exists():raise FileNotFoundError(script)
        if gpu_stage:wait_for_gpu(state,path,args.gpu,name)
        cmd=[sys.executable,'-u',str(CODE/script),*arguments];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        log=OUT/f'logs/postprocess_{name}.log'
        with log.open('a') as stream:
            child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=OUT)
            state['stages'][name]=dict(execution_status='running',pid=child.pid,command=cmd,log=str(log),started_at=time.time())
            state.update(execution_status='running',active_stage=name);write(path,state)
            while child.poll() is None:state['heartbeat']=time.time();write(path,state);time.sleep(10)
        state['stages'][name].update(execution_status='completed' if child.returncode==0 else 'failed',returncode=child.returncode,finished_at=time.time())
        write(path,state)
        if child.returncode:state.update(execution_status='failed',error=f'{name} returned {child.returncode}');write(path,state);return
    state.update(execution_status='completed',active_stage=None,evidence_status='awaiting_cross_dataset_independent_review',finished_at=time.time());write(path,state)


if __name__=='__main__':main()
