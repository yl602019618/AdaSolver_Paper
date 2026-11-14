"""Persistent, ownership-aware queue. It never terminates other GPU jobs."""
from __future__ import annotations
import json,os,subprocess,time,traceback,datetime,fcntl
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
CURRENT=OUT.parents[1]
GPU=6


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {}


def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
    if p.name=='car_workflow.json' and value.get('execution_status')!='completed':
        accuracy=read(OUT/'runtime/car_accuracy.json')
        message=("Reconstruction follow-up — current execution\n\n"
            f"Updated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
            f"Controller PID: {value.get('pid')}; status: {value.get('execution_status')}; stage: {value.get('active_stage')}\n"
            f"Car accuracy pairs: {accuracy.get('completed_pairs',0)}/7400; formal timing needs330 unique conditions per scope.\n\n"
            "Completed: shared API v1.1;6 contract cases;108 real GPU fixed-prediction pairs;12 CPU stage fixtures;16 CPU request-boundary A/C regressions.\n"
            "These validation records do not replace full test accuracy or formal GPU latency.\n\n"
            "The queue admits only an idle GPU6 and never stops external jobs. A/B/C use20 warmups and100 formal repeats; each condition retains its source/hash and scope.\n"
            "Scorer/sampler geometry work is inside every request; only C may reuse interpolation maps.\n"
            "runtime/car_workflow.json records child processes, commands and logs. config/car_protocol.json is the locked plan.\n"
            "No remote push, manuscript overwrite, main update or Feishu modification.\n")
        (OUT/'read.me').write_text(message)


def gpu_available():
    g=subprocess.run(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if g.returncode or p.returncode:return False,{'error':'nvidia-smi failed'}
    uuid,memory,utilization=[x.strip() for x in g.stdout.strip().split(',')]
    pids=[int(line.split(',')[1].strip()) for line in p.stdout.splitlines() if line.split(',')[0].strip()==uuid]
    # The known external training may be loading CPU data before CUDA init.
    ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout.splitlines()
    known_claims=[line.strip() for line in ps if 'rebuttal_20261114_standard_seeds/code/train_backbone.py' in line and f'--gpu {GPU}' in line]
    ok=not pids and not known_claims and int(memory)<=32 and int(utilization)<=1
    return ok,dict(memory_mib=int(memory),utilization=int(utilization),pids=pids,known_external_claims=known_claims)


def main():
    lock=(OUT/'runtime/car_workflow.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(OUT/'config/car_protocol.json');status_path=OUT/'runtime/car_workflow.json'
    state=read(status_path) or dict(execution_status='planned',evidence_status='pending',stages={})
    state.update(pid=os.getpid(),physical_gpu=GPU,cpu_affinity=list(range(128,136)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
             ADASOLVER_NATIVE_CACHE=str(OUT/'runtime/native_fps_cache'))
    jobs=[('cpu_preparation',['prepare_car_runtime.py']),('gpu_pilot',['car_accuracy.py','--pilot']),('accuracy',['car_accuracy.py'])]
    jobs +=[(f'timing_{scope}',['car_timing.py','--scope',scope,'--gpu',str(GPU)]) for scope in ['A','B','C']]
    jobs +=[(f'memory_{arch}_{variant}_{scope}',['car_memory.py','--backbone',arch,'--variant',variant,'--scope',scope,'--gpu',str(GPU)])
            for arch,variants in protocol['timing_variants'].items() for variant in variants for scope in ['A','B','C']]
    jobs += [('aggregate',['aggregate_car.py'])]
    admitted=False
    for name,arguments in jobs:
        if state['stages'].get(name,{}).get('execution_status')=='completed':continue
        if name not in ['aggregate','cpu_preparation']:
            since=None
            while True:
                ready,gpu=gpu_available();now=time.time()
                since=now if ready and since is None else since if ready else None
                state.update(execution_status='waiting_for_gpu',active_stage=name,gpu=gpu,last_update=now)
                write(status_path,state)
                # Initial admission waits for a stable idle card. Subsequent
                # stages still recheck ownership, without a minute per process.
                if ready and now-since>=(5 if admitted else 60):break
                time.sleep(10)
            admitted=True
        log=OUT/'runtime'/f'{name}.log'
        cmd=['taskset','-c','128-135','python3',str(OUT/'code'/arguments[0]),*arguments[1:]]
        with log.open('a') as stream:
            proc=subprocess.Popen(cmd,cwd=CURRENT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['stages'][name]=dict(execution_status='running',pid=proc.pid,command=cmd,log=str(log),started_at=time.time())
            state.update(execution_status='running',active_stage=name);write(status_path,state)
            while proc.poll() is None:
                state['last_update']=time.time();write(status_path,state);time.sleep(15)
        state['stages'][name].update(returncode=proc.returncode,finished_at=time.time(),execution_status='completed' if proc.returncode==0 else 'failed')
        if proc.returncode:
            state['execution_status']='partial';write(status_path,state);return
        write(status_path,state)
    state.update(execution_status='completed',active_stage=None);write(status_path,state)


if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/'runtime/workflow_failure.txt').write_text(traceback.format_exc());raise
