"""Own two idle GPUs, then execute one complete matched-resolution workflow."""
from pathlib import Path
import argparse,datetime as dt,fcntl,json,os,subprocess,sys,time,traceback

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'AdaSolver_official'))
from inference_speed.nvml import NVML


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except FileNotFoundError:return {}
        except (OSError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    for attempt in range(5):
        try:tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path);return
        except OSError:
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def snapshot(devices):
    rows=[]
    for index,device in devices.items():
        data,jobs=device.snapshot();rows.append(dict(index=index,gpu=data,processes=jobs))
    return rows


def own_descendants(root_pid):
    rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()
    pairs=[tuple(map(int,row.split())) for row in rows if len(row.split())==2]
    pids={root_pid}
    while True:
        more={pid for pid,parent in pairs if parent in pids}
        if more<=pids:return pids
        pids|=more


def acquire(gpus,state):
    devices={i:NVML(i) for i in gpus}
    while True:
        locks=[]
        try:
            for i in sorted(gpus):
                file=open(f'/tmp/adasolver_profile_gpu_{i}.lock','a')
                try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except Exception:file.close();raise
                locks.append(file)
            for _ in range(3):
                rows=snapshot(devices)
                if any(r['processes'] or int(r['gpu']['memory.used'])>32 or int(r['gpu']['utilization.gpu'])>1 for r in rows):
                    raise RuntimeError('GPU pair is occupied')
                time.sleep(.4)
            state.update(status='running',gpu_admission=rows);return locks,devices
        except (BlockingIOError,RuntimeError) as error:
            for file in locks:fcntl.flock(file,fcntl.LOCK_UN);file.close()
            state.update(status='waiting_for_idle_gpus',reason=str(error),gpu_snapshot=snapshot(devices))
            write(OUT/f'runtime/r{ARGS.resolution}_workflow.json',state);time.sleep(10)


def run_stage(name,command,gpus,state,devices):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus));env['PYTHONDONTWRITEBYTECODE']='1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='4'
    env['ADASOLVER_NATIVE_CACHE']=str(OUT/'runtime/native_fps_cache')
    log=OUT/f'logs/r{ARGS.resolution}_{name}.log'
    with log.open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        stage=dict(status='running',pid=process.pid,command=command,log=str(log),started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        state['stage']=name;state.setdefault('stages',{})[name]=stage;write(OUT/f'runtime/r{ARGS.resolution}_workflow.json',state)
        with (OUT/f'runtime/r{ARGS.resolution}_occupancy.jsonl').open('a') as gpu_log:
            while process.poll() is None:
                owners=own_descendants(process.pid);observed=snapshot(devices)
                foreign=[job for row in observed for job in row['processes'] if job['pid'] not in owners]
                gpu_log.write(json.dumps(dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage=name,owners=sorted(owners),gpus=observed,foreign_processes=foreign))+'\n');gpu_log.flush()
                if foreign:state.setdefault('external_occupancy_observations',[]).append(dict(stage=name,processes=foreign))
                state['heartbeat_utc']=dt.datetime.now(dt.timezone.utc).isoformat();write(OUT/f'runtime/r{ARGS.resolution}_workflow.json',state)
                time.sleep(10)
    stage.update(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(OUT/f'runtime/r{ARGS.resolution}_workflow.json',state)
    if process.returncode:raise RuntimeError(f'{name} failed with exit {process.returncode}; see {log}')


def distributed(script,n,micro=2,resume=False):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(OUT/'code'/script),'--resolution',str(n)]
    if script in ['preflight_training.py','train_ddp.py']:command+=['--microbatch-size',str(micro)]
    if resume:command+=['--resume']
    return command


def main():
    n=ARGS.resolution;p=read(OUT/'protocol.json');planned_gpus=p['resources']['training_gpu_pairs'][str(n)]
    gpus=list(map(int,ARGS.gpus.split(','))) if ARGS.gpus else planned_gpus
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Exactly two distinct GPU indices are required')
    with (OUT/f'runtime/r{n}_workflow.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='starting',pid=os.getpid(),resolution=n,gpus=gpus,planned_gpus=planned_gpus,resource_override=bool(ARGS.gpus),stages={},started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        assert read(OUT/'data/prepared.json')['prepared']
        locks,devices=acquire(gpus,state)
        try:
            micro=2
            receipt=read(OUT/f'results/training_preflight_r{n}.json')
            if not receipt.get('passed'):
                run_stage('training_preflight',distributed('preflight_training.py',n,micro),gpus,state,devices)
            else:micro=receipt['microbatch_size']
            current=read(OUT/f'checkpoints/r{n}/manifest.json')
            if current.get('execution_status')!='completed':
                resume=(OUT/f'checkpoints/r{n}/initial.pt').exists()
                run_stage('backbone',distributed('train_ddp.py',n,micro,resume),gpus,state,devices)
            # Head and evaluation entrypoints are prepared while full training runs.
            # Their commands are registered in the protocol-owned stage contract.
            stage_contract=OUT/'config/post_backbone_commands.json'
            while not stage_contract.exists():
                state.update(stage='waiting_post_backbone_code',heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat());write(OUT/f'runtime/r{n}_workflow.json',state);time.sleep(10)
            commands=read(stage_contract)[str(n)]
            for row in commands:
                gate=OUT/row['completion_receipt']
                if gate.exists() and read(gate).get(row.get('completion_field','passed'))==row.get('completion_value',True):continue
                command=[str(token).replace('{python}',sys.executable).replace('{out}',str(OUT)) for token in row['command']]
                visible=gpus if row.get('gpu_count',2)==2 else gpus[:1]
                run_stage(row['stage'],command,visible,state,devices)
            state.update(status='completed',stage=None,finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat());write(OUT/f'runtime/r{n}_workflow.json',state)
        except Exception:
            state.update(status='failed',traceback=traceback.format_exc());write(OUT/f'runtime/r{n}_workflow.json',state);raise
        finally:
            for file in locks:fcntl.flock(file,fcntl.LOCK_UN);file.close()
            for device in devices.values():device.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True);parser.add_argument('--gpus',help='Resource-only override: two physical GPU indices');ARGS=parser.parse_args();main()
