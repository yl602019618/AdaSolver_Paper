"""Persistent per-backbone training/head/evaluation supervisor."""
from common import *
import argparse,subprocess,time,traceback,datetime


def robust_read(path):
    for attempt in range(5):
        try:return read(path)
        except (OSError,json.JSONDecodeError):time.sleep(.2*(attempt+1))
    return read(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['t3','linearno','linearno_conv'],required=True);args=p.parse_args();arch=args.architecture
    state_path=OUT/f'runtime/{arch}_workflow.json'
    state=dict(architecture=arch,pid=os.getpid(),execution_status='running',evidence_status='pending',stages={},
               started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_affinity=sorted(os.sched_getaffinity(0)),
               physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(state_path,state)
    stages=[('train','train.py')]
    if arch!='linearno_conv':stages.append(('head','train_head.py'))
    for name,script in stages:
        state['active_stage']=name
        cmd=[sys.executable,'-u',str(OUT/'code'/script),'--architecture',arch]
        log_path=OUT/f'logs/{arch}_{name}.log'
        with log_path.open('x') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=CURRENT)
            state['stages'][name]=dict(execution_status='running',pid=process.pid,command=cmd,log=str(log_path));write(state_path,state)
            while process.poll() is None:
                state['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(state_path,state);time.sleep(15)
        state['stages'][name].update(execution_status='completed' if process.returncode==0 else 'failed',returncode=process.returncode)
        write(state_path,state)
        if process.returncode:
            state['execution_status']='failed';write(state_path,state);raise SystemExit(process.returncode)
    state['active_stage']='waiting_reconstruction';write(state_path,state)
    gate=OUT/'config/evaluation_gate.json'
    while not gate.exists() or robust_read(gate).get('accepted') is not True:
        state['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(state_path,state);time.sleep(30)
    cmd=[sys.executable,'-u',str(OUT/'code/evaluate.py'),'--architecture',arch]
    log_path=OUT/f'logs/{arch}_evaluation.log'
    with log_path.open('x') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=CURRENT)
        state['active_stage']='evaluation';state['stages']['evaluation']=dict(execution_status='running',pid=process.pid,command=cmd,log=str(log_path));write(state_path,state)
        while process.poll() is None:
            state['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(state_path,state);time.sleep(15)
    state['stages']['evaluation'].update(execution_status='completed' if process.returncode==0 else 'failed',returncode=process.returncode)
    state.update(execution_status='completed' if process.returncode==0 else 'failed',active_stage=None);write(state_path,state)


if __name__=='__main__':main()
