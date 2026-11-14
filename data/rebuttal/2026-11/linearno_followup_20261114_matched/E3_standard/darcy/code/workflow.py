"""Persistent per-backbone stage controller; scientific gates stay explicit."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
CODE=Path(__file__).resolve().parent

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {}

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    for attempt in range(4):
        try:temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path);return
        except OSError:
            if attempt==3:raise
            time.sleep(.5)

def wait_for_gpu(state,path,physical_gpu,stage):
    """Admit only after a stable idle interval; never terminate another job."""
    idle_since=None
    while True:
        try:
            result=subprocess.run(['nvidia-smi','-i',str(physical_gpu),
                '--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            pids=[int(s.strip()) for s in result.stdout.splitlines() if s.strip().isdigit()]
            known=result.returncode==0
        except (OSError,subprocess.TimeoutExpired):known=False;pids=[]
        if known and not pids:
            idle_since=idle_since or time.monotonic()
            if time.monotonic()-idle_since>=30:return
        else:idle_since=None
        state.update(execution_status='waiting_resource',active_stage=stage,
                     waiting_for=f'idle GPU{physical_gpu}',observed_gpu_pids=pids,heartbeat=time.time())
        write(path,state);time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--key',choices=['transolver3','linearno','linearno_conv'],required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    key=args.key;backbone='Transolver3' if key=='transolver3' else 'LinearNO'
    path=OUT/'runtime'/f'{key}_workflow.json'
    state=read(path) or dict(model=key,stages={})
    state.update(pid=os.getpid(),execution_status='running',evidence_status='pending')
    stages=[('backbone','train_backbone.py',['--backbone',backbone]+(['--variant','conv'] if key=='linearno_conv' else []))]
    if key!='linearno_conv':
        stages += [('baseline_evaluation','evaluate.py',['--backbone',backbone,'--phase','baseline']),
                   ('head_bank','prepare_head.py',['--backbone',backbone]),('head','train_head.py',['--backbone',backbone]),
                   ('evaluation','evaluate.py',['--backbone',backbone,'--phase','ca'])]
    else:stages += [('evaluation','evaluate.py',['--backbone',backbone,'--conv-only'])]
    for stage,script,argv in stages:
        if state['stages'].get(stage,{}).get('execution_status')=='completed':continue
        while not (CODE/script).exists() or ('evaluation' in stage and not (OUT/'config/evaluation_lock.json').exists()):
            state.update(active_stage=stage,waiting_for='implementation or accepted reconstruction gate',heartbeat=time.time())
            write(path,state);time.sleep(10)
        wait_for_gpu(state,path,args.gpu,stage)
        command=[sys.executable,'-u',str(CODE/script),*argv]
        state['stages'][stage]=dict(execution_status='running',command=command,started_at=time.time())
        log=OUT/'logs'/f'{key}_{stage}.log'
        with log.open('a') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            state['stages'][stage]['pid']=child.pid;state['active_stage']=stage;state['execution_status']='running';state.pop('waiting_for',None);write(path,state)
            while child.poll() is None:
                state['heartbeat']=time.time();write(path,state);time.sleep(10)
            rc=child.returncode
        state['stages'][stage].update(returncode=rc,finished_at=time.time(),execution_status='completed' if rc==0 else 'failed')
        write(path,state)
        if rc:
            state.update(execution_status='failed',error=f'{stage} returncode{rc}');write(path,state);return
    state.update(execution_status='completed',active_stage=None,finished_at=time.time());write(path,state)

if __name__=='__main__':main()
