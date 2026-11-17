#!/usr/bin/env python3
"""Complete the existing LinearNO seed0 PI/CA training and native inference."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT, E3, sha256, verify_lock, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['head_adaptive','controls'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();verify_lock()
    state={'group':args.group,'physical_gpu':args.gpu,'training_seed':0,
           'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'execution_status':'running','pid':os.getpid(),'stages':{}}
    path=ROOT/'runs/seed0'/f'{args.group}_workflow.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    child=None
    def stop(sig,frame):
        if child and child.poll() is None:child.send_signal(signal.SIGINT)
        state['execution_status']='interrupted';write_json(path,state)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(name,command):
        nonlocal child
        log=ROOT/'runs/seed0'/f'{name}.log';item={'status':'running','command':command,'log':str(log),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['stages'][name]=item;state['active_stage']=name;started=time.monotonic()
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            item['pid']=child.pid;write_json(path,state);code=child.wait()
        item.update(status='completed' if code==0 else 'failed',returncode=code,wall_seconds=time.monotonic()-started,
                    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(path,state)
        if code:raise RuntimeError(f'{name} exited {code}')
    try:
        if args.group=='head_adaptive':
            for stage in ['head_dataset','head','adaptive']:
                run(stage,[sys.executable,'-B','-u',str(ROOT/'run_stage.py'),'--seed','0','--stage',stage,'--gpu',str(args.gpu)])
        else:
            gpu_lock=open(f'/tmp/adasolver_profile_gpu_{args.gpu}.lock','a')
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            busy=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if busy.strip():raise RuntimeError('GPU occupied; no control run launched')
            manifest=E3/'checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json'
            common=[sys.executable,'-B','-u',str(ROOT/'evaluate.py'),'--manifest',str(manifest),
                    '--splits',str(ROOT/'splits/seed0.json'),'--split','test','--device','cuda:0']
            run('control_full',common+['--method','fullmesh','--seeds','0','--output',str(ROOT/'records/seed0/control_full')])
            run('controls',common+['--configs',str(ROOT/'controls.json'),'--seeds','0','1','2','--output',str(ROOT/'records/seed0/controls')])
            gpu_lock.close()
        state.update(execution_status='execution_complete_acceptance_pending',active_stage=None,
                     finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(path,state)
    except BaseException as error:
        state.update(execution_status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        write_json(path,state);raise


if __name__=='__main__':main()
