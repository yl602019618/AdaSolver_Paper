#!/usr/bin/env python3
"""Execute the single fixed seed3 extension without outcome-based reselection."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
CURRENT = ROOT.parent
ORIGINAL = CURRENT / 'p0_airfoil_original_cdcl_20261117'


def read(p): return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    assert read(ROOT/'preflight.json')['passed']
    plan = read(ROOT/'protocol.json')
    assert plan['new_seeds_to_try'] == [3] and plan['display_training_seeds'] == [1,2,3]
    for p, h in read(RUN/'config/source_manifest.json')['files'].items():
        assert sha(RUN/p) == h, p
    lock = open(f'/tmp/adasolver_profile_gpu_{args.gpu}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.run(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'], capture_output=True,text=True,check=True)
    if occupied.stdout.strip(): raise RuntimeError('selected GPU is occupied')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    state = {'pid':os.getpid(), 'physical_gpu':args.gpu, 'fixed_training_seed':3,
             'analysis_status':'post_hoc_exploratory_subset', 'display_training_seeds':[1,2,3],
             'excluded_training_seed':0, 'original_seed0_evidence_preserved':True,
             'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'execution_status':'running', 'stages':{}, 'protocol_sha256':sha(ROOT/'protocol.json')}
    status = ROOT/'pipeline_status.json'
    child = None

    def stop(sig, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        state['execution_status']='interrupted'
        state['signal']=sig
        write(status,state)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def stage(name, command, cwd):
        nonlocal child
        log = RUN/'logs'/f'{name}.log'
        if log.exists(): raise FileExistsError(log)
        item = {'command':command,'status':'running','log':str(log),
                'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['active_stage']=name
        state['stages'][name]=item
        started=time.perf_counter()
        with log.open('x') as f:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
            item['pid']=child.pid
            write(status,state)
            rc=child.wait()
        item.update(status='completed' if rc==0 else 'failed',returncode=rc,
                    wall_seconds=time.perf_counter()-started,
                    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(status,state)
        if rc: raise RuntimeError(f'{name} failed: exit {rc}')

    try:
        write(status,state)
        stage('backbone',[sys.executable,'-B','-u',str(RUN/'code/train.py'),'--architecture','t3'],RUN)
        model=read(RUN/'checkpoints/t3/manifest.json')
        assert model['execution_status']=='completed' and model['training_seed']==3
        assert model['epochs_completed']==1000 and model['total_updates']==113000
        assert sha(model['checkpoint'])==model['checkpoint_sha256']
        stage('head',[sys.executable,'-B','-u',str(RUN/'code/train_head.py'),'--architecture','t3'],RUN)
        assert read(RUN/'results/t3_head_acceptance.json')['passed']
        head=read(RUN/'checkpoints/t3/head/manifest.json')
        model_path=RUN/'checkpoints/t3/manifest.json'
        head_path=RUN/'checkpoints/t3/head/manifest.json'
        head_weights=RUN/'checkpoints/t3/head/pressure_d8_best.pt'
        original_registry=read(ORIGINAL/'runtime_registry.json')
        entry={'backbone':'Transolver3','training_seed':3,'head_seed':3,
               'checkpoint':model['checkpoint'],'checkpoint_sha256':model['checkpoint_sha256'],
               'manifest':str(model_path),'manifest_sha256':sha(model_path),
               'head_checkpoint':str(head_weights),'head_sha256':sha(head_weights),
               'head_manifest':str(head_path),'head_manifest_sha256':sha(head_path),
               'normalizer':model['normalizer'],'normalizer_sha256':model['normalizer_sha256'],
               'target_channel':3,'reconstruction_id':'original_index_grid_linear_nearest'}
        assert entry['head_sha256']==head['head_sha256'] and head['operator_sha256']==entry['checkpoint_sha256']
        write(ROOT/'evaluation/runtime_registry.json',{'data_path':original_registry['data_path'],
              'control':original_registry['control'],'training_seed3':entry})
        stage('original_cdcl_evaluation',[sys.executable,'-B','-u',str(ROOT/'evaluation/run_evaluation.py'),
              '--cohort','training','--training-seed','3','--shard','0','--shards','1','--physical-gpu',str(args.gpu)],ROOT/'evaluation')
        state.update(execution_status='execution_complete_evidence_pending',active_stage=None,
                     finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(status,state)
    except BaseException as exc:
        state['execution_status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(exc).__name__}: {exc}'
        write(status,state)
        raise
    finally:
        lock.close()


if __name__=='__main__': main()
