"""Persistent ordered calibration: baseline check -> validation -> lock -> test."""
from pathlib import Path
import datetime as dt
import hashlib,json,os,shutil,subprocess,sys,time

OUT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def phase(name,state):
    processes={};streams={};p=read(OUT/'protocol.json')
    for n in p['resolutions']:
        status_path=OUT/f'runtime/{name}_r{n}.json'
        if status_path.exists() and read(status_path).get('status')=='completed':continue
        if status_path.exists():
            backup=OUT/'runtime/previous_attempts';backup.mkdir(exist_ok=True)
            shutil.copyfile(status_path,backup/f'{name}_r{n}_{time.time_ns()}.json')
        gpu=p['physical_gpus'][str(n)];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONDONTWRITEBYTECODE']='1'
        for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[var]='4'
        command=[sys.executable,'-u',str(OUT/'code/run.py'),'--resolution',str(n),'--gpu',str(gpu),'--phase',name]
        stream=(OUT/f'logs/{name}_r{n}.log').open('a');streams[n]=stream
        process=subprocess.Popen(command,cwd=OUT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT)
        processes[n]=process;state.setdefault('commands',[]).append(dict(phase=name,resolution=n,pid=process.pid,command=command,gpu=gpu))
    while any(p.poll() is None for p in processes.values()):
        progress={str(n):read(OUT/f'runtime/{name}_r{n}.json') if (OUT/f'runtime/{name}_r{n}.json').exists() else {} for n in p['resolutions']}
        state.update(stage=name,progress=progress,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(OUT/'runtime/workflow.json',state)
        lines=[f'PI beta calibration: {name}. Updated {state["updated_at_utc"]}.',
               'One frozen E4 checkpoint; calibration and final test populations remain separate.',
               'All validation candidates retained. No test-based beta choice or below-one guarantee.']
        for n,row in progress.items():lines.append(f"r{n}: {row.get('status','starting')}; {row.get('completed_cases',0)}/{row.get('expected_cases','?')} cases; {row.get('metric_rows',0)} rows.")
        (OUT/'read.me').write_text('\n'.join(lines)+'\n')
        time.sleep(10)
    for n,process in processes.items():
        streams[n].close()
        if process.returncode:raise RuntimeError(f'{name} r{n} exited {process.returncode}; see its preserved log')
    for n in p['resolutions']:
        assert read(OUT/f'runtime/{name}_r{n}.json')['status']=='completed'
    state.setdefault('completed_stages',[]).append(name);write(OUT/'runtime/workflow.json',state)


def main():
    import fcntl
    with (OUT/'runtime/workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='running',pid=os.getpid(),started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:
            assert read(OUT/'data_cache/prepared.json')['prepared']
            source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'code').glob('*.py')}
            write(OUT/'runtime/executed_sources.json',source)
            phase('check',state)
            phase('validation',state)
            subprocess.run([sys.executable,str(OUT/'code/summarize.py'),'--select'],cwd=OUT.parent,check=True)
            state['selection']=read(OUT/'selection.json');write(OUT/'runtime/workflow.json',state)
            print(json.dumps(dict(stage='selection_locked',shared_beta=state['selection']['shared_beta'],per_resolution_beta=state['selection']['per_resolution_beta'])),flush=True)
            phase('test',state)
            subprocess.run([sys.executable,str(OUT/'code/summarize.py'),'--finalize'],cwd=OUT.parent,check=True)
            state.update(status='completed',stage='completed',finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            write(OUT/'runtime/workflow.json',state)
        except Exception as exc:
            state.update(status='failed',error=repr(exc));write(OUT/'runtime/workflow.json',state);raise


if __name__=='__main__':main()
