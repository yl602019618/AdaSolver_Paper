"""Persistent ordered calibration: baseline check -> validation -> lock -> test."""
from pathlib import Path
import datetime as dt
import csv,hashlib,json,os,shutil,subprocess,sys,time

OUT=Path(__file__).resolve().parents[1]


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except (OSError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.2*(attempt+1))
def write(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


class ExistingWorker:
    """Observe an already running owned worker after a monitor-only failure."""
    def __init__(self,pid,tag):self.pid=pid;self.tag=tag;self.returncode=None
    def poll(self):
        path=Path(f'/proc/{self.pid}')
        try:
            command=(path/'cmdline').read_bytes()
            status=(path/'stat').read_text().split()[2]
            if str(OUT/'code/run.py').encode() in command and status!='Z':return None
        except (FileNotFoundError,ProcessLookupError):pass
        state=read(OUT/f'runtime/{self.tag}.json');gpu=OUT/f'runtime/{self.tag}_gpu.json'
        self.returncode=0 if state.get('status')=='completed' and gpu.exists() and read(gpu).get('exclusive') else 1
        return self.returncode


def phase(name,state):
    processes={};streams={};p=read(OUT/'protocol.json')
    jobs=[]
    for n in p['resolutions']:
        status_path=OUT/f'runtime/{name}_r{n}.json'
        if status_path.exists() and read(status_path).get('status')=='completed':continue
        if status_path.exists():
            backup=OUT/'runtime/previous_attempts';backup.mkdir(exist_ok=True)
            shutil.copyfile(status_path,backup/f'{name}_r{n}_{time.time_ns()}.json')
        count=2 if name=='test' else 1
        jobs.extend((n,s,count) for s in range(count))
    for n,shard,count in jobs:
        tag=f'{name}_r{n}'+(f'_s{shard}' if count>1 else '')
        if (OUT/f'runtime/{tag}.json').exists() and read(OUT/f'runtime/{tag}.json').get('status')=='completed':continue
        existing=read(OUT/f'runtime/{tag}.json') if (OUT/f'runtime/{tag}.json').exists() else {}
        pid=existing.get('pid');cmd=Path(f'/proc/{pid}/cmdline')
        if pid and cmd.exists() and str(OUT/'code/run.py').encode() in cmd.read_bytes():
            processes[tag]=ExistingWorker(pid,tag)
            state.setdefault('adopted_workers',[]).append(dict(tag=tag,pid=pid))
            continue
        gpu=p['physical_gpus'][str(n)]+3*shard;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['PYTHONDONTWRITEBYTECODE']='1'
        for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[var]='4'
        command=[sys.executable,'-u',str(OUT/'code/run.py'),'--resolution',str(n),'--gpu',str(gpu),'--phase',name,'--shards',str(count),'--shard',str(shard)]
        stream=(OUT/f'logs/{tag}.log').open('a');streams[tag]=stream
        process=subprocess.Popen(command,cwd=OUT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT)
        processes[tag]=process;state.setdefault('commands',[]).append(dict(phase=name,resolution=n,shard=shard,pid=process.pid,command=command,gpu=gpu))
    while any(p.poll() is None for p in processes.values()):
        progress={tag:read(OUT/f'runtime/{tag}.json') if (OUT/f'runtime/{tag}.json').exists() else {} for tag in processes}
        state.update(stage=name,progress=progress,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(OUT/'runtime/workflow.json',state)
        lines=[f'PI beta calibration: {name}. Updated {state["updated_at_utc"]}.',
               'One frozen E4 checkpoint; calibration and final test populations remain separate.',
               'All validation candidates retained. No test-based beta choice or below-one guarantee.']
        for n,row in progress.items():lines.append(f"r{n}: {row.get('status','starting')}; {row.get('completed_cases',0)}/{row.get('expected_cases','?')} cases; {row.get('metric_rows',0)} rows.")
        (OUT/'read.me').write_text('\n'.join(lines)+'\n')
        time.sleep(10)
    for tag,process in processes.items():
        if tag in streams:streams[tag].close()
        if process.returncode:raise RuntimeError(f'{tag} exited {process.returncode}; see its preserved log')
    if name=='test':
        selection=read(OUT/'selection.json')
        for n in p['resolutions']:
            rows=[];fields=None
            for sid in p['test_case_ids']:
                with (OUT/f'records/cases/test_r{n}_case{sid}.csv').open() as stream:
                    reader=csv.DictReader(stream);fields=reader.fieldnames;rows.extend(reader)
            assert len(rows)==selection['expected_counts']['test_by_resolution'][str(n)]
            target=OUT/f'records/test_r{n}.csv';temp=target.with_suffix('.tmp')
            with temp.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
            temp.replace(target)
            diffs=[float(r['legacy_metric_difference']) for r in rows if r['legacy_metric_difference']!='']
            write(OUT/f'runtime/test_r{n}.json',dict(status='completed',phase='test',resolution=n,
                completed_cases=200,metric_rows=len(rows),max_legacy_metric_difference=max(diffs),
                baseline_checks=len(diffs),execution='merged disjoint case shards plus immutable earlier completed cases',
                physical_gpus=[p['physical_gpus'][str(n)],p['physical_gpus'][str(n)]+3]))
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
            write(OUT/'runtime/executed_sources_current.json',source)
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
