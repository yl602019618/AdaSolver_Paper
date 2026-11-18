"""After all accuracy matrices finish, audit and render the reviewable result set."""
from pathlib import Path
import argparse,datetime as dt,json,os,subprocess,sys,time,traceback

OUT=Path(__file__).resolve().parents[1]


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except FileNotFoundError:return {}
        except (OSError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def ready():
    return all(read(OUT/f'results/r{n}_accuracy_acceptance.json').get('passed') is True for n in [85,141,211])


def main(watch=False):
    status=OUT/'runtime/finalization.json'
    state=dict(status='waiting_for_accuracy',pid=os.getpid(),stages=[],started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(status,state)
    while not ready():
        if not watch:raise RuntimeError('Complete accepted accuracy is required for all three resolutions')
        state['heartbeat_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        state['upstream_status']={str(n):read(OUT/f'runtime/r{n}_workflow.json').get('status') for n in [85,141,211]}
        failed=[n for n,value in state['upstream_status'].items() if value=='failed']
        if failed:
            state.update(status='upstream_failed',failed_resolutions=failed)
            write(status,state)
            raise RuntimeError(f'Upstream workflows failed for {failed}; recover them before restarting finalization')
        write(status,state);time.sleep(30)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[name]='4'
    state['status']='running';write(status,state)
    for script in ['verify_training.py','verify_design.py','recompute_physical_regions.py','verify_physical_regions.py','aggregate.py','verify_statistics.py','plot_results.py']:
        row=dict(script=script,status='running',started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        state['stages'].append(row);write(status,state)
        command=[sys.executable,str(OUT/'code'/script)]
        with (OUT/f'logs/finalize_{Path(script).stem}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=OUT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=process.pid
            while process.poll() is None:
                state['heartbeat_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
                write(status,state);time.sleep(5)
        row.update(returncode=process.returncode,status='completed' if process.returncode==0 else 'failed',finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(status,state)
        if process.returncode:
            state['status']='failed';write(status,state);raise RuntimeError(f'{script} failed; see its finalization log')
    for name in ['training_verification','design_acceptance','physical_region_acceptance','physical_region_independent_checks','statistics_acceptance','figures_receipt']:
        receipt=read(OUT/f'results/{name}.json');assert receipt['passed'],name
    assert read(OUT/'results/statistics_acceptance.json')['raw_metric_rows']==44400
    state.update(status='ready_for_visual_review',finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat());write(status,state)
    write(OUT/'results/ready_for_review.json',dict(passed=True,accuracy_matrices=3,physical_requests=22200,metric_rows=44400,
        statistical_and_identity_checks=True,figures_generated=True,visual_review_pending=True,
        finished_at_utc=state['finished_at_utc']))
    print('All experiments and deterministic checks complete; result set is ready for visual review.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    try:main(parser.parse_args().watch)
    except Exception:
        status=OUT/'runtime/finalization.json';state=read(status)
        state.update(status='failed',traceback=traceback.format_exc(),finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(status,state)
        raise
