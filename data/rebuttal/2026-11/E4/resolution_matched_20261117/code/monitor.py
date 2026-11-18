"""Publish measured progress; never turn running or missing stages into results."""
from pathlib import Path
import argparse,datetime as dt,json,os,statistics,time

OUT=Path(__file__).resolve().parents[1]


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except FileNotFoundError:return {}
        except (OSError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    for attempt in range(5):
        try:tmp.write_text(value);tmp.replace(path);return
        except OSError:
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def epoch_progress(n,head=False):
    folder=OUT/f'checkpoints/r{n}'
    if head:folder=folder/'head'
    m=read(folder/'manifest.json');rows=[]
    if head:
        path=folder/'history.jsonl'
        if path.exists():
            for line in path.read_text().splitlines()[-6:]:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
    else:rows=[read(p) for p in sorted((folder/'history').glob('epoch_*.json'))[-5:]]
    times=[r['epoch_seconds'] for r in rows if r.get('epoch_seconds',0)>0]
    epoch=m.get('epochs_completed',0)
    median=statistics.median(times) if times else None
    return dict(status=m.get('execution_status','not_started'),epochs=epoch,total_epochs=1000,
        median_recent_epoch_seconds=median,remaining_training_hours=(1000-epoch)*median/3600 if median else None,
        best_epoch=m.get('best_epoch'),best_validation=m.get('best_validation_relative_l2',m.get('best_validation')))


def snapshot():
    now=dt.datetime.now(dt.timezone.utc);rows=[]
    for n in [85,141,211]:
        w=read(OUT/f'runtime/r{n}_workflow.json');accuracy=read(OUT/f'results/r{n}_accuracy_acceptance.json')
        row=dict(resolution=n,workflow=w.get('status','not_started'),stage=w.get('stage'),
            controller_alive=alive(w.get('pid')),gpus=w.get('gpus'),heartbeat_utc=w.get('heartbeat_utc'),
            backbone=epoch_progress(n),head=epoch_progress(n,True),accuracy_accepted=accuracy.get('passed',False))
        shards=[read(OUT/f'runtime/evaluation_r{n}_all_shard{i}of2.json') for i in [0,1]]
        row['evaluation_cases_completed']=sum(s.get('completed_cases',0) for s in shards)
        row['evaluation_requests_completed']=sum(s.get('completed_requests',0) for s in shards)
        if w.get('status')=='running' and not row['controller_alive']:row['alert']='controller no longer alive'
        if w.get('status')=='failed':row['alert']=w.get('traceback','workflow failed')
        rows.append(row)
    complete=all(row['accuracy_accepted'] for row in rows)
    final=read(OUT/'results/final_acceptance.json')
    result=dict(updated_at_utc=now.isoformat(),updated_at_beijing=now.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        status='completed' if final.get('passed') else 'failed' if any(r.get('alert') for r in rows) else 'accuracy_complete_finalization_pending' if complete else 'in_progress',
        resolutions=rows,expected_physical_requests=22200,expected_common_native_metric_rows=44400,
        training_seeds=[0],sampling_seeds=[0,1,2])
    atomic(OUT/'runtime/status.json',json.dumps(result,indent=2)+'\n')
    lines=['Darcy resolution-matched training study',f'Updated: {result["updated_at_beijing"]}',f'Status: {result["status"]}',
        '', 'Resolution | Stage | Backbone epochs | Measured seconds/epoch | Remaining backbone hours | CA head epochs | Accuracy accepted',
        '--- | --- | --- | --- | --- | --- | ---']
    for row in rows:
        b=row['backbone'];h=row['head'];sec=b['median_recent_epoch_seconds'];hours=b['remaining_training_hours']
        lines.append(f'{row["resolution"]}² | {row["stage"] or row["workflow"]} | {b["epochs"]}/1000 | {sec:.2f}'
            .replace('None','N/A')+f' | {hours:.2f} | {h["epochs"]}/1000 | {row["accuracy_accepted"]}' if sec is not None else
            f'{row["resolution"]}² | {row["stage"] or row["workflow"]} | {b["epochs"]}/1000 | N/A | N/A | {h["epochs"]}/1000 | {row["accuracy_accepted"]}')
    lines+=['','Time estimates above cover backbone training only; each backbone is followed by its own CA head and complete evaluation.',
        'True 421² source nodes, strides 5/3/2. Same smooth1 rows 0:900 training and 900:1000 validation; smooth2 rows 0:200 evaluation.',
        'The new 85² control is retrained because previous weights do not match the requested contiguous split.',
        'All backbones use seed 0, FP32, full native inputs, 1000 epochs, effective batch 4, and 225000 optimizer updates.',
        'PI uses the predeclared shared beta=1 with original E4 prior. CPG/CA preserve E4 settings and include coarse points in the final budget.',
        'Common211 and native-grid errors are separate. Error bars represent sampling seeds, not independent training runs.',
        'Original paper85 remains a labeled historical reference; only new85/141/211 form the controlled trend.',
        'Resource mapping change for 141² is recorded in runtime/resource_amendment.json without changing the scientific protocol.',
        '', 'Paths: protocol.md, protocol.json, split_manifest.json, config/, checkpoints/, evaluation/, results/, logs/, runtime/.',
        'Progress: python3 code/monitor.py. Reproduction commands: config/post_backbone_commands.json and runtime/*_workflow.json.',
        ('Final experiment acceptance has passed; all training, accuracy matrices, statistics and scientific figures are complete. See results/final_acceptance.json.' if final.get('passed') else 'Final conclusions remain pending until results/final_acceptance.json passes.')]
    lines+=['','Complete-case evaluation progress (37 requests per case):']
    lines += [f'{r["resolution"]}²: {r["evaluation_cases_completed"]}/200 cases; {r["evaluation_requests_completed"]}/7400 requests.' for r in rows]
    recovery=OUT/'development/head_acceptance_batch_context_20261119/README.md'
    if recovery.exists():lines+=['','The original head-verifier interruption and its fix are documented in development/head_acceptance_batch_context_20261119/README.md. Matching the original batch8 replay passed unchanged tolerances; all trained weights were preserved.']
    alerts=[f'{r["resolution"]}²: {r["alert"]}' for r in rows if r.get('alert')]
    if alerts:lines+=['','Observed execution issues:']+alerts
    atomic(OUT/'read.me','\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        state=snapshot()
        print(json.dumps(dict(time=state['updated_at_beijing'],status=state['status'],
            progress=[dict(resolution=r['resolution'],stage=r['stage'],backbone=r['backbone']['epochs'],head=r['head']['epochs']) for r in state['resolutions']])),flush=True)
        if not args.watch or state['status']=='completed':break
        time.sleep(30)
