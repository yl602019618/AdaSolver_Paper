"""Persist concise real progress without launching or terminating experiments."""
from pathlib import Path
import os,json,time,datetime,fcntl

OUT=Path(__file__).resolve().parents[1]
LOCK=(OUT/'runtime/progress_monitor.lock').open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)

def read(path):
    for attempt in range(4):
        try:return json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.2)
    return {}

def write(path,value):
    p=path.with_name(path.name+f'.{os.getpid()}.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)

while True:
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[]
    for arch in ['t3','linearno','linearno_conv']:
        flow=read(OUT/f'runtime/{arch}_workflow.json');model=read(OUT/f'checkpoints/{arch}/manifest.json')
        sub=read(OUT/f'runtime/{arch}_head_substages.json') if flow.get('active_stage')=='head' else {}
        rows.append(dict(architecture=arch,workflow=flow.get('execution_status','not_started'),
                         stage=sub.get('active_stage') or flow.get('active_stage'),outer_stage=flow.get('active_stage'),epochs_completed=model.get('epochs_completed',0),
                         best_validation_relative_l2=model.get('best_validation_relative_l2'),
                         train_pid=model.get('pid'),training_seconds=model.get('training_seconds')))
    timing=read(OUT/'runtime/latency_workflow.json')
    state=dict(updated_at_utc=now,models=rows,latency=timing.get('execution_status','not_started'))
    write(OUT/'runtime/progress.json',state)
    text=f'Updated UTC: {now}\n'
    for row in rows:text+=f"{row['architecture']}: {row['workflow']}, stage={row['stage']}, epochs={row['epochs_completed']}/1000, best validation physical L2={row['best_validation_relative_l2']}\n"
    text+='Main models use 900train/100validation; test population is the fixed published110 cases.\n'
    text+='LinearNO point-temp and conv-temp are distinct official model variants; conv is a full-grid supplement only.\n'
    text+='Training costs are observed wall time; an early shared-GPU interval is documented. Formal latency requires exclusivity.\n'
    text+='Raw/equivalence results: records/{architecture}; immutable phase metrics: metrics_preliminary.csv and metrics_complete.csv; metrics.csv is a compatibility view.\n'
    text+='Final reconstruction/latency uses accepted v1.1 with actual imported source identity; old/new predictions share indices.\n'
    text+=f"Latency: {state['latency']}. Results, figures and proof of completeness are separate from process completion.\n"
    text+='This Airfoil runner performs no Git operations. Root handles the user-authorized independent-branch snapshot; main and Feishu remain outside this runner.\n'
    p=OUT/'read.me';temp=p.with_name(p.name+'.tmp');temp.write_text(text);temp.replace(p)
    if all(r['workflow']=='completed' for r in rows) and state['latency']=='completed':break
    time.sleep(30)
