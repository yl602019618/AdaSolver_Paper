"""Persistent accuracy -> controlled timing -> isolated memory -> aggregation queue."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from workflow import OUT,CODE,read,write,wait_for_gpu


def update_readme(state):
    lines=['Darcy follow-up status — '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'',
        'Locked: shared 900/100 split, 900-only normalizers, native T3 / official LinearNO point-temp, additional conv-temp Full-only, reconstruction v1.1. See protocol.md, LINEARNO_ADAPTER.md and COMMANDS.md.',
        'Completed implementation checks are in results/*preflight.json. Initialized-model checks are not trained accuracy or latency evidence.','']
    for key in ['transolver3','linearno','linearno_conv']:
        workflow=read(OUT/'runtime'/f'{key}_workflow.json');model=read(OUT/'checkpoints'/key/'manifest.json')
        head=read(OUT/'checkpoints'/key/'head/manifest.json');evaluation=read(OUT/'records'/key/'run_manifest.json')
        lines.append(f'{key}: {workflow.get("execution_status","pending")}; stage {workflow.get("active_stage")}; controller PID {workflow.get("pid")}; backbone {model.get("epochs_completed",0)}/1000 epochs; head {head.get("epochs_completed",0)}/1000 epochs; accuracy {evaluation.get("completed_requests",0)}/{evaluation.get("expected_requests",200 if key=="linearno_conv" else 9200)} requests.')
        if workflow.get('waiting_for'):lines.append('  Waiting for '+workflow['waiting_for']+'; observed GPU PIDs '+str(workflow.get('observed_gpu_pids',[])))
    lines.extend(['',f'Postprocess: {state.get("execution_status")}; stage {state.get("active_stage")}; PID {state.get("pid")}.',
        'Baseline evaluation precedes own CA head training: 7,400 baseline requests plus 1,800 CA additions per point model. Conv contributes 200 Full requests.',
        'Formal A/B/C matrix: 333 groups / 1,323 conditions / 132,300 repetitions, all on GPU0 after stable idle admission. Separate 11-process memory matrix; no external job is terminated.',
        'Final cross-dataset plots and independent rAUEC review are parent-owned. No remote push, old result overwrite or Feishu write.'])
    path=OUT/'read.me';temporary=path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
    lock=(OUT/'runtime/postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statefile=OUT/'runtime/postprocess_workflow.json';state=read(statefile) or dict(stages={})
    state.update(pid=os.getpid(),execution_status='waiting_dependency',evidence_status='pending',gpu=args.gpu)
    while True:
        dependencies={key:read(OUT/'records'/key/'run_manifest.json').get('execution_status','pending')
                      for key in ['transolver3','linearno','linearno_conv']}
        if all(v=='completed' for v in dependencies.values()):break
        state.update(dependencies=dependencies,heartbeat=time.time());write(statefile,state);update_readme(state);time.sleep(10)
    stages=[('accuracy_aggregation','finalize_accuracy.py',[],False),
            ('paired_latency','benchmark.py',['--phase','paired','--gpu',str(args.gpu)],True)]
    for backbone in ['Transolver3','LinearNO']:
        for method in ['full','uniform','pi','cpg','ca']:
            stages.append((f'memory_{backbone}_{method}','benchmark.py',
                ['--phase','memory','--backbone',backbone,'--method',method,'--gpu',str(args.gpu)],True))
    stages.append(('memory_LinearNO_conv_full','benchmark.py',
                   ['--phase','memory','--backbone','LinearNO','--conv-only','--method','full','--gpu',str(args.gpu)],True))
    stages.append(('latency_aggregation','finalize_latency.py',[],False))
    stages.append(('local_report','report_results.py',[],False))
    for name,script,arguments,gpu_stage in stages:
        if state['stages'].get(name,{}).get('execution_status')=='completed':continue
        if not (CODE/script).exists():raise FileNotFoundError(script)
        if gpu_stage:wait_for_gpu(state,statefile,args.gpu,name)
        command=[sys.executable,'-u',str(CODE/script),*arguments]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        log=OUT/'logs'/f'postprocess_{name}.log'
        with log.open('a') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state['stages'][name]=dict(execution_status='running',command=command,pid=child.pid,started_at=time.time(),log=str(log))
            state.update(execution_status='running',active_stage=name);state.pop('waiting_for',None);write(statefile,state)
            while child.poll() is None:
                state['heartbeat']=time.time();write(statefile,state);update_readme(state);time.sleep(10)
        state['stages'][name].update(execution_status='completed' if child.returncode==0 else 'failed',
                                    returncode=child.returncode,finished_at=time.time());write(statefile,state)
        if child.returncode:
            state.update(execution_status='failed',error=f'{name}: returncode {child.returncode}');write(statefile,state);update_readme(state);return
    state.update(execution_status='completed',active_stage=None,finished_at=time.time(),
                 evidence_status='pending_cross_dataset_independent_review');write(statefile,state);update_readme(state)


if __name__=='__main__':main()
