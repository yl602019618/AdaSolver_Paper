"""Full old/new reconstruction accuracy matrix after model-specific validation."""
from __future__ import annotations
import argparse,csv,json,time,os,hashlib
from pathlib import Path
import numpy as np
import torch
from common import OUT,write,read,digest
from inference import Runtime,comparison,metric,array_hash,build_plan,native_reconstruct,GRID_NATIVE,GRID_PLAN,reconstruction_module

FIELDS=['dataset','backbone','model_variant','checkpoint_id','normalizer_id','training_seed','head_seed','sampling_seed',
        'method','beta','budget_fraction','retained_points','case_id','metric_target','test_population','reconstruction_id',
        'metric','value','unit','status','selected_indices_hash','coarse_indices_hash','coarse_prediction_hash','full_alias',
        'arrays_path','old_new_max_abs','old_new_relative_difference','old_new_metric_difference','compensation_weights_hash']


def publish_immutable_bytes(path,payload):
    """Publish a complete file without replacing an existing, different artifact."""
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=payload:raise FileExistsError(f'Conflicting immutable artifact: {path}')
        return
    temporary=path.with_name(path.name+f'.{os.getpid()}.snapshot.tmp')
    created=False
    try:
        with temporary.open('xb') as stream:
            created=True
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=payload:raise FileExistsError(f'Conflicting immutable artifact: {path}')
    finally:
        if created and temporary.exists():temporary.unlink()


def snapshot_metrics(directory,phase,state):
    """A completion receipt is published only after its immutable CSV exists."""
    if phase not in ['baseline','complete']:raise ValueError(phase)
    directory=Path(directory);live=directory/'metrics_records.csv';payload=live.read_bytes()
    records=list(csv.DictReader(payload.decode().splitlines()))
    expected=int(state['completed_requests'])*2
    if len(records)!=expected:raise RuntimeError('Snapshot count differs from completed requests')
    target=directory/f'metrics_{phase}.csv';publish_immutable_bytes(target,payload)
    receipt=dict(execution_status='completed',phase=phase,metrics_csv=str(target),
        metrics_csv_sha256=hashlib.sha256(payload).hexdigest(),checkpoint_id=state['checkpoint_id'],
        normalizer_id=state['normalizer_id'],backbone=state['backbone'],dataset='darcy',training_seed=0,
        physical_requests=int(state['completed_requests']),metric_rows=len(records),
        methods=sorted({r['method'] for r in records}),case_ids=sorted({int(r['case_id']) for r in records}),
        reconstruction_ids=state['reconstruction_ids'],metric_target='field',test_population='historical_smooth2_200',
        model_specific_validation_accepted=state['model_specific_validation_accepted'],
        coarse_prediction_space=state['coarse_prediction_space'],retained_prediction_space=state['retained_prediction_space'],
        source_live_csv=str(live),evaluator_sha256=digest(__file__))
    receipt_path=directory/f'{phase}_completion.json'
    publish_immutable_bytes(receipt_path,(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode())
    return receipt


@torch.no_grad()
def predictions(runtime,sid,budget,seed,methods):
    cpg_sampling=None
    for method in methods:
        if method.startswith('cpg'):
            if cpg_sampling is None:cpg_sampling=runtime.sample(sid,'cpg',budget,seed)
            ids,weights,coarse,coarse_prediction,metadata=cpg_sampling
            comp=np.ones(len(ids),dtype=np.float32) if method=='cpg_beta0' else weights
            runtime.context.set(torch.as_tensor(comp,dtype=torch.float32,device=runtime.device))
            try:pred=runtime.forward(runtime.case(sid),ids)*runtime.std[0]+runtime.mean[0]
            finally:runtime.context.clear()
            actual_metadata={**metadata,'compensation_beta':0. if method=='cpg_beta0' else 1.,'selection_shared_across_beta_pair':True}
            info=dict(indices=ids,weights=comp,coarse_indices=coarse,coarse_prediction=coarse_prediction,sampling_metadata=actual_metadata)
        else:pred,info=runtime.predict(sid,method,budget,seed)
        yield method,pred,info


@torch.no_grad()
def reconstruct(runtime,pred,info,full=False):
    if full:
        field=pred.cpu().numpy().astype(np.float64);return field,field
    old=native_reconstruct('darcy',info['indices'],pred.cpu().numpy(),runtime.reference)
    plan=build_plan('darcy',runtime.reference,info['indices'],channels=1,reconstruction_id=GRID_PLAN,device=runtime.device)
    new=plan.apply_tensor(pred).cpu().numpy()
    return old,new


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',required=True);parser.add_argument('--conv-only',action='store_true')
    parser.add_argument('--phase',choices=['all','baseline','ca'],default='all');args=parser.parse_args()
    if args.conv_only and args.phase=='ca':raise ValueError('conv Full-only baseline has no CA phase')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=read(OUT/'config/evaluation_lock.json')
    if reconstruction_module.IMPLEMENTATION_VERSION!=lock['implementation_version'] or reconstruction_module.IMPLEMENTATION_SOURCE_SHA256!=lock['implementation_source_sha256']:
        raise RuntimeError('Loaded reconstruction module identity differs from accepted lock')
    if lock['new_reconstruction_id']!=GRID_PLAN:raise RuntimeError('unrecognized reconstruction contract')
    for name,expected in lock['reconstruction_sources'].items():
        if digest(Path(lock['reconstruction_code'])/name)!=expected:raise RuntimeError('reconstruction implementation changed after lock')
    needs_head=not args.conv_only and args.phase!='baseline'
    runtime=Runtime(args.backbone,'cuda:0',conv=args.conv_only,require_head=needs_head,load_head=needs_head)
    key='linearno_conv' if args.conv_only else args.backbone.lower();directory=OUT/'records'/key;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metrics_records.csv').exists() and args.phase!='ca':raise FileExistsError('Completed/partial evaluation exists; preserve before explicit recovery')
    phase_list=[('baseline',['uniform','pi','cpg_beta0','cpg']),('ca',['ca'])]
    if args.phase!='all':phase_list=[(phase,methods) for phase,methods in phase_list if phase==args.phase]
    if args.conv_only:phase_list=[]
    methods=[method for phase,names in phase_list for method in names]
    validations=[]
    for sid in lock['validation_case_ids']:
        if args.conv_only:
            pred,info=runtime.predict(sid,'full',1.,None);old,new=reconstruct(runtime,pred,info,True)
            validations.append(dict(case_id=sid,method='full',**comparison(old,new,runtime.case(sid)['truth'])))
        else:
            for budget in [.375,.5,.75]:
                for seed in [0,1,2]:
                    for method,pred,info in predictions(runtime,sid,budget,seed,methods):
                        old,new=reconstruct(runtime,pred,info)
                        check=comparison(old,new,runtime.case(sid)['truth'])
                        validations.append(dict(case_id=sid,method=method,budget=budget,seed=seed,**check))
    accepted=all(x['violating_elements']==0 and x['metric_difference']<=x['metric_tolerance'] for x in validations)
    write(OUT/'results'/f'{key}_{args.phase}_reconstruction_validation.json',dict(accepted=accepted,rows=validations,
         reconstruction_lock=lock,checkpoint_id=runtime.metadata['checkpoint_sha256']))
    if not accepted:raise RuntimeError('New checkpoint reconstruction contract failed before test')
    arrays=directory/'arrays';arrays.mkdir(exist_ok=True)
    expected=200 if args.conv_only else 9200
    state=dict(execution_status='running',evidence_status='pending',backbone=runtime.label,expected_requests=expected,
               expected_metric_rows=expected*2,completed_requests=0,completed_metric_rows=0,
               checkpoint_id=runtime.metadata['checkpoint_sha256'],normalizer_id=runtime.metadata['normalizer_sha256'],
               test_ids=lock['test_case_ids'],reconstruction_ids=[GRID_NATIVE,GRID_PLAN],
               coarse_prediction_space='normalized solution before physical decode; coarse_prediction_hash covers these normalized bytes',
               retained_prediction_space='physical solution after output standard deviation and mean; field errors computed after full reconstruction',
               model_specific_validation_accepted=True)
    rows=[];paired=[];previous_seconds=0.
    if args.phase=='ca':
        previous=read(directory/'run_manifest.json')
        if previous.get('execution_status')!='baseline_completed':raise RuntimeError('CA append requires completed baseline phase')
        if previous['checkpoint_id']!=state['checkpoint_id'] or previous['normalizer_id']!=state['normalizer_id']:
            raise RuntimeError('CA phase checkpoint/normalizer differs from baseline')
        rows=list(csv.DictReader((directory/'metrics_records.csv').open()))
        if len(rows)!=14800 or any(row['method']=='ca' for row in rows):raise RuntimeError('baseline matrix is not exactly complete')
        state.update(completed_requests=7400,completed_metric_rows=14800)
        paired=read(directory/'cpg_pair_validation.json');previous_seconds=previous['elapsed_seconds']
    state.update(active_phase=args.phase);write(directory/'run_manifest.json',state);start=time.perf_counter()
    with (directory/'metrics_records.csv').open('a' if args.phase=='ca' else 'x',newline='') as stream,(directory/'requests.jsonl').open('a' if args.phase=='ca' else 'x') as raw:
        writer=csv.DictWriter(stream,FIELDS)
        if args.phase!='ca':writer.writeheader()
        def record(sid,method,budget,seed,pred,info):
            full=method=='full';old,new=reconstruct(runtime,pred,info,full);check=comparison(old,new,runtime.case(sid)['truth'])
            if check['violating_elements'] or check['metric_difference']>check['metric_tolerance']:
                write(directory/'reconstruction_failure.json',dict(case_id=sid,method=method,budget=budget,seed=seed,**check))
                raise RuntimeError('Test reconstruction equivalence failed; no tolerance relaxation')
            identity=array_hash(np.asarray(info['indices'],np.int64));coarse=info['coarse_indices'];cp=info['coarse_prediction']
            ci=None if coarse is None else array_hash(np.asarray(coarse,np.int64));ch=None if cp is None else array_hash(cp)
            weights=info['weights'];wh=None if weights is None else array_hash(np.asarray(weights,np.float32))
            name=f'case{sid}_{method}_s{seed}_b{budget:.3f}.npz'
            np.savez_compressed(arrays/name,indices=info['indices'],prediction=pred.cpu().numpy(),
                coarse_indices=np.array([],np.int64) if coarse is None else coarse,
                coarse_prediction=np.array([],np.float32) if cp is None else cp,
                compensation_weights=np.array([],np.float32) if weights is None else weights)
            request=dict(case_id=sid,method=method,budget=budget,sampling_seed=seed,indices_hash=identity,
                         coarse_indices_hash=ci,coarse_prediction_hash=ch,compensation_weights_hash=wh,sampling_metadata=info['sampling_metadata'],
                         checkpoint_id=runtime.metadata['checkpoint_sha256'],normalizer_id=runtime.metadata['normalizer_sha256'],
                         arrays=name,**check)
            raw.write(json.dumps(request,allow_nan=False)+'\n')
            beta=None if method in ['full','uniform'] else .5 if method=='pi' else 0. if method=='cpg_beta0' else 1.
            for rid,value in [(GRID_NATIVE,check['metric_old']),(GRID_PLAN,check['metric_new'])]:
                row=dict(dataset='darcy',backbone=runtime.label,model_variant=runtime.variant,
                    checkpoint_id=runtime.metadata['checkpoint_sha256'],normalizer_id=runtime.metadata['normalizer_sha256'],
                    training_seed=0,head_seed=0 if method=='ca' else None,sampling_seed=seed,method=method,beta=beta,
                    budget_fraction=budget,retained_points=len(info['indices']),case_id=sid,metric_target='field',
                    test_population='historical_smooth2_200',reconstruction_id=rid,metric='field_relative_l2',value=value,
                    unit='ratio',status='ok',selected_indices_hash=identity,coarse_indices_hash=ci,coarse_prediction_hash=ch,
                    full_alias=full,arrays_path='arrays/'+name,old_new_max_abs=check['max_absolute_difference'],
                    old_new_relative_difference=check['relative_difference'],old_new_metric_difference=check['metric_difference'],
                    compensation_weights_hash=wh)
                writer.writerow(row);rows.append(row)
            state['completed_requests']+=1;state['completed_metric_rows']+=2
            return request
        # Full is evaluated once per physical case, never three fake seeds.
        if args.phase!='ca':
            for number,sid in enumerate(lock['test_case_ids'],1):
                pred,info=runtime.predict(sid,'full',1.,None);record(sid,'full',1.,None,pred,info)
        # Export the requested Full/Uniform/PI/CPG first; CA is its own final phase.
        for phase,phase_methods in phase_list:
            for number,sid in enumerate(lock['test_case_ids'],1):
                for budget in [.375,.5,.75]:
                    for seed in [0,1,2]:
                        pair={}
                        for method,pred,info in predictions(runtime,sid,budget,seed,phase_methods):
                            rr=record(sid,method,budget,seed,pred,info)
                            if method.startswith('cpg'):pair[method]=rr
                        if pair:
                            assert pair['cpg']['indices_hash']==pair['cpg_beta0']['indices_hash']
                            assert pair['cpg']['coarse_indices_hash']==pair['cpg_beta0']['coarse_indices_hash']
                            assert pair['cpg']['coarse_prediction_hash']==pair['cpg_beta0']['coarse_prediction_hash']
                            paired.append(dict(case_id=sid,budget=budget,seed=seed,identical_coarse=True,identical_final=True))
                stream.flush();raw.flush();write(directory/'run_manifest.json',state)
                print(key,phase,number,'/200',state['completed_requests'],flush=True)
            write(directory/f'{phase}_phase_complete.json',dict(execution_status='completed',records=state['completed_requests']))
            if phase=='baseline':
                stream.flush();raw.flush();snapshot_metrics(directory,'baseline',state)
    phase_expected=7400 if args.phase=='baseline' and not args.conv_only else expected
    assert state['completed_requests']==phase_expected and len(rows)==2*phase_expected
    state.update(execution_status='completed' if phase_expected==expected else 'baseline_completed',evidence_status='pending',elapsed_seconds=previous_seconds+time.perf_counter()-start,
                 cpg_pairs=len(paired),paired_indices_and_coarse_predictions=True,
                 raw_schema=FIELDS,source_sha256=digest(__file__))
    snapshot_metrics(directory,'complete' if phase_expected==expected else 'baseline',state)
    write(directory/'run_manifest.json',state);write(directory/'cpg_pair_validation.json',paired)


if __name__=='__main__':main()
