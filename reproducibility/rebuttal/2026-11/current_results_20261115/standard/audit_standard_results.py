"""Read-only CPU audit of the completed three-seed standard Transolver3 cohort."""
from __future__ import annotations
import argparse,csv,datetime,gzip,hashlib,io,json,math,os,statistics,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCHMARKS=('elasticity','pipe','darcy')
METHODS=('uniform','pi','cpg','ca')
BUDGETS=(.375,.5,.75)
PREPARED_COMMIT='8e102921f3335ffe9b44270c18b40a16d0a73cc6'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def csvrows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def savecsv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    global PREPARED_COMMIT
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--release-commit',help='Immutable existing result commit; defaults to source worktree HEAD observed before audit');args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use CUDA_VISIBLE_DEVICES empty for CPU-only validation'
    workspace=args.workspace.resolve();out=args.output.resolve();root=workspace/'rebuttal_20261114_standard_seeds';out.mkdir(parents=True,exist_ok=True)
    PREPARED_COMMIT=args.release_commit or subprocess.check_output(['git','-C',str(root/'paper'),'rev-parse','HEAD'],text=True).strip()
    if (out/'audit.json').exists():raise FileExistsError('Preserve previous audit; choose a new output directory')
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(root/'code/vendor'))
    from adasolver.models import Transolver3ModelSpec
    from adasolver.inference.ca.head import TransolverGradientHead
    from adasolver.inference.ca.elasticity_stress import ElasticityStressHead
    from adasolver.inference.elasticity_contract import sample_rng
    from adasolver.utils.seed import make_numpy_rng
    sources={};raw_index=[];registry=[];model_proofs=[];all_raw=[];groups=defaultdict(list);index_checks=[]
    deltas=defaultdict(float)
    def source(path,role,expected=None,**extra):
        path=Path(path);digest=sha(path)
        if expected is not None:assert digest==expected,(str(path),digest,expected)
        rel=path.relative_to(workspace).as_posix();stat=path.stat()
        sources[rel]=dict(path=rel,role=role,bytes=stat.st_size,sha256=digest,mtime_utc=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat(),payload_published=False,**extra)
        return sources[rel]
    def near(actual,expected,key,tolerance=1e-12):
        delta=abs(float(actual)-float(expected));deltas[key]=max(deltas[key],delta);assert delta<=tolerance,(key,actual,expected,delta)
    def compare_state(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
    policies=read(root/'config/frozen_methods.json');protocol=read(root/'config/protocol_lock.json')
    for rel in ['config/frozen_methods.json','config/protocol_lock.json','PROTOCOL.md','results/verification.json','runtime/supervisor.json','run_manifest.json']:
        source(root/rel,'protocol_or_existing_verification')
    interpolation_path=root/'code/vendor/adasolver/utils/interpolation.py';interpolation_sha=source(interpolation_path,'reconstruction_source')['sha256']
    reconstruction_id='source:adasolver.utils.interpolation.reconstruct_field@'+interpolation_sha
    def beta(benchmark,method):
        if method in ['full','uniform']:return None
        cfg=policies[benchmark][method]
        if benchmark=='elasticity':return cfg['pi' if method=='pi' else 'density']['beta']
        if 'beta_by_budget' in cfg:
            values=set(cfg['beta_by_budget'].values());assert len(values)==1;return next(iter(values))
        return cfg['beta']
    baseline_indices={}
    for benchmark in BENCHMARKS:
        cfg=read(root/f'config/{benchmark}.json');source(root/f'config/{benchmark}.json','frozen_dataset_config')
        expected_ids=cfg['split']['test_ids'];assert len(expected_ids)==200
        assert len(cfg['split']['train_ids'])==1000 and len(cfg['split']['validation_ids'])==200
        assert not set(cfg['split']['train_ids'])&set(cfg['split']['validation_ids']) and not set(expected_ids)&set(cfg['split']['validation_ids'])
        n=cfg['nfull'];normalizer=source(root/f'data/{benchmark}/normalizer.npz','normalizer_hash_only')
        model_ids=[];head_ids=[]
        for seed in [0,1,2]:
            ck=root/f'checkpoints/{benchmark}/seed{seed}';directory=root/f'records/{benchmark}/seed{seed}'
            mm=read(ck/'backbone/manifest.json');hm=read(ck/'head/manifest.json');completion=read(directory/'completion.json')
            assert mm['status']==hm['status']==completion['status']=='completed'
            assert mm['training_seed']==hm['training_seed']==hm['head_seed']==seed
            assert mm['training']==cfg['training'] and hm['head_configuration']==cfg['head']
            assert mm['split']==cfg['split'] and hm['train_ids']==cfg['split']['train_ids'] and hm['validation_ids']==cfg['split']['validation_ids']
            assert not mm['test_used_for_selection'] and not hm['test_used_for_selection']
            assert mm['best_sha256']==hm['operator_sha256']==completion['operator_sha256']
            assert hm['best_sha256']==completion['head_sha256'] and normalizer['sha256']==mm['normalizer_sha256']==hm['normalizer_sha256']==completion['normalizer_sha256']
            assert completion['records']==7400 and completion['configurations']==policies[benchmark]
            stage_proofs={}
            for stage,meta,epochs in [('backbone',mm,cfg['training']['epochs']),('head',hm,cfg['head']['epochs'])]:
                folder=ck/stage;source(folder/'manifest.json','model_or_head_manifest')
                source(folder/'best.pt','checkpoint_hash_only',meta['best_sha256']);source(folder/'last.pt','last_checkpoint_hash_only',meta['last_sha256'])
                source(folder/'resume.pt','resume_hash_only');source(folder/'history.jsonl','training_history_hash_only')
                log=root/'logs'/f'{benchmark}_seed{seed}_{stage}.log'
                if log.exists():source(log,'original_stage_log_hash_only')
                hist=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines() if line]
                assert meta['epochs_completed']==epochs and len(hist)==epochs and [r['epoch'] for r in hist]==list(range(1,epochs+1))
                vkey='validation_relative_l2' if stage=='backbone' else 'validation_loss';mkey='best_validation_relative_l2' if stage=='backbone' else 'best_validation_loss'
                assert all(math.isfinite(r[vkey]) and math.isfinite(r['elapsed_seconds']) for r in hist)
                best=min(hist,key=lambda r:r[vkey]);near(best[vkey],meta[mkey],'validation_selection',0);assert best['epoch']==meta['best_epoch']
                near(hist[-1]['elapsed_seconds'],meta['elapsed_seconds'],'training_wall_binding',0)
                resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
                assert resume['epochs_completed']==epochs and resume['best_epoch']==meta['best_epoch'];near(resume['best'],meta[mkey],'resume_best',0)
                assert compare_state(last,resume['model' if stage=='backbone' else 'head'])
                steps={int(v['step'].item()) for v in resume['optimizer']['state'].values()};assert len(steps)==1
                if stage=='backbone':
                    updates=epochs*math.ceil(1000/cfg['training']['batch_size']);model=Transolver3ModelSpec(**cfg['model']).build()
                else:
                    if benchmark=='elasticity':
                        bank=read(directory/'head_bank/manifest.json');train_points=sum(x['views']*x['count'] for x in bank['shards'] if x['split']=='train')
                        updates=epochs*math.ceil(train_points/65536);model=ElasticityStressHead(4)
                    else:updates=epochs*250;model=TransolverGradientHead()
                    assert resume['operator_sha256']==mm['best_sha256']
                assert steps=={updates},(benchmark,seed,stage,steps,updates)
                scheduler=cfg['training' if stage=='backbone' else 'head']['scheduler']
                assert resume['scheduler']['last_epoch']==(epochs if scheduler=='CosineAnnealingLR' else updates)
                if scheduler=='OneCycleLR':assert resume['scheduler']['total_steps']==updates
                model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
                device=meta.get('training_device') or hist[-1].get('device') or meta.get('environment',{}).get('gpu') or 'cuda:0'
                if benchmark=='elasticity' and stage=='head':assert device=='cpu' and all(r.get('device')=='cpu' and r.get('gpu') is None for r in hist)
                stage_proofs[stage]=dict(epochs=epochs,optimizer_updates=updates,optimizer_parameter_states=len(resume['optimizer']['state']),optimizer_step_values=sorted(steps),strict_state_load=True,last_matches_resume=True,validation_selection_replayed=True,best_epoch=meta['best_epoch'],validation_value=meta[mkey],history_rows=len(hist),training_device=device,training_wall_seconds=meta['elapsed_seconds'],class_name=type(model).__module__+'.'+type(model).__name__,parameter_count=sum(p.numel() for p in model.parameters()))
            source(directory/'completion.json','evaluation_completion')
            if benchmark!='pipe':
                bank_path=directory/'head_bank/manifest.json';bank=read(bank_path);source(bank_path,'own_backbone_feature_bank_manifest')
                assert bank['status']=='completed' and bank['operator_sha256']==mm['best_sha256'] and bank['training_seed']==seed
            wanted={(sid,None,'full',1.) for sid in expected_ids}|{(sid,s,m,b) for sid in expected_ids for s in [0,1,2] for m in METHODS for b in BUDGETS}
            rows=[]
            for filename in completion['files']:
                path=directory/filename;rr=[json.loads(line) for line in path.read_text().splitlines() if line]
                identity=source(path,'native_raw_observation_log',row_count=len(rr),dataset=benchmark,training_seed=seed)
                raw_index.append(dict(dataset=benchmark,training_seed=seed,path=identity['path'],rows=len(rr),bytes=identity['bytes'],sha256=identity['sha256'],checkpoint_sha256=mm['best_sha256'],head_sha256=hm['best_sha256'],normalizer_sha256=normalizer['sha256'],status='ok'))
                rows.extend(rr)
            keys={(r['sample_id'],r['sampling_seed'],r['method'],r['budget_fraction']) for r in rows};assert len(rows)==len(keys)==7400 and keys==wanted
            n0=97 if benchmark=='elasticity' else round(.1*n)
            for row in rows:
                assert row['benchmark']==benchmark and row['training_seed']==seed and row['status']=='ok'
                assert row['operator_sha256']==mm['best_sha256'] and row['head_sha256']==(hm['best_sha256'] if row['method']=='ca' else None)
                assert row['nfull']==n and row['selected_count']==round(n*row['budget_fraction']) and row['coarse_count']==(n0 if row['method'] in ['cpg','ca'] else 0)
                assert math.isfinite(row['relative_l2']) and row['relative_l2']>=0
                near(row['relative_l2'],row['relative_l2_numerator']/max(row['relative_l2_denominator'],1e-12),'raw_error_ratio',1e-14)
                groups[benchmark,seed,row['method'],row['budget_fraction'],row['sampling_seed']].append(row['relative_l2'])
            all_raw.extend(rows)
            for s in [0,1,2]:
                coarse=[]
                for local,sid in enumerate(expected_ids):
                    rng=sample_rng(sid,s) if benchmark=='elasticity' else make_numpy_rng(s,sid) if benchmark=='darcy' else np.random.default_rng(1000*s+local)
                    coarse.append(rng.choice(n,n0,replace=False))
                coarse=np.stack(coarse)
                for budget in BUDGETS:
                    count=round(n*budget);path=directory/f'sampling{s}_n{count}.npz';source(path,'selected_indices_hash_only')
                    with np.load(path,allow_pickle=False) as archive:
                        for method in METHODS:
                            ids=archive[method];assert ids.shape==(200,count) and ids.min()>=0 and ids.max()<n
                            assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
                            if method in ['cpg','ca']:assert np.array_equal(ids[:,:n0],coarse)
                            else:
                                key=benchmark,s,budget,method
                                if seed==0:baseline_indices[key]=ids.copy()
                                else:assert np.array_equal(ids,baseline_indices[key])
                    index_checks.append(dict(dataset=benchmark,training_seed=seed,sampling_seed=s,budget_fraction=budget,cases=200,methods=4,indices_valid=True,coarse_prefixes_match=True))
            model_ids.append(mm['best_sha256']);head_ids.append(hm['best_sha256'])
            model_proofs.append(dict(dataset=benchmark,training_seed=seed,head_seed=seed,stages=stage_proofs,raw_records=7400,case_population_hash=canonical(expected_ids),split_hash=canonical(cfg['split'])))
            registry.append(dict(dataset=benchmark,backbone='Transolver3',training_seed=seed,head_seed=seed,protocol_id=f'E5_standard_{benchmark}_1000train_200validation_v1',protocol_group='legacy_alternative_to_matched900' if benchmark=='darcy' else 'standard_three_training_seed_cohort',model_class=stage_proofs['backbone']['class_name'],head_class=stage_proofs['head']['class_name'],model_parameters=stage_proofs['backbone']['parameter_count'],head_parameters=stage_proofs['head']['parameter_count'],backbone_epochs=stage_proofs['backbone']['epochs'],head_epochs=stage_proofs['head']['epochs'],backbone_updates=stage_proofs['backbone']['optimizer_updates'],head_updates=stage_proofs['head']['optimizer_updates'],checkpoint_sha256=mm['best_sha256'],head_sha256=hm['best_sha256'],normalizer_sha256=normalizer['sha256'],split_sha256=canonical(cfg['split']),case_population_sha256=canonical(expected_ids),backbone_best_epoch=mm['best_epoch'],head_best_epoch=hm['best_epoch'],backbone_validation_score=mm['best_validation_relative_l2'],head_validation_score=hm['best_validation_loss'],backbone_selection_rule=mm['checkpoint_selection'],head_selection_rule=cfg['head']['checkpoint_selection'],backbone_training_wall_seconds=mm['elapsed_seconds'],head_training_wall_seconds=hm['elapsed_seconds'],head_training_device=stage_proofs['head']['training_device'],backbone_manifest=(ck/'backbone/manifest.json').relative_to(workspace).as_posix(),head_manifest=(ck/'head/manifest.json').relative_to(workspace).as_posix(),source_prepared_commit=PREPARED_COMMIT,source_commit_scope='prepared result/reproduction package; not a contemporaneously recorded training-code commit',status='VERIFIED_COMPLETE'))
        assert len(set(model_ids))==len(set(head_ids))==3
        print('Verified model/head/raw/selection matrices:',benchmark,flush=True)
    assert len(all_raw)==66600 and all(len(x)==200 for x in groups.values())
    compressed=root/'results/metrics_records.csv.gz';source(compressed,'public_numeric_observation_csv')
    with gzip.open(compressed,'rt',newline='') as f:public=list(csv.DictReader(f))
    assert len(public)==len(all_raw)
    for published,raw in zip(public,all_raw):assert all(published[k]==('' if v is None else str(v)) for k,v in raw.items()),'public compressed CSV differs from native raw records'
    means={key:math.fsum(values)/len(values) for key,values in groups.items()};model_mean={};absolute=[];sampling=[];rauec=[];seed_area=[];paired=[]
    for benchmark in BENCHMARKS:
        population=canonical(read(root/f'config/{benchmark}.json')['split']['test_ids'])
        for method in ['full',*METHODS]:
            for budget in ([1.] if method=='full' else BUDGETS):
                values=[]
                for t in [0,1,2]:
                    within=[means[benchmark,t,method,budget,s] for s in ([None] if method=='full' else [0,1,2])];mean=statistics.mean(within)
                    model_mean[benchmark,t,method,budget]=mean;values.append(mean)
                    sampling.append(dict(dataset=benchmark,backbone='Transolver3',training_seed=t,head_seed=t if method=='ca' else None,method=method,beta=beta(benchmark,method),budget_fraction=budget,mean_error=mean,sampling_seed0=within[0] if method!='full' else None,sampling_seed1=within[1] if method!='full' else None,sampling_seed2=within[2] if method!='full' else None,sampling_seed_sd=statistics.stdev(within) if method!='full' else None,sampling_seed_count=0 if method=='full' else 3,full_reference_repeats=1 if method=='full' else 0,cases_per_sampling_seed=200,population_sha256=population))
                absolute.append(dict(dataset=benchmark,backbone='Transolver3',metric='field_relative_l2',method=method,beta=beta(benchmark,method),budget_fraction=budget,training_seed0=values[0],training_seed1=values[1],training_seed2=values[2],mean=statistics.mean(values),training_seed_sd=statistics.stdev(values),independent_training_runs=3,cases_per_training_run=200,sampling_seeds_per_training_run=0 if method=='full' else 3,full_reference_repeats_per_training_run=1 if method=='full' else 0,population_sha256=population,reconstruction_id='identity_full' if method=='full' else reconstruction_id))
        for method in METHODS:
            values=[]
            for t in [0,1,2]:
                ratios=[]
                for s in [0,1,2]:
                    v=[means[benchmark,t,method,b,s] for b in BUDGETS];u=[means[benchmark,t,'uniform',b,s] for b in BUDGETS]
                    numerator=(v[0]+v[1])/16+(v[1]+v[2])/8;denominator=(u[0]+u[1])/16+(u[1]+u[2])/8;assert denominator>0
                    ratio=numerator/denominator;near(ratio,np.trapezoid(v,BUDGETS)/np.trapezoid(u,BUDGETS),'independent_area_identity')
                    ratios.append(ratio);seed_area.append(dict(dataset=benchmark,backbone='Transolver3',training_seed=t,sampling_seed=s,method=method,beta=beta(benchmark,method),budget_grid='0.375;0.5;0.75',method_area=numerator,matched_uniform_area=denominator,rauec=ratio,population_sha256=population,checkpoint_sha256=next(r['checkpoint_sha256'] for r in registry if r['dataset']==benchmark and r['training_seed']==t),reconstruction_id=reconstruction_id))
                values.append(statistics.mean(ratios))
            rauec.append(dict(dataset=benchmark,backbone='Transolver3',metric='field',method=method,beta=beta(benchmark,method),training_seed0=values[0],training_seed1=values[1],training_seed2=values[2],mean=statistics.mean(values),training_seed_sd=statistics.stdev(values),independent_training_runs=3,budget_grid='0.375;0.5;0.75',population_sha256=population,reconstruction_id=reconstruction_id,aggregation='mean of matched per-sampling-seed area ratios within each training run; sample SD across3training runs'))
        for method,reference in [('pi','uniform'),('cpg','uniform'),('ca','uniform'),('cpg','pi'),('ca','pi'),('ca','cpg')]:
            for budget in BUDGETS:
                values=[model_mean[benchmark,t,method,budget]-model_mean[benchmark,t,reference,budget] for t in [0,1,2]]
                paired.append(dict(dataset=benchmark,method=method,reference=reference,budget_fraction=budget,training_seed0_delta=values[0],training_seed1_delta=values[1],training_seed2_delta=values[2],mean_paired_delta=statistics.mean(values),training_seed_sd=statistics.stdev(values),independent_training_runs=3))
    lookups={
      'training_seed_summary.csv':({(r['dataset'],r['method'],r['budget_fraction']):r for r in absolute},['training_seed0','training_seed1','training_seed2','mean','training_seed_sd']),
      'training_seed_rauec.csv':({(r['dataset'],r['method']):r for r in rauec},['training_seed0','training_seed1','training_seed2','mean','training_seed_sd']),
      'paired_training_differences.csv':({(r['dataset'],r['method'],r['reference'],r['budget_fraction']):r for r in paired},['training_seed0_delta','training_seed1_delta','training_seed2_delta','mean_paired_delta','training_seed_sd']),
      'sampling_variability.csv':({(r['dataset'],r['training_seed'],r['method'],r['budget_fraction']):r for r in sampling},['mean_error','sampling_seed0','sampling_seed1','sampling_seed2','sampling_seed_sd'])}
    for filename,(lookup,fields) in lookups.items():
        source(root/'results'/filename,'existing_statistical_table');old=csvrows(root/'results'/filename);assert len(old)==len(lookup)
        for row in old:
            if filename=='training_seed_rauec.csv':key=row['dataset'],row['method']
            elif filename=='paired_training_differences.csv':key=row['dataset'],row['method'],row['reference'],float(row['budget_fraction'])
            elif filename=='sampling_variability.csv':key=row['dataset'],int(row['training_seed']),row['method'],float(row['budget_fraction'])
            else:key=row['dataset'],row['method'],float(row['budget_fraction'])
            new=lookup[key]
            for field in fields:
                # The source display aliases Full to sampling_seed0. This release
                # keeps Full unsampled and validates that alias only numerically.
                wanted=new['mean_error'] if filename=='sampling_variability.csv' and row['method']=='full' and field=='sampling_seed0' else new[field]
                if wanted is None:assert row[field]==''
                else:near(row[field],wanted,filename+'/'+field)
    release_path='rebuttal/2026-11/E5_standard_RELEASE_MANIFEST.json'
    release_blob=subprocess.check_output(['git','-C',str(root/'paper'),'show',PREPARED_COMMIT+':'+release_path]);release=json.loads(release_blob);release_checks=[];release_conflicts=[]
    assert release['file_count']==len(release['files'])
    for item in release['files']:
        blob=subprocess.check_output(['git','-C',str(root/'paper'),'show',PREPARED_COMMIT+':'+item['path']]);actual_hash=hashlib.sha256(blob).hexdigest()
        matches=actual_hash==item['sha256'] and len(blob)==item['size_bytes']
        if not matches:
            restored=blob.replace(b'\n',b'\r\n')
            release_conflicts.append(dict(path=item['path'],status='STALE_OR_CONFLICT',manifest_sha256=item['sha256'],actual_commit_sha256=actual_hash,manifest_bytes=item['size_bytes'],actual_commit_bytes=len(blob),CRLF_restoration_matches_manifest=hashlib.sha256(restored).hexdigest()==item['sha256'] and len(restored)==item['size_bytes']))
        sources['git:'+PREPARED_COMMIT+':'+item['path']]=dict(path=item['path'],git_commit=PREPARED_COMMIT,role='immutable_prepared_public_release',bytes=len(blob),sha256=actual_hash,payload_published=False)
        original_result=root/'results'/Path(item['path']).name
        if original_result.exists() and item['path'].endswith('.csv'):
            assert list(csv.DictReader(io.StringIO(blob.decode())))==csvrows(original_result),'prepared result CSV values differ from audited native result table'
        elif original_result.exists() and item['path'].endswith('.csv.gz'):assert sha(original_result)==actual_hash,'prepared compressed records differ from audited native observations'
        release_checks.append(dict(path=item['path'],sha256=actual_hash,bytes=len(blob),commit_tree_verified=True,original_manifest_hash_matches=matches))
    sources['git:'+PREPARED_COMMIT+':'+release_path]=dict(path=release_path,git_commit=PREPARED_COMMIT,role='immutable_prepared_release_manifest',bytes=len(release_blob),sha256=hashlib.sha256(release_blob).hexdigest(),payload_published=False)
    supervisor=read(root/'runtime/supervisor.json');assert supervisor['status']=='completed' and len(supervisor['finished_stages'])==27 and not supervisor['active']
    failures=supervisor['failed'];assert len(failures)==1 and failures[0]['benchmark']=='elasticity' and failures[0]['seed']==0 and failures[0]['stage']=='head'
    failure_text=(root/'logs/elasticity_seed0_head.log').read_text();assert 'BlockingIOError' in failure_text and 'fcntl.flock' in failure_text
    successful_runtime=read(root/'runtime/elasticity_seed0_head.json');assert successful_runtime['status']=='completed' and successful_runtime['epoch']==200 and successful_runtime['device']=='cpu'
    recovery=dict(status='VERIFIED_COMPLETE',failure_type='GPU/startup flock contention before training',failure_log='rebuttal_20261114_standard_seeds/logs/elasticity_seed0_head.log',failure_log_retained=True,successful_training_evidence='complete200epoch CPU history plus best/last/resume state, optimizer counters, runtime and7400completed observations',successful_training_pid=successful_runtime['pid'],successful_epochs=200,all_three_elasticity_heads_on_cpu=True,success_stdout_in_original_failure_log='COMPLETE HEAD' in failure_text,exact_recovery_launch_command='not_present_in_retained_stage_log')
    report=dict(status='VERIFIED_COMPLETE',scope='completed9Transolver3 backbones+9correspondingheads and frozen3dataset accuracy matrices only',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paper_baseline='91c1fa7cc81b8c1cb61a43618296c1c892d818c0',source_prepared_commit=PREPARED_COMMIT,counts=dict(backbones=9,heads=9,training_runs_per_dataset=3,test_cases_per_run=200,raw_records=66600,case_mean_groups=len(groups),absolute_rows=len(absolute),rauec_rows=len(rauec),sampling_rows=len(sampling),paired_rows=len(paired),area_detail_rows=len(seed_area),selection_archive_groups=len(index_checks)),numeric_max_absolute_differences=dict(deltas),model_and_head_checks=model_proofs,selection_checks=index_checks,elasticity_head_recovery=recovery,prepared_release=dict(commit=PREPARED_COMMIT,file_count=len(release_checks),manifest_sha256=hashlib.sha256(release_blob).hexdigest(),commit_tree_files=release_checks,previous_publication_receipt=read(root/'internal/publication_status.json'),remote_state='not_checked_here; previous authentication failure is historical, parent release owns current remote verification'),protocol_boundaries=dict(darcy='1000smooth1 training/200smooth2 validation/200smooth2 test; normalized-output loss; legacy alternative to the separate900/100matched LinearNO cohort',pipe='Frozen policy historically calibrated on first50cases of current200test cases; no retuning in this training-seed study',elasticity='Train0..999/validation1000..1199/test1600..1799; all3heads consistently use200epochs CPU FP32 after startup-lock recovery',checkpoint_selection='Validation only; normalizers fitted on training; historical test visibility is not relabelled as a sealed holdout',Full='One full-reference observation per case/checkpoint, no sampling seed, excluded from rAUEC',timing='No formal latency or GPU busy-time result accepted by this audit'),experiments_modified=False,GPU_used=False,model_forward_executed=False)
    report['prepared_release'].update(status='STALE_OR_CONFLICT' if release_conflicts else 'VERIFIED_COMPLETE',manifest_conflicts=release_conflicts,CSV_values_equal_audited_results=True,action='Rebuild destination release hashes from final bytes; keep this source-commit conflict receipt')
    for name,data in [('checkpoint_registry.csv',registry),('training_seed_summary.csv',absolute),('training_seed_rauec.csv',rauec),('sampling_variability.csv',sampling),('paired_training_differences.csv',paired),('rauec_per_training_sampling_seed.csv',seed_area),('raw_records_index.csv',raw_index)]:savecsv(out/name,data)
    for filename,value in [('audit.json',report),('source_hashes.json',dict(scope='Original external source bytes; paths are relative to the active experiment workspace; no weights or long-log payloads included',files=list(sources.values())))]:
        with (out/filename).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'status':'VERIFIED_COMPLETE','records':66600,'backbones':9,'heads':9,'max_numeric_difference':max(deltas.values()),'output_files':9}))


if __name__=='__main__':main()
