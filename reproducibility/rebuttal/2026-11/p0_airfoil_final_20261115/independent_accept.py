#!/usr/bin/env python3
"""CPU acceptance for one Airfoil training seed's complete P0 evidence.

Seed0 may merge a verified legacy timing root with the disjoint beta0 supplement.
Every condition is keyed by checkpoint/case/seed/budget/reconstruction/scope.
Missing observations are never synthesized; final acceptance requires1638.
"""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,math,os
from pathlib import Path
import numpy as np
from benchmark_adapter import CASES,SCOPES,METHODS,LABELS,BUDGETS


def sanitize(value,workspace):
    if isinstance(value,str):return value.replace(str(workspace)+'/','workspace:').replace(str(workspace),'workspace:')
    if isinstance(value,list):return [sanitize(v,workspace) for v in value]
    if isinstance(value,dict):return {sanitize(k,workspace):sanitize(v,workspace) for k,v in value.items()}
    return value


def csv_write(path,rows):
    if not rows:return
    final=path
    path=path.with_name(path.name+f'.{os.getpid()}.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    if str(final).endswith('.gz'):
        with path.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as compressed:
                import io
                with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                    w=csv.DictWriter(text,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    else:
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    path.replace(final)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed-root',type=Path,required=True);p.add_argument('--timing-root',type=Path,required=True)
    p.add_argument('--legacy-timing-root',type=Path);p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();seed_root=a.seed_root.resolve();output=a.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    workspace=next(x for x in seed_root.parents if x.name=='07_current')
    inputs={};hash_checks=[];content_cache={}
    def content(path):
        path=Path(path);name=str(path)
        if name not in content_cache:
            data=path.read_bytes();content_cache[name]=data
            inputs[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        return content_cache[name]
    def sha(path):
        path=Path(path);name=str(path)
        if path.suffix in ['.json','.jsonl']:
            content(path);return inputs[name]['sha256']
        if name not in inputs:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
            inputs[name]=dict(sha256=h.hexdigest(),bytes=path.stat().st_size)
        return inputs[name]['sha256']
    def read(path):return json.loads(content(path))
    def rows(path):return [json.loads(line) for line in content(path).splitlines()]
    def verify(path,want):
        actual=sha(path);assert actual==want,(path,actual,want)
        hash_checks.append(dict(path=str(path),sha256=actual,passed=True))
    def write(name,value):
        target=output/name;temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(sanitize(value,workspace),indent=2)+'\n');temp.replace(target)
    split_path=seed_root/'config/split_manifest.json'
    if not split_path.exists():split_path=seed_root.parent/'split_manifest.json'
    split=read(split_path);test_ids=split['airfoil']['primary_test_ids'];assert len(test_ids)==110
    raw_accuracy={};flat=[];registry=[];pending=[];parity=[]
    for arch,backbone in LABELS.items():
        receipt_path=seed_root/f'results/{arch}_accuracy_acceptance.json'
        if not receipt_path.exists():pending.append(f'{arch} accuracy acceptance absent');continue
        receipt=read(receipt_path);assert receipt['passed']
        model=read(seed_root/f'checkpoints/{arch}/manifest.json')
        head_receipt=read(seed_root/f'results/{arch}_head_acceptance.json');assert head_receipt['passed']
        head=read(seed_root/f'checkpoints/{arch}/head/manifest.json')
        assert model['training_seed']==a.seed and head['seed']==a.seed and head['operator_sha256']==model['checkpoint_sha256']
        assert model['normalizer_sha256']==head['normalizer_sha256']==receipt['normalizer_sha256']
        assert model['split_manifest_sha256']==sha(split_path)
        verify(model['checkpoint'],model['checkpoint_sha256']);verify(model['normalizer'],model['normalizer_sha256'])
        verify(seed_root/f'checkpoints/{arch}/head/pressure_d8_best.pt',head['head_sha256'])
        verify(receipt['metrics_csv'],receipt['metrics_sha256'])
        history_path=seed_root/f'checkpoints/{arch}/history.jsonl'
        history=rows(history_path)
        assert len(history)==1000 and [h['epoch'] for h in history]==list(range(1000))
        assert [h['global_step'] for h in history]==[(i+1)*113 for i in range(1000)]
        best_value=float('inf');best_epoch=None
        for h in history:
            value=h['validation_physical_relative_l2'];assert math.isfinite(value) and value>=0
            if value<best_value:best_value=value;best_epoch=h['epoch']
            assert h['best_epoch']==best_epoch and abs(h['best_validation']-best_value)<1e-12
        assert model['best_epoch']==best_epoch and abs(model['best_validation_relative_l2']-best_value)<1e-12
        assert abs(model['training_seconds']-history[-1]['training_seconds'])<1e-6
        identity_receipt=read(seed_root/f'results/{arch}_identity_validation.json')
        assert identity_receipt['passed'] and identity_receipt['checkpoint_sha256']==model['checkpoint_sha256']
        assert identity_receipt['history_sha256']==sha(history_path)
        assert identity_receipt['epochs_completed']==1000 and identity_receipt['updates_completed']==113000 and identity_receipt['optimizer_parameter_steps']==113000
        head_history=head['training_history']
        assert len(head_history)==200 and [h['epoch'] for h in head_history]==list(range(200))
        head_best=float('inf');head_epoch=None
        # This is the original native head's prespecified selection threshold,
        # not a new tolerance introduced to repair an unfavorable result.
        for h in head_history:
            value=h['validation_huber'];assert math.isfinite(value) and value>=0
            if value<head_best-1e-7:head_best=value;head_epoch=h['epoch']
        assert head['best_epoch']==head_epoch and abs(head['best_validation_huber']-head_best)<1e-12
        head_min=min(head_history,key=lambda h:h['validation_huber'])
        head_history_hash=hashlib.sha256(json.dumps(head_history,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        launcher_path=seed_root/f'runtime/{arch}_backbone_launch.json'
        head_launcher_path=seed_root/f'runtime/{arch}_head_launch.json'
        launcher=read(launcher_path) if launcher_path.exists() else {}
        head_launcher=read(head_launcher_path) if head_launcher_path.exists() else {}
        workflow_path=seed_root/f'runtime/{arch}_workflow.json'
        workflow=read(workflow_path) if workflow_path.exists() else {}
        console=Path(launcher['log']) if launcher.get('log') else Path(workflow.get('stages',{}).get('train',{}).get('log',str(seed_root/f'logs/{arch}_train.log')))
        head_console=Path(head_launcher['log']) if head_launcher.get('log') else Path(workflow.get('stages',{}).get('head',{}).get('log',str(seed_root/f'logs/{arch}_head.log')))
        history_audit=dict(backbone_history_epochs=1000,last_global_step=history[-1]['global_step'],backbone_validation_argmin_epoch=best_epoch,backbone_validation_argmin=best_value,head_history_epochs=200,head_selection_epoch=head_epoch,head_validation_literal_min_epoch=head_min['epoch'],head_validation_literal_min=head_min['validation_huber'],head_selection_minus_literal_min=head_best-head_min['validation_huber'],head_rule='native validation Huber improvement greater than1e-7, replayed over all200 epochs',resume_optimizer_steps_source='existing strict identity-validation receipt with actual optimizer_parameter_steps113000')
        local={}
        for path in receipt['source_files']:
            for r in rows(path):
                key=(arch,str(r['case_id']),r['method'],r['sampling_seed'],r['budget_fraction'])
                assert key not in local
                assert r['checkpoint_sha256']==model['checkpoint_sha256'] and r['normalizer_sha256']==model['normalizer_sha256']
                if r['method']=='ca':assert r['head_sha256']==head['head_sha256']
                if r['method']!='full':
                    assert r['agreement']['passed'] and r['agreement']['elementwise_violations']==0
                    parity.append(dict(dataset='airfoil2d',backbone=backbone,training_seed=a.seed,case_id=r['case_id'],method=r['method'],sampling_seed=r['sampling_seed'],budget_fraction=r['budget_fraction'],checkpoint_sha256=model['checkpoint_sha256'],head_sha256=head['head_sha256'] if r['method']=='ca' else None,selected_indices_hash=r['indices_hash'],old_reconstruction_id=r['old_reconstruction_id'],new_reconstruction_id=r['new_reconstruction_id'],max_absolute_difference=r['agreement']['max_abs'],relative_norm_difference=r['agreement']['relative_norm'],elementwise_violations=0,field_metric_difference=r['agreement']['metric_differences']['field_relative_l2'],atol=1e-6,rtol=1e-5,passed=True))
                local[key]=r;raw_accuracy[key]=r
                for impl in ['old','new']:
                    rid='full_native_identity_float64' if r['method']=='full' else r[impl+'_reconstruction_id']
                    for metric in ['field_relative_l2','cd_abs_error','cl_abs_error','cdcl_abs_error']:
                        value=r[impl][metric];assert math.isfinite(value) and value>=0
                        flat.append(dict(dataset='airfoil2d',backbone=backbone,training_seed=a.seed,head_seed=a.seed if r['method']=='ca' else None,checkpoint_sha256=model['checkpoint_sha256'],normalizer_sha256=model['normalizer_sha256'],head_sha256=head['head_sha256'] if r['method']=='ca' else None,case_id=r['case_id'],method=r['method'],sampling_seed=r['sampling_seed'],budget_fraction=r['budget_fraction'],beta=r.get('beta'),implementation=impl,reconstruction_id=rid,metric=metric,error=value))
        expected={(arch,str(case),method,ss,budget) for case in test_ids for method in METHODS for ss in ([None] if method=='full' else [0,1,2]) for budget in ([1.] if method=='full' else BUDGETS)}
        assert set(local)==expected and len(local)==5060
        for key,r in local.items():
            if r['method']=='cpg':
                zero=local[(arch,str(r['case_id']),'cpg_beta0',r['sampling_seed'],r['budget_fraction'])]
                for field in ['indices_hash','coarse_indices_hash','coarse_prediction_hash']:assert r[field]==zero[field]
                assert r['beta']==1 and zero['beta']==0
        registry.append(dict(dataset='airfoil2d',backbone=backbone,model_class=model.get('model_class'),native_model_class=model.get('native_model_class'),native_attention_class=model.get('native_attention_class'),head_class=head.get('architecture'),training_seed=a.seed,head_seed=a.seed,checkpoint_sha256=model['checkpoint_sha256'],head_sha256=head['head_sha256'],normalizer_sha256=model['normalizer_sha256'],split_sha256=sha(split_path),parameters=model['parameter_count'],epochs=model['epochs_completed'],updates=model['total_updates'],validation_score=model['best_validation_relative_l2'],head_validation_score=head['best_validation_huber'],checkpoint_selection=model['validation_criterion'],head_selection=head['checkpoint_selection'],training_wall_seconds=model['training_seconds'],head_training_wall_seconds=head['training_seconds'],GPU_busy_seconds=None,verified=True))
        registry[-1].update(training_history_path=str(history_path),training_history_sha256=sha(history_path),training_log_path=str(console) if console.exists() else None,training_log_sha256=sha(console) if console.exists() else None,training_launch_path=str(launcher_path) if launcher_path.exists() else None,training_launch_sha256=sha(launcher_path) if launcher_path.exists() else None,head_history_path=str(seed_root/f'checkpoints/{arch}/head/manifest.json')+'#training_history',head_history_sha256=head_history_hash,head_manifest_sha256=sha(seed_root/f'checkpoints/{arch}/head/manifest.json'),head_log_path=str(head_console) if head_console.exists() else None,head_log_sha256=sha(head_console) if head_console.exists() else None,head_launch_path=str(head_launcher_path) if head_launcher_path.exists() else None,head_launch_sha256=sha(head_launcher_path) if head_launcher_path.exists() else None,train_loop_seconds=history[-1]['training_seconds'],launcher_stage_wall_seconds=launcher.get('wall_seconds'),launcher_reserved_GPU_seconds=launcher.get('reserved_GPU_seconds'),head_loop_seconds=head['training_seconds'],head_launcher_stage_wall_seconds=head_launcher.get('wall_seconds'),head_reserved_GPU_seconds=head_launcher.get('reserved_GPU_seconds'),training_physical_gpu=launcher.get('physical_gpu',workflow.get('physical_gpu')),head_physical_gpu=head_launcher.get('physical_gpu',workflow.get('physical_gpu')),cost_scope='train_loop_seconds is native recorded training-loop wall; launcher is process/reservation wall for that launch and may cover only a resumed segment; no GPU busy-time claim; unavailable legacy launcher fields remain blank',training_history_verified=True,head_history_verified=True,head_validation_literal_min_epoch=head_min['epoch'],head_selection_epoch=head_epoch)
        hash_checks.append(dict(kind='training_and_head_history_replay',backbone=backbone,training_seed=a.seed,passed=True,**history_audit))
    roots=[]
    for x in [a.legacy_timing_root,a.timing_root]:
        if x is None:continue
        q=x.resolve();roots.append(q if (q/'groups').exists() else q/'timing')
    memory={};groups={};excluded_legacy_conv=0;setups={};source_states={}
    for troot in roots:
        paired_marker=troot/'paired_complete.json'
        paired_done=paired_marker.exists() and read(paired_marker).get('execution_status')=='completed'
        source_states[str(troot)]=dict(paired_complete=paired_done,producer_may_be_running=not paired_done,observation='only completed immutable groups with matching memory are accepted')
        if (troot/'setup.json').exists():setups[troot]=read(troot/'setup.json')
        source_memory_count=0;source_group_conditions=0
        for path in sorted((troot/'memory').glob('*.json')):
            if path.name.endswith('_gpu.json'):
                tele=read(path);assert tele['exclusive'] and not tele['errors']
                if troot in setups:assert tele['physical_gpu']==setups[troot]['physical_gpu']
                continue
            doc=read(path)
            if doc.get('execution_status')!='completed':continue
            for r in doc['rows']:
                if r['backbone'] not in LABELS.values():continue
                source_memory_count+=1
                key=(r['scope'],r['backbone'],str(r['case_id']),r['method'],r['sampling_seed'],r['budget_fraction'],r['implementation'])
                assert key not in memory;memory[key]=r
        if (troot/'gpu_telemetry.json').exists():
            tele=read(troot/'gpu_telemetry.json');assert tele['exclusive'] and not tele['errors']
        for path in sorted((troot/'groups').glob('*.json')):
            if path.name=='heartbeat.json':continue
            doc=read(path)
            if doc.get('execution_status')!='completed' or doc.get('profiles_status')!='completed':
                pending.append('incomplete timing/profile '+str(path));continue
            assert doc['group_id'] not in groups,'duplicate legacy/new group'
            source_group_conditions+=sum(m['backbone'] in LABELS.values() for m in doc['metadata'].values())
            groups[doc['group_id']]=(path,doc,troot)
        source_states[str(troot)].update(completed_primary_group_conditions=source_group_conditions,completed_primary_memory_conditions=source_memory_count,producer_may_be_running=(not paired_done or source_memory_count<source_group_conditions))
    conditions=[];raw_export=[];profile_export=[];reuse=[];seen=set();condition_repeats={}
    for group_id,(path,doc,troot) in groups.items():
        base=doc['order']['base'];first={r['condition']:r for r in doc['first_requests']};summaries={r['condition']:r for r in doc['summary']}
        for r in doc['raw']:assert base[(r['repeat_id']+r['order_position'])%len(base)]==r['condition']
        for condition,meta in doc['metadata'].items():
            if meta['backbone'] not in LABELS.values():excluded_legacy_conv+=1;continue
            assert meta['training_seed']==a.seed
            key=(meta['scope'],meta['backbone'],str(meta['case_id']),meta['method'],meta['sampling_seed'],meta['budget_fraction'],meta['implementation'])
            assert key not in seen;seen.add(key)
            if key not in memory:pending.append('memory absent '+str(key));continue
            m=memory[key]
            for name in ['checkpoint_id','normalizer_id','reconstruction_id','request_code_sha256','runtime_code_sha256','reconstruction_sha256']:assert m[name]==meta[name]
            snapshot=troot/'sources'/meta['source_snapshot_id']
            for field,name in [('request_code_sha256','request.py'),('runtime_code_sha256','runtime.py'),('benchmark_code_sha256','benchmark.py'),('timing_core_sha256','benchmark_core.py')]:verify(snapshot/name,meta[field])
            if 'p0_adapter_sha256' in meta:verify(snapshot/'p0_benchmark_adapter.py',meta['p0_adapter_sha256'])
            series=[r for r in doc['raw'] if r['condition']==condition]
            formal=sorted([r for r in series if r['kind']=='formal'],key=lambda r:r['repeat_id']);warm=[r for r in series if r['kind']=='warmup']
            assert len(formal)==100 and len(warm)==20 and {r['repeat_id'] for r in formal}==set(range(100))
            values=np.array([r['total_ms'] for r in formal]);assert np.isfinite(values).all() and (values>0).all()
            assert np.isclose(np.median(values),summaries[condition]['median_ms'],atol=1e-10,rtol=1e-12)
            assert np.isclose(np.percentile(values,95),summaries[condition]['p95_ms'],atol=1e-10,rtol=1e-12)
            output_check=first[condition]['output_check'];arch=next(k for k,v in LABELS.items() if v==meta['backbone'])
            accuracy_key=(arch,str(meta['case_id']),meta['method'],meta['sampling_seed'],meta['budget_fraction'])
            if accuracy_key not in raw_accuracy:pending.append('accuracy absent '+str(accuracy_key));continue
            reference=raw_accuracy[accuracy_key];assert reference['checkpoint_sha256']==meta['checkpoint_id']
            impl='old' if meta['implementation'] in ['old','full_shared'] else 'new'
            assert abs(output_check['field_error']-reference[impl]['field_relative_l2'])<=1e-8+1e-5*abs(reference[impl]['field_relative_l2'])
            if meta['method']=='full':assert meta['sampling_seed'] is None
            else:
                assert output_check['selected_indices_hash']==reference['indices_hash']
                if meta['method'] in ['cpg','cpg_beta0']:assert output_check['coarse_prediction_hash']==reference['coarse_prediction_hash']
            components=doc['profiles'][condition]
            reconstruction=[r['ms'] for r in components if r['component']=='reconstruction_complete_output']
            assert meta['method']=='full' or len(reconstruction)==5
            row={**meta,'group_id':group_id,'condition':condition,'direct_median_ms':float(np.median(values)),'direct_p95_ms':float(np.percentile(values,95)),'timing_repeat_sd_ms':float(np.std(values,ddof=1)),'formal_repeats':100,'warmups':20,'reconstruction_profile_median_ms':float(np.median(reconstruction)) if reconstruction else 0.,'first_request_ms':first[condition]['first_request_ms'],'timing_case_error':output_check['field_error'],'selected_indices_hash':output_check['selected_indices_hash'],'cached_map_cpu_bytes':output_check.get('cached_map_cpu_bytes'),'cached_map_gpu_bytes':output_check.get('cached_map_gpu_bytes'),'peak_cuda_allocated_bytes':m['peak_cuda_allocated_bytes'],'peak_cuda_reserved_bytes':m['peak_cuda_reserved_bytes'],'source_json':str(path),'source_sha256':sha(path),'status':'VERIFIED_COMPLETE'}
            setup=setups[troot]
            row.update(physical_gpu=setup['physical_gpu'],cpu_affinity=json.dumps(setup['cpu_affinity']),cpu_threads=setup['cpu_threads'],
                reconstruction_profile_p95_ms=float(np.percentile(reconstruction,95)) if reconstruction else 0.,
                first_request_peak_cuda_allocated_bytes=m['first_request_peak_cuda_allocated_bytes'],
                first_request_peak_cuda_reserved_bytes=m['first_request_peak_cuda_reserved_bytes'],
                allocator_scope=m['allocator_scope'],external_memory_scope=m['external_scope'],order_base_json=json.dumps(base))
            conditions.append(row);condition_repeats[key]=values
            for r in series:raw_export.append(dict(training_seed=a.seed,group_id=group_id,**r))
            for r in components:profile_export.append(dict(training_seed=a.seed,group_id=group_id,condition=condition,**r))
            if meta['scope'].startswith('C_') and meta['method']!='full':
                for uses in [1,2,5,10,20,50,100]:
                    reuse.append({k:row[k] for k in ['training_seed','backbone','case_id','method','sampling_seed','budget_fraction','implementation','checkpoint_id']}|dict(uses=uses,first_request_ms=row['first_request_ms'],steady_median_ms=row['direct_median_ms'],derived_total_ms=row['first_request_ms']+(uses-1)*row['direct_median_ms'],derived_per_request_ms=(row['first_request_ms']+(uses-1)*row['direct_median_ms'])/uses,cached_map_cpu_bytes=row['cached_map_cpu_bytes'],cached_map_gpu_bytes=row['cached_map_gpu_bytes'],scope='C_cpu_cpu_cached_map',cost_kind='derived first+(uses-1)*steady median; not contiguous wall measurement'))
    expected={(scope,bb,str(case),method,ss,budget,impl) for scope in SCOPES for bb in LABELS.values() for case in CASES for method in METHODS for ss in ([None] if method=='full' else [0,1,2]) for budget in ([1.] if method=='full' else BUDGETS) for impl in (['full_shared'] if method=='full' else ['old','new'])}
    available={(r['scope'],r['backbone'],str(r['case_id']),r['method'],r['sampling_seed'],r['budget_fraction'],r['implementation']) for r in conditions}
    assert available<=expected
    missing=expected-available;complete=not missing and len(registry)==2 and len(raw_accuracy)==10120
    if missing:pending.append(f'{len(missing)} of1638 physical timing/memory conditions absent')
    receipt=dict(status='VERIFIED_COMPLETE' if complete else 'PARTIAL',training_seed=a.seed,accuracy_requests=len(raw_accuracy),expected_accuracy_requests=10120,physical_conditions=len(conditions),expected_conditions=1638,formal_repeats=len(conditions)*100,warmups=len(conditions)*20,matched_memory_conditions=len(conditions),expected_groups=414,observed_groups=len(groups),excluded_legacy_conv_conditions=excluded_legacy_conv,pending=pending,missing_conditions=[list(k) for k in sorted(missing,key=str)],hash_checks=hash_checks,source_inputs=inputs,executed_script_sha256=sha(Path(__file__)),statistics='Case and sampling aggregation remain within each model; cross-training SD is computed only by aggregate_three_seeds.py after3 independent accepted seeds.')
    receipt['source_states']=source_states
    receipt['input_snapshot_binding']='Each JSON/JSONL path is parsed and SHA256-hashed from the same first-read bytes, cached for this audit invocation.'
    if not complete and not a.allow_partial:
        write('acceptance.json',receipt)
        print(json.dumps({'status':'PARTIAL','pending':pending},indent=2));raise SystemExit(2)
    for name,values in [('checkpoint_registry.csv',registry),('accuracy_records.csv.gz',flat),('reconstruction_parity.csv.gz',parity),('latency_conditions.csv',conditions),('timing_repeats.csv.gz',raw_export),('stage_profiles.csv.gz',profile_export),('cache_reuse.csv',reuse)]:
        csv_write(output/name,sanitize(values,workspace))
    # Deterministic sufficient statistics preserve sampling seeds before any
    # cross-training aggregation. Full retains a null sampling-seed field.
    acc_groups=collections.defaultdict(list)
    keys=['backbone','training_seed','checkpoint_sha256','normalizer_sha256','implementation','reconstruction_id','method','beta','metric','budget_fraction','sampling_seed']
    for r in flat:acc_groups[tuple(r[k] for k in keys)].append(r['error'])
    means=[]
    for key,values in acc_groups.items():
        assert len(values)==110;means.append(dict(zip(keys,key))|dict(case_mean=float(np.mean(values)),case_count=110))
    csv_write(output/'case_sampling_means.csv',means)
    latency_groups=collections.defaultdict(list)
    for r in conditions:
        key=(r['training_seed'],r['scope'],r['backbone'],r['method'],r['budget_fraction'],r['implementation'])
        identity=(r['scope'],r['backbone'],str(r['case_id']),r['method'],r['sampling_seed'],r['budget_fraction'],r['implementation'])
        latency_groups[key].extend(condition_repeats[identity])
    grouped=[]
    for key,values in latency_groups.items():grouped.append(dict(zip(['training_seed','scope','backbone','method','budget_fraction','implementation'],key))|dict(pooled_median_ms=float(np.median(values)),pooled_p95_ms=float(np.percentile(values,95)),formal_observations=len(values),timing_cases=3))
    csv_write(output/'latency_grouped.csv',grouped)
    receipt['data_files']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir()) if p.name.endswith(('.csv','.csv.gz'))}
    receipt['parity_pairs']=len(parity)
    receipt['accuracy_complete']=len(raw_accuracy)==10120 and len(registry)==2
    receipt['full_reference_requests']=sum(r['method']=='full' for r in raw_accuracy.values())
    receipt['physical_gpus']=sorted({r['physical_gpu'] for r in conditions})
    assert len(receipt['physical_gpus'])<=1,'Never pool one seed over multiple physical GPUs'
    receipt['seed_components']=dict(existing_verified_conditions=sum(r['method']!='cpg_beta0' for r in conditions) if a.seed==0 else 0,beta0_conditions_verified=sum(r['method']=='cpg_beta0' for r in conditions),expected_conditions=1638)
    write('acceptance.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','training_seed','accuracy_requests','physical_conditions','formal_repeats','warmups','excluded_legacy_conv_conditions']},indent=2))


if __name__=='__main__':main()
