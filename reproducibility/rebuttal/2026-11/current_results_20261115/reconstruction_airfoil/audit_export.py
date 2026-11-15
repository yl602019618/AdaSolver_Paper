#!/usr/bin/env python3
"""Read immutable Car/Airfoil measurements and export audited current results.

CPU-only. All output is confined to --output. The failed original Airfoil
finalizer is preserved; its duplicate-key reporting bug is bypassed here with
explicit dictionary merging and independent repeat/count/identity checks.
"""
import argparse,collections,csv,datetime,hashlib,json,math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root,out=args.workspace.resolve(),args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    car=root/'linearno_followup_20261114/reconstruction';air=root/'linearno_followup_20261114/standard_benchmarks/airfoil'
    inputs,checks,counts={},[],{}
    def public(v):
        if isinstance(v,str):return v.replace(str(root)+'/','workspace:').replace(str(root),'workspace:')
        if isinstance(v,list):return [public(x) for x in v]
        if isinstance(v,dict):return {public(k):public(x) for k,x in v.items()}
        return v
    def sha(path):
        path=Path(path);key=str(path.relative_to(root))
        if key not in inputs:
            data=path.read_bytes();inputs[key]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        return inputs[key]['sha256']
    def check(path,want,label):
        actual=sha(path);assert actual==want,(label,path,actual,want)
        checks.append({'kind':label,'path':str(Path(path).relative_to(root)),'sha256':actual,'passed':True})
    def readj(path):sha(path);return json.loads(Path(path).read_text())
    def readc(path):
        sha(path)
        with Path(path).open(newline='') as f:return list(csv.DictReader(f))
    def readl(path):sha(path);return [json.loads(line) for line in Path(path).read_text().splitlines()]
    def writec(name,rows):
        assert rows,name
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(public(rows))
    def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b)
    def safe_seed(value):return None if value in [None,'','N/A','None'] else int(value)
    def identity(r):
        return (r['dataset'],r['backbone'],str(r['case_id']),r['method'],float(r['budget_fraction']),safe_seed(r['sampling_seed']),r['implementation'],r['scope'])
    def telemetry(path):
        doc=readj(path);assert doc.get('exclusive') is True and not doc.get('errors')
        return {'path':str(path.relative_to(root)),'pid':doc['pid'],'exclusive':True,'samples':len(doc.get('samples',[]))}
    def aggregate_accuracy(flat,prefix):
        groups=collections.defaultdict(list)
        keys=['dataset','backbone','checkpoint_sha256','normalizer_sha256','implementation','reconstruction_id','method','beta','metric','budget_fraction','sampling_seed']
        for r in flat:groups[tuple(r[k] for k in keys)].append(r)
        seed_rows=[]
        for key,rs in groups.items():
            assert len({r['case_id'] for r in rs})==len(rs)
            row=dict(zip(keys,key));row.update(error_mean=float(np.mean([r['error'] for r in rs])),case_count=len(rs),training_seed=0)
            seed_rows.append(row)
        grouped=collections.defaultdict(list)
        for r in seed_rows:grouped[tuple(r[k] for k in keys[:-1])].append(r)
        budgets=[]
        for key,rs in grouped.items():
            row=dict(zip(keys[:-1],key));v=[r['error_mean'] for r in rs]
            assert len(v)==(1 if row['method']=='full' else 3)
            row.update(error_mean=float(np.mean(v)),sampling_sd=float(np.std(v,ddof=1)) if len(v)>1 else '',case_count=rs[0]['case_count'],training_seed=0,sampling_seed_count=0 if len(v)==1 else 3)
            budgets.append(row)
        lookup={(r['dataset'],r['backbone'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'],r['method'],r['beta'],r['metric'],r['budget_fraction'],r['sampling_seed']):r['error_mean'] for r in seed_rows}
        ratio_groups=collections.defaultdict(list)
        for r in budgets:
            if r['method']=='full':continue
            k=tuple(r[k] for k in keys[:9]);ratio_groups[k].append(r)
        rauec=[]
        for key,rs in ratio_groups.items():
            r=rs[0];assert sorted(x['budget_fraction'] for x in rs)==[.375,.5,.75];ratios=[];denominators=[]
            for s in [0,1,2]:
                base=(r['dataset'],r['backbone'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'])
                y=[lookup[(*base,r['method'],r['beta'],r['metric'],b,s)] for b in [.375,.5,.75]]
                u=[lookup[(*base,'uniform',None,r['metric'],b,s)] for b in [.375,.5,.75]]
                area=float(np.trapezoid(u,[.375,.5,.75]));denominators.append(area);ratios.append(float(np.trapezoid(y,[.375,.5,.75]))/area)
            row={k:r[k] for k in keys[:9]};row.update(training_seed=0,budget_grid='0.375;0.5;0.75',case_count=r['case_count'],mean=float(np.mean(ratios)),sampling_sd=float(np.std(ratios,ddof=1)))
            for s in [0,1,2]:row[f'sampling_seed{s}']=ratios[s];row[f'matched_uniform_area_seed{s}']=denominators[s]
            rauec.append(row)
        writec(prefix+'_accuracy_budget.csv',budgets);writec(prefix+'_sampling_seed_errors.csv',seed_rows);writec(prefix+'_rauec.csv',rauec)
        return budgets,rauec
    # Car original cohort: exactly matched old/new predictions on 100 cases.
    protocol=readj(car/'config/car_protocol.json');delivery=readj(car/'results/car_delivery_receipt.json')
    check(car/'records/car_accuracy/pairs.jsonl',delivery['raw_accuracy_sha256'],'Car accuracy receipt')
    check(car/'results/car_timing_repeats.csv',delivery['raw_repeats_sha256'],'Car repeat export receipt')
    pairs=readl(car/'records/car_accuracy/pairs.jsonl');assert len(pairs)==7400
    expected={(bb,case,m,s,b) for bb,methods in protocol['accuracy_variants'].items() for case in protocol['test_ids'] for m in methods for s in ([None] if m=='full' else [0,1,2]) for b in ([1.] if m=='full' else [.375,.5,.75])}
    paired={(r['backbone'],r['case_id'],r['variant'],r['sampling_seed'],r['budget_fraction']):r for r in pairs}
    assert set(paired)==expected and len(paired)==len(pairs)
    car_flat=[];car_parity=[];beta_pairs=0
    for r in pairs:
        assert r['difference']['violations']==0 and r['training_seed']==0 and r['output_dtype']=='float64'
        if r['variant']=='cpg_beta0':
            mate=paired[(r['backbone'],r['case_id'],'cpg_beta1',r['sampling_seed'],r['budget_fraction'])]
            assert r['selected_indices_sha256']==mate['selected_indices_sha256'] and r['coarse_prediction_sha256']==mate['coarse_prediction_sha256'];beta_pairs+=1
        car_parity.append({k:r[k] for k in ['backbone','case_id','variant','sampling_seed','budget_fraction','operator_sha256','selected_indices_sha256','old_reconstruction_id','new_reconstruction_id']}|{'max_abs':r['difference']['max_abs'],'relative_norm':r['difference']['relative_norm'],'violations':0,'field_metric_difference':r['metric_differences']['field_error'],'atol':1e-6,'rtol':1e-5,'status':'verified'})
        for impl in ['old','new']:
            for metric in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error']:
                car_flat.append(dict(dataset='shapenet_car',backbone=r['backbone'],case_id=r['case_id'],checkpoint_sha256=r['operator_sha256'],normalizer_sha256=r['normalizer_sha256'],implementation=impl,reconstruction_id=r[impl+'_reconstruction_id'],method=r['variant'],beta=r['beta'],metric=metric,budget_fraction=r['budget_fraction'],sampling_seed=r['sampling_seed'],error=r[impl+'_metrics'][metric]))
    assert beta_pairs==1800
    car_budgets,car_rauec=aggregate_accuracy(car_flat,'car');writec('car_reconstruction_parity.csv',car_parity)
    conditions=readc(car/'results/reconstruction_summary.csv');assert len(conditions)==990
    car_raw={};car_profiles={};car_repeat_export=[];car_sources={};car_memory={}
    for p in (car/'records/car_memory').glob('*.json'):
        doc=readj(p)
        if 'rows' not in doc:continue
        assert doc['execution_status']=='completed'
        for r in doc['rows']:car_memory[(r['scope'],r['backbone'],r['variant'],r['implementation'],str(r['case_id']),float(r['budget_fraction']),r['sampling_seed'])]=r
    assert len(car_memory)==990
    for scope in ['A','B','C']:
        car_sources[scope]=telemetry(car/'records/car_timing'/scope/'gpu_telemetry.json')
        source=readj(car/'records/car_timing'/scope/'source_manifest.json')
        for name,h in source.items():check(car/'code'/name,h,'Car frozen timing source')
    for row in conditions:
        path=car/row['raw_source'];s=safe_seed(row['sampling_seed']);b=float(row['budget_fraction'])
        if path not in car_raw:
            raw=readc(path);doc=readj(path.parent/'complete.json');assert doc['execution_status']=='completed'
            car_profiles[path]=readj(path.parent/'profiles.json')
            assert readj(path.parent/'alignment.json')['passed'];order=readj(path.parent/'order.json')['base']
            for x in raw:
                assert order[(int(x['repeat_id'])+int(x['order_position']))%len(order)]==x['condition']
                car_repeat_export.append(dict(source_group=str(path.parent.relative_to(car)),case_id=doc['case_id'],budget_fraction=doc['budget_fraction'],sampling_seed=doc['sampling_seed'],**x))
            car_raw[path]=raw
        raw=[x for x in car_raw[path] if x['condition']==row['condition']]
        formal=[x for x in raw if x['kind']=='formal'];warm=[x for x in raw if x['kind']=='warmup']
        assert len(formal)==100 and len(warm)==20 and {int(x['repeat_id']) for x in formal}==set(range(100))
        values=[float(x['total_ms']) for x in formal];assert np.isfinite(values).all() and min(values)>0
        close(np.median(values),row['direct_total_ms']);close(np.percentile(values,95),row['p95_ms'])
        profiles_for_condition=[x for x in car_profiles[path] if x['condition']==row['condition'] and (row['scope']!='C' or row['variant']=='full' or x['cache_hit'])]
        reconstruction_profile=[float(x['ms']) for x in profiles_for_condition if x['component']=='reconstruction_pipeline_ms']
        assert reconstruction_profile
        close(np.median(reconstruction_profile),row['reconstruction_ms'])
        row.update(reconstruction_profile_p95_ms=float(np.percentile(reconstruction_profile,95)),reconstruction_profile_observations=len(reconstruction_profile))
        ref=paired[(row['backbone'],row['case_id'],row['variant'],s,b)]
        assert row['checkpoint_sha256']==ref['operator_sha256'] and row['selected_indices_sha256']==ref['selected_indices_sha256']
        mem=car_memory[(row['scope'],row['backbone'],row['variant'],row['implementation'],row['case_id'],b,s)]
        assert int(row['peak_memory_allocated_bytes'])==mem['peak_cuda_allocated_bytes']
        row.update(training_seed=0,error_on_timing_case=ref['old_metrics' if row['implementation'] in ['old','shared'] else 'new_metrics']['field_error'],status='verified_from_raw_repeats',source_sha256=sha(path))
    writec('car_reconstruction_conditions.csv',conditions);writec('car_raw_timing_repeats.csv',car_repeat_export)
    for name in ['car_latency_grouped.csv','car_old_new_speedups.csv','car_reuse_costs.csv']:
        rows=readc(car/'results'/name)
        if name=='car_reuse_costs.csv':
            for r in rows:close(float(r['first_request_ms'])+(int(r['uses'])-1)*float(r['subsequent_request_median_ms']),r['estimated_total_ms'])
        writec(name,rows)
    counts['car']={'accuracy_pairs':7400,'beta_pairs':1800,'timing_conditions':990,'formal_repeats':99000,'warmups':19800,'memory_conditions':990,'timing_raw_groups':len(car_raw),'status':'VERIFIED_COMPLETE within the existing seed0 protocol','missing':'LinearNO PI/CA not in this protocol; no extra training-seed claim'}
    # Airfoil three completed models: variable-point T3/point-temp and Full-only conv-temp.
    air_flat=[];air_lookup={};air_parity=[];air_registry=[]
    for arch in ['t3','linearno','linearno_conv']:
        model=readj(air/f'checkpoints/{arch}/manifest.json');accept=readj(air/f'results/{arch}_accuracy_acceptance.json')
        assert accept['passed'];check(model['checkpoint'],model['checkpoint_sha256'],'Airfoil checkpoint');check(model['normalizer'],model['normalizer_sha256'],'Airfoil normalizer')
        check(accept['metrics_csv'],accept['metrics_sha256'],'Airfoil accepted accuracy CSV')
        sources=accept['source_files'];raws={}
        for path in sources:
            for r in readl(path):
                k=(str(r['case_id']),r['method'],r['sampling_seed'],r['budget_fraction']);assert k not in raws;raws[k]=r
                assert r['checkpoint_sha256']==model['checkpoint_sha256']
                if r['method']!='full':
                    assert r['agreement']['passed'] and r['agreement']['elementwise_violations']==0
                    air_parity.append(dict(backbone=arch,case_id=r['case_id'],method=r['method'],budget_fraction=r['budget_fraction'],sampling_seed=r['sampling_seed'],checkpoint_sha256=r['checkpoint_sha256'],selected_indices_hash=r['indices_hash'],max_abs=r['agreement']['max_abs'],relative_norm=r['agreement']['relative_norm'],field_metric_difference=r['agreement']['metric_differences']['field_relative_l2'],atol=1e-6,rtol=1e-5,status='verified'))
                air_lookup[(model['checkpoint_sha256'],*k)]=r
        assert len(raws)==accept['expected_prediction_requests']
        for r in readc(accept['metrics_csv']):
            ss=safe_seed(r['sampling_seed']);bb=float(r['budget_fraction']);ref=raws[(r['case_id'],r['method'],ss,bb)]
            impl='old' if r['reconstruction_id']=='grid_scipy_linear_nearest_native_v1' else 'new'
            close(r['value'],ref[impl][r['metric']])
            beta=None if not r['beta'] else float(r['beta'])
            air_flat.append(dict(dataset='airfoil2d',backbone=r['backbone'],case_id=r['case_id'],checkpoint_sha256=r['checkpoint_id'],normalizer_sha256=r['normalizer_sha256'],implementation=impl,reconstruction_id=r['reconstruction_id'],method=r['method'],beta=beta,metric=r['metric'],budget_fraction=bb,sampling_seed=ss,error=float(r['value'])))
        reg={k:model[k] for k in ['architecture','architecture_variant','model_class','native_model_class','native_attention_class','training_seed','epochs_completed','total_updates','parameter_count','checkpoint_sha256','normalizer_sha256','best_epoch','best_validation_relative_l2','training_seconds']}
        reg.update(status='verified',accuracy_requests=len(raws),variable_point_support=arch!='linearno_conv',test_population='published110 positive-lift cases; historically exposed')
        if arch!='linearno_conv':
            head=readj(air/f'checkpoints/{arch}/head/manifest.json');check(air/f'checkpoints/{arch}/head/pressure_d8_best.pt',head['head_sha256'],'Airfoil own head')
            reg.update(head_sha256=head['head_sha256'],head_seed=head['seed'],head_training_seconds=head['training_seconds'])
        air_registry.append(reg)
    air_budgets,air_rauec=aggregate_accuracy(air_flat,'airfoil');writec('airfoil_reconstruction_parity.csv',air_parity);writec('airfoil_checkpoint_registry.csv',air_registry)
    setup=readj(air/'timing/setup.json');snapshot=air/'timing/sources'/setup['source_snapshot_id']
    for name,h in setup['sources'].items():check(snapshot/name,h,'Airfoil executed timing snapshot')
    air_gpu=telemetry(air/'timing/gpu_telemetry.json');memories={};memory_telemetry=[]
    for path in (air/'timing/memory').glob('*.json'):
        if path.name.endswith('_gpu.json'):memory_telemetry.append(telemetry(path));continue
        doc=readj(path);assert doc['execution_status']=='completed'
        for r in doc['rows']:assert identity(r) not in memories;memories[identity(r)]=r
    assert len(memories)==1323
    arows=[];repeats=[];profiles=[];reuse=[];pooled=collections.defaultdict(list);profile_pooled=collections.defaultdict(list)
    group_paths=[p for p in sorted((air/'timing/groups').glob('*.json')) if p.name!='heartbeat.json'];assert len(group_paths)==333
    for path in group_paths:
        group=readj(path);assert group['execution_status']=='completed' and group['profiles_status']=='completed'
        first={r['condition']:r for r in group['first_requests']};saved={r['condition']:r for r in group['summary']}
        base=group['order']['base']
        for r in group['raw']:
            assert base[(int(r['repeat_id'])+int(r['order_position']))%len(base)]==r['condition']
            repeats.append(dict(group_id=group['group_id'],**r))
        for condition,meta in group['metadata'].items():
            raw=[r for r in group['raw'] if r['condition']==condition];formal=[r for r in raw if r['kind']=='formal'];warm=[r for r in raw if r['kind']=='warmup']
            assert len(formal)==100 and len(warm)==20 and {r['repeat_id'] for r in formal}==set(range(100))
            values=np.array([r['total_ms'] for r in formal]);assert np.isfinite(values).all() and (values>0).all()
            close(np.median(values),saved[condition]['median_ms']);close(np.percentile(values,95),saved[condition]['p95_ms'])
            mem=memories[identity(meta)]
            assert meta['source_snapshot_id']==setup['source_snapshot_id']
            for field,name in [('request_code_sha256','request.py'),('runtime_code_sha256','runtime.py'),('benchmark_code_sha256','benchmark.py'),('timing_core_sha256','benchmark_core.py')]:
                assert meta[field]==setup['sources'][name]
            profile_meta=group['profile_metadata'][condition]
            for field in ['request_code_sha256','runtime_code_sha256','reconstruction_sha256']:
                assert profile_meta[field]==meta[field]
            for field in ['checkpoint_id','normalizer_id','reconstruction_id','request_code_sha256','runtime_code_sha256','reconstruction_sha256']:assert mem[field]==meta[field]
            check_out=first[condition]['output_check']
            ref=air_lookup[(meta['checkpoint_id'],str(meta['case_id']),meta['method'],meta['sampling_seed'],meta['budget_fraction'])]
            impl='old' if meta['implementation'] in ['old','full_shared'] else 'new'
            close(check_out['field_error'],ref[impl]['field_relative_l2'],1e-8)
            if meta['method']=='full':assert meta['sampling_seed'] is None
            else:assert check_out['selected_indices_hash']==ref['indices_hash']
            components=group['profiles'][condition]
            recon=[r['ms'] for r in components if r['component']=='reconstruction_complete_output']
            assert meta['method']=='full' or len(recon)==5
            # Explicit merging avoids the original repeated reuse_assumption key.
            row={**meta,'group_id':group['group_id'],'condition':condition,'direct_total_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'mean_ms':float(values.mean()),'timing_repeat_sd_ms':float(values.std(ddof=1)),'formal_repeats':100,'warmups':20,
                'reconstruction_ms':0. if meta['method']=='full' else float(np.median(recon)),'reconstruction_p95_ms':0. if meta['method']=='full' else float(np.percentile(recon,95)),
                'first_request_ms':first[condition]['first_request_ms'],'field_error_on_timing_case':check_out['field_error'],'selected_indices_hash':check_out['selected_indices_hash'],
                'cached_map_cpu_bytes':check_out.get('cached_map_cpu_bytes'),'cached_map_gpu_bytes':check_out.get('cached_map_gpu_bytes'),
                'peak_cuda_allocated_bytes':mem['peak_cuda_allocated_bytes'],'peak_cuda_reserved_bytes':mem['peak_cuda_reserved_bytes'],
                'first_request_peak_cuda_allocated_bytes':mem['first_request_peak_cuda_allocated_bytes'],'first_request_peak_cuda_reserved_bytes':mem['first_request_peak_cuda_reserved_bytes'],
                'allocator_scope':mem['allocator_scope'],'external_scope':mem['external_scope'],'source_json':str(path),'source_sha256':sha(path),'status':'verified_from_raw'}
            arows.append(row)
            pk=(meta['scope'],meta['backbone'],meta['method'],meta['budget_fraction'],meta['implementation'])
            pooled[pk].extend(values.tolist());profile_pooled[pk].extend(recon)
            for prof in components:profiles.append({**meta,'group_id':group['group_id'],'condition':condition,**prof})
            if meta['scope'].startswith('C_') and meta['method']!='full':
                cold=[r['ms'] for r in components if r['component']=='first_cache_map_build']
                for use_count in [1,2,5,10,20,50,100]:
                    reuse.append({**meta,'requests':use_count,'first_request_ms':row['first_request_ms'],'steady_median_ms':row['direct_total_ms'],'modeled_total_ms':row['first_request_ms']+(use_count-1)*row['direct_total_ms'],
                        'assembled_observation_sum_ms':row['first_request_ms']+sum(r['total_ms'] for r in sorted(formal,key=lambda r:r['repeat_id'])[:use_count-1]),
                        'separate_map_build_ms':cold[0] if cold else None,'cached_map_cpu_bytes':row['cached_map_cpu_bytes'],'cached_map_gpu_bytes':row['cached_map_gpu_bytes'],
                        'cost_kind':'derived cold first request + post-warmup observations/median; not contiguous end-to-end measurement'})
    assert len(arows)==1323 and len({identity(r) for r in arows})==1323
    assert sum(r['kind']=='formal' for r in repeats)==132300
    writec('airfoil_reconstruction_conditions.csv',arows);writec('airfoil_raw_timing_repeats.csv',repeats);writec('airfoil_stage_profiles.csv',profiles);writec('airfoil_cache_reuse_costs.csv',reuse)
    grouped=[]
    for key,values in sorted(pooled.items()):
        scope,bb,method,budget,impl=key;rs=[r for r in arows if (r['scope'],r['backbone'],r['method'],r['budget_fraction'],r['implementation'])==key]
        grouped.append(dict(dataset='airfoil2d',scope=scope,backbone=bb,method=method,budget_fraction=budget,implementation=impl,direct_total_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),reconstruction_profile_median_ms=float(np.median(profile_pooled[key])) if profile_pooled[key] else 0.,cases=3,sampling_seeds=0 if method=='full' else 3,physical_conditions=len(rs),formal_observations=len(values),peak_memory_bytes=max(r['peak_cuda_allocated_bytes'] for r in rs),first_request_ms=float(np.mean([r['first_request_ms'] for r in rs])),metric_scope='timing cases only; full110-case accuracy is separate',timing_case_error=float(np.mean([r['field_error_on_timing_case'] for r in rs]))))
    writec('airfoil_latency_grouped.csv',grouped)
    speedups=[]
    for r in grouped:
        if r['implementation']!='old':continue
        other=next(s for s in grouped if all(s[k]==r[k] for k in ['scope','backbone','method','budget_fraction']) and s['implementation']=='new')
        speedups.append({k:r[k] for k in ['dataset','scope','backbone','method','budget_fraction']}|dict(old_median_ms=r['direct_total_ms'],new_median_ms=other['direct_total_ms'],old_over_new_speedup=r['direct_total_ms']/other['direct_total_ms'],new_slower=other['direct_total_ms']>r['direct_total_ms']))
    writec('airfoil_old_new_speedups.csv',speedups)
    # Portable plotting join: full-population error and independently measured
    # three-case latency retain the same checkpoint and reconstruction arm.
    accuracy_latency=[]
    for dataset,table,errors in [('car',readc(car/'results/car_latency_grouped.csv'),car_budgets),('airfoil',grouped,air_budgets)]:
        metric='field_error' if dataset=='car' else 'field_relative_l2'
        for t in table:
            method=t['variant'] if dataset=='car' else t['method']
            impl=t['implementation'] if t['implementation'] in ['old','new'] else 'new'
            e=next(r for r in errors if r['backbone']==t['backbone'] and r['method']==method and r['implementation']==impl and r['metric']==metric and float(r['budget_fraction'])==float(t['budget_fraction']))
            accuracy_latency.append(dict(dataset=e['dataset'],backbone=e['backbone'],method=method,beta=e['beta'],training_seed=0,budget_fraction=t['budget_fraction'],implementation=t['implementation'],scope=t['scope'],metric=metric,error=e['error_mean'],sampling_sd=e['sampling_sd'],accuracy_cases=e['case_count'],timing_cases=3,checkpoint_sha256=e['checkpoint_sha256'],accuracy_reconstruction_id=e['reconstruction_id'],complete_latency_median_ms=t['direct_total_ms'],complete_latency_p95_ms=t['p95_ms'],formal_repeats_per_condition=100,warmups_per_condition=20,peak_cuda_allocated_bytes=t.get('peak_memory_bytes',t.get('peak_memory_allocated_bytes')),reconstruction_binding='same mathematical rule/checkpoint/selected set; cache implementation explicit by scope; old/new parity verified'))
    writec('accuracy_latency_summary.csv',accuracy_latency)
    counts['airfoil']={'accuracy_requests':sum(r['accuracy_requests'] for r in air_registry),'timing_groups':333,'timing_conditions':1323,'formal_repeats':132300,'warmups':26460,'memory_conditions':1323,'status':'VERIFIED_COMPLETE within seed0 and Full-only conv protocol','original_failure':'finalize_latency.py dict(**meta, reuse_assumption=...) duplicate key; numeric data were complete','reporting_recovery':'new independent output only; original source/status/log untouched'}
    audit={'status':'passed','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only CPU current-results audit and independent aggregation; no training or GPU execution','executed_audit_script_sha256':sha(Path(__file__)),'counts':counts,'hash_checks':checks,'gpu_exclusivity':{'car':car_sources,'airfoil':air_gpu,'airfoil_memory':memory_telemetry},'source_inputs':inputs,
        'original_airfoil_failure_log':str((air/'logs/postprocess_latency_aggregation.log').relative_to(root)),
        'latency_boundaries':{'A':'CPU input -> complete CPU float64 field; each request constructs its map','B':'resident GPU model tensors + explicit CPU geometry -> complete GPU float64 field','C':'CPU -> CPU with same-geometry/ordered-point-set interpolation-map cache only; model/score/selection still recomputed'},
        'limits':['All models here use training seed0; sampling seeds0/1/2 are not independent training repeats.','Conv-temp Airfoil is Full-only and is not relabeled point-temp.','Full has no sampling seed and is excluded from rAUEC.','Caching first-plus-steady totals are derived estimates, not directly measured contiguous request batches.','Allocator peaks exclude CPU and transient external GPU memory; post-request NVML is separately bounded.','Car LinearNO PI/CA are unavailable in this preserved protocol.','Airfoil original finalizer failed only in reporting; this recovery does not rewrite its old failure status.']}
    (out/'audit.json').write_text(json.dumps(public(audit),indent=2)+'\n')
    writec('source_index.csv',[dict(path=k,**v) for k,v in inputs.items()])
    print(json.dumps({'counts':counts,'hash_checks':len(checks),'input_files':len(inputs),'airfoil_new_slower_group_count':sum(r['new_slower'] for r in speedups),'airfoil_speedup_range':[min(r['old_over_new_speedup'] for r in speedups),max(r['old_over_new_speedup'] for r in speedups)]},indent=2))


if __name__=='__main__':main()
