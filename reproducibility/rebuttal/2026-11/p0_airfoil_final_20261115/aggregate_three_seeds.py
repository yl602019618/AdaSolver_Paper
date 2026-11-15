#!/usr/bin/env python3
"""Aggregate accepted independent Airfoil training seeds without mixing GPUs."""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,math,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from independent_accept import csv_write

EXPECTED=dict(models=6,heads=6,training_seeds=3,accuracy_requests=30360,physical_conditions=4914,formal_repeats=491400,warmups=98280,memory_conditions=4914,parity_pairs=29700,full_reference_requests=660)


def read_csv(path):
    if not path.exists():return []
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--accepted-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    accepted=a.accepted_dir.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    receipts={};inputs={};tables=collections.defaultdict(list);missing=[]
    filenames=['checkpoint_registry.csv','accuracy_records.csv.gz','reconstruction_parity.csv.gz','case_sampling_means.csv','latency_conditions.csv','latency_grouped.csv','timing_repeats.csv.gz','stage_profiles.csv.gz','cache_reuse.csv']
    for seed in [0,1,2]:
        root=accepted/f'seed{seed}';path=root/'acceptance.json'
        source_kind='P0_seed_acceptance'
        if seed==0 and not path.exists():
            legacy=accepted.parent/'preflight/seed0_existing_accept'
            if (legacy/'acceptance.json').exists():
                root=legacy;path=root/'acceptance.json';source_kind='verified_seed0_legacy_fallback'
        if not path.exists():missing.append(f'training_seed{seed}: acceptance absent');continue
        receipt=json.loads(path.read_text())
        if receipt['training_seed']!=seed:raise ValueError('wrong training-seed receipt')
        if not receipt.get('data_files'):missing.append(f'training_seed{seed}: no accepted data-file index');continue
        for name,meta in receipt['data_files'].items():
            file=root/name
            assert file.stat().st_size==meta['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==meta['sha256'],file
        inputs[str(seed)]={'source_kind':source_kind,'acceptance_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data_files':receipt['data_files']}
        receipts[seed]=receipt
        if receipt['status']!='VERIFIED_COMPLETE':missing.append(f'training_seed{seed}: {receipt["physical_conditions"]}/1638 latency+memory conditions')
        for name in filenames:tables[name].extend(read_csv(root/name))
    registry=tables['checkpoint_registry.csv'];means=tables['case_sampling_means.csv'];conditions=tables['latency_conditions.csv'];raw= tables['timing_repeats.csv.gz'];parity=tables['reconstruction_parity.csv.gz']
    complete=len(receipts)==3 and all(r['status']=='VERIFIED_COMPLETE' for r in receipts.values())
    counts=dict(models=len(registry),heads=sum(bool(r.get('head_sha256')) for r in registry),training_seeds=len({int(r['training_seed']) for r in registry}),accuracy_requests=sum(r['accuracy_requests'] for r in receipts.values()),physical_conditions=len(conditions),formal_repeats=sum(r['kind']=='formal' for r in raw),warmups=sum(r['kind']=='warmup' for r in raw),memory_conditions=len(conditions),parity_pairs=len(parity),full_reference_requests=sum(r.get('full_reference_requests',0) for r in receipts.values()))
    if complete:assert counts==EXPECTED,(counts,EXPECTED)
    if not complete and not a.allow_partial:
        (out/'verification.json').write_text(json.dumps(dict(status='PARTIAL',complete=False,counts=counts,expected_counts=EXPECTED,missing=missing),indent=2)+'\n')
        print(json.dumps({'status':'PARTIAL','missing':missing}));raise SystemExit(2)
    def write(name,rows,header=None):
        if rows:csv_write(out/name,rows)
        elif header:
            with (out/name).open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerow(header)
    write('checkpoint_registry.csv',registry,['dataset','backbone','training_seed','head_seed','checkpoint_sha256','head_sha256','normalizer_sha256','split_sha256'])
    source_index=[]
    for seed,receipt in receipts.items():
        for path,meta in receipt.get('source_inputs',{}).items():
            source_index.append(dict(training_seed=seed,path=path,sha256=meta['sha256'],bytes=meta['bytes'],publication_scope='external source index only; raw long logs, weights and fields are not bundled'))
    write('source_input_index.csv',source_index,['training_seed','path','sha256','bytes','publication_scope'])
    sampling=[];absolute=[];ratios=[];training=[]
    keys=['backbone','training_seed','checkpoint_sha256','normalizer_sha256','implementation','reconstruction_id','method','beta','metric','budget_fraction']
    groups=collections.defaultdict(list)
    for r in means:groups[tuple(r[k] for k in keys)].append(r)
    full_lookup={}
    for key,rs in groups.items():
        if rs[0]['method']=='full':full_lookup[(rs[0]['backbone'],rs[0]['training_seed'],rs[0]['implementation'],rs[0]['metric'])]=float(rs[0]['case_mean'])
    for key,rs in groups.items():
        r=rs[0];values=[float(x['case_mean']) for x in rs]
        assert len(rs)==(1 if r['method']=='full' else 3)
        model_mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if len(values)>1 else None
        base=dict(dataset='airfoil2d',**dict(zip(keys,key)),case_count=110,population_id='published_cl_gt_0p02_110',main_accuracy_group=r['implementation']=='new')
        for x in rs:sampling.append(base|dict(sampling_seed=x['sampling_seed'],case_mean=float(x['case_mean']),model_mean=model_mean,sampling_sd=sd,sampling_seed_count=0 if r['method']=='full' else 3))
        absolute.append(base|dict(error=model_mean,full_error=full_lookup[(r['backbone'],r['training_seed'],r['implementation'],r['metric'])],sampling_sd=sd,sampling_seed_count=0 if r['method']=='full' else 3))
    lookup={(r['backbone'],r['training_seed'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'],r['method'],r['metric'],float(r['budget_fraction']),r['sampling_seed']):float(r['case_mean']) for r in means}
    config_groups=collections.defaultdict(list)
    for r in absolute:
        if r['method']!='full':config_groups[(r['backbone'],r['training_seed'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'],r['method'],r['metric'])].append(r)
    for key,rs in config_groups.items():
        r=rs[0];assert sorted(float(x['budget_fraction']) for x in rs)==[.375,.5,.75]
        values=[];denominators=[]
        for seed in ['0','1','2']:
            prefix=(r['backbone'],r['training_seed'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'])
            curve=[lookup[(*prefix,r['method'],r['metric'],b,seed)] for b in [.375,.5,.75]]
            uniform=[lookup[(*prefix,'uniform',r['metric'],b,seed)] for b in [.375,.5,.75]]
            denominator=float(np.trapezoid(uniform,[.375,.5,.75]));assert denominator>0
            denominators.append(denominator);values.append(float(np.trapezoid(curve,[.375,.5,.75]))/denominator)
        row={k:r[k] for k in ['dataset','backbone','training_seed','checkpoint_sha256','normalizer_sha256','implementation','reconstruction_id','method','beta','metric','case_count','population_id','main_accuracy_group']}
        row.update(budget_grid='0.375;0.5;0.75',mean=float(np.mean(values)),sampling_sd=float(np.std(values,ddof=1)),independent_training_count=1)
        for s in range(3):row[f'sampling_seed{s}']=values[s];row[f'matched_uniform_area_seed{s}']=denominators[s]
        ratios.append(row)
    quantity_rows=[]
    for r in absolute:quantity_rows.append(r|dict(quantity='absolute_error',value=r['error'],budget=r['budget_fraction']))
    for r in ratios:quantity_rows.append(r|dict(quantity='rAUEC',value=r['mean'],budget=r['budget_grid']))
    training_groups=collections.defaultdict(list)
    tkeys=['backbone','implementation','reconstruction_id','metric','method','beta','quantity','budget']
    for r in quantity_rows:training_groups[tuple(r[k] for k in tkeys)].append(r)
    for key,rs in training_groups.items():
        values={int(r['training_seed']):float(r['value']) for r in rs};assert len(values)==len(rs)
        row=dict(zip(tkeys,key));row.update(dataset='airfoil2d',seed0=values.get(0),seed1=values.get(1),seed2=values.get(2),mean=float(np.mean(list(values.values()))),sample_sd=float(np.std(list(values.values()),ddof=1)) if len(values)==3 else None,independent_training_count=len(values),expected_training_count=3,status='VERIFIED_COMPLETE' if len(values)==3 else 'PARTIAL',aggregation='case mean -> sampling aggregation per checkpoint -> mean and sampleSD over3 independently trained models')
        training.append(row)
    write('sampling_variability.csv',sampling,['backbone','training_seed','sampling_seed','method','metric','budget_fraction','case_mean','sampling_sd'])
    write('backbone_absolute_errors.csv',absolute,['backbone','training_seed','method','metric','budget_fraction','error','full_error'])
    write('backbone_rauec.csv',ratios,['backbone','training_seed','method','metric','mean','sampling_sd','reconstruction_id'])
    write('training_seed_summary.csv',training,['dataset','backbone','metric','method','quantity','budget','seed0','seed1','seed2','mean','sample_sd','independent_training_count','status'])
    write('accuracy_records.csv.gz',tables['accuracy_records.csv.gz'])
    write('reconstruction_parity.csv.gz',parity)
    write('latency_summary.csv',conditions,['training_seed','backbone','method','budget_fraction','scope','implementation','physical_gpu','direct_median_ms','direct_p95_ms'])
    grouped_latency=[]
    for r in tables['latency_grouped.csv']:
        matching=[x for x in conditions if all(x[k]==r[k] for k in ['training_seed','scope','backbone','method','budget_fraction','implementation'])]
        assert matching
        assert len({x['checkpoint_id'] for x in matching})==1 and len({x['physical_gpu'] for x in matching})==1
        grouped_latency.append(r|dict(checkpoint_sha256=matching[0]['checkpoint_id'],physical_gpu=matching[0]['physical_gpu'],cpu_affinity=matching[0]['cpu_affinity']))
    write('latency_grouped.csv',grouped_latency)
    write('backbone_latency_summary.csv',[r|dict(statistic_definition='pooled genuine formal wall-time observations within one checkpoint/training-seed/GPU only',formal_repeats_per_condition=100,warmups_per_condition=20) for r in grouped_latency])
    write('latency_repeats.csv.gz',raw)
    write('stage_profiles.csv.gz',tables['stage_profiles.csv.gz'])
    write('cache_reuse.csv',tables['cache_reuse.csv'])
    reconstruction_fields=['training_seed','backbone','checkpoint_id','method','beta','budget_fraction','sampling_seed','case_id','scope','implementation','reconstruction_id','physical_gpu','cpu_affinity','reconstruction_profile_median_ms','reconstruction_profile_p95_ms','direct_median_ms','direct_p95_ms','first_request_ms','cached_map_cpu_bytes','cached_map_gpu_bytes']
    write('reconstruction_summary.csv',[{k:r.get(k) for k in reconstruction_fields} for r in conditions],reconstruction_fields)
    memory_fields=['training_seed','backbone','checkpoint_id','method','budget_fraction','sampling_seed','case_id','scope','implementation','physical_gpu','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','first_request_peak_cuda_allocated_bytes','first_request_peak_cuda_reserved_bytes','cached_map_cpu_bytes','cached_map_gpu_bytes','allocator_scope','external_memory_scope']
    write('memory_summary.csv',[{k:r.get(k) for k in memory_fields} for r in conditions],memory_fields)
    # Never pool raw timings across different training seeds or physical GPUs.
    latency_blocks=collections.defaultdict(list)
    for r in conditions:latency_blocks[(r['training_seed'],r['physical_gpu'],r['backbone'],r['method'],r['budget_fraction'],r['scope'],r['implementation'])].append(r)
    latency_models=[]
    for key,rs in latency_blocks.items():
        seed,gpu,bb,method,budget,scope,impl=key
        latency_models.append(dict(training_seed=seed,physical_gpu=gpu,backbone=bb,checkpoint_sha256=rs[0]['checkpoint_id'],method=method,budget_fraction=budget,scope=scope,implementation=impl,condition_median_mean_ms=float(np.mean([float(r['direct_median_ms']) for r in rs])),condition_p95_mean_ms=float(np.mean([float(r['direct_p95_ms']) for r in rs])),conditions=len(rs),cpu_affinity=rs[0]['cpu_affinity'],aggregation='mean of per-case/sampling-condition formal medians within this checkpoint/GPU only'))
    write('latency_checkpoint_summary.csv',latency_models)
    joined=[]
    for t in latency_models:
        impl=t['implementation'] if t['implementation'] in ['old','new'] else 'new'
        e=next(r for r in absolute if r['backbone']==t['backbone'] and str(r['training_seed'])==str(t['training_seed']) and r['implementation']==impl and r['method']==t['method'] and float(r['budget_fraction'])==float(t['budget_fraction']) and r['metric']=='field_relative_l2')
        joined.append(t|dict(error=e['error'],sampling_sd=e['sampling_sd'],accuracy_cases=110,timing_cases=3,accuracy_reconstruction_id=e['reconstruction_id']))
    write('backbone_accuracy_latency.csv',joined)
    seed_components={}
    for seed in [0,1,2]:
        r=receipts.get(seed,{})
        seed_components[str(seed)]={'status':r.get('status','NOT_FOUND'),'existing_verified_conditions':r.get('seed_components',{}).get('existing_verified_conditions',0),'beta0_conditions_verified':r.get('seed_components',{}).get('beta0_conditions_verified',0),'expected_conditions':1638,'physical_conditions':r.get('physical_conditions',0),'physical_gpus':r.get('physical_gpus',[]),'accuracy_requests':r.get('accuracy_requests',0)}
    verification=dict(status='VERIFIED_COMPLETE' if complete else 'PARTIAL',complete=complete,counts=counts,expected_counts=EXPECTED,accepted_training_seeds=[s for s,r in receipts.items() if r['status']=='VERIFIED_COMPLETE'],accuracy_training_seeds=[s for s,r in receipts.items() if r.get('accuracy_complete')],missing=missing,seed_components=seed_components,source_acceptances=inputs,latency_pooling='never across training seed or physical GPU; compare checkpoint/GPU blocks explicitly',primary_reconstruction_id='grid_qhull_sparse_apply_v1p1',old_control_group='grid_scipy_linear_nearest_native_v1',executed_aggregator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (out/'latency_verification.json').write_text(json.dumps({k:verification[k] for k in ['status','complete','counts','expected_counts','seed_components','latency_pooling']},indent=2)+'\n')
    code=out/'code';code.mkdir(exist_ok=True)
    for name in ['aggregate_three_seeds.py','independent_accept.py','benchmark_adapter.py','gpu_monitor.py','validate_results.py','plot_final.py']:
        source=Path(__file__).with_name(name)
        if source.exists():shutil.copyfile(source,code/name)
    readme=f"""# Airfoil P0 results

Status: {verification['status']}. Accepted complete training seeds: {verification['accepted_training_seeds']}.

The required study has two point models, training seeds0/1/2, their own heads,110 historical positive-lift test cases, and budgets37.5%,50%,75%. Full100% has no sampling seed. Accuracy and timing include Uniform, PI, CPG beta0/beta1 and CA. Each seed's complete timing matrix has414 groups and1,638 physical conditions over scopesA/B/C, with20 warmups and100 formal repeats. Seed0 reuses1,314 verified point-model conditions and adds only324 beta0 conditions; legacy conv-temp Full-only observations are excluded from the primary three-training-seed totals.

Observed counts: {json.dumps(counts,sort_keys=True)}.

Accuracy aggregation first averages cases within a sampling seed, then sampling results within one checkpoint, then calculates mean and sampleSD across three independently trained models. Each rAUEC uses its own checkpoint/population/reconstruction matched Uniform denominator per sampling seed before aggregation. No training SD is filled when fewer than three models are available. Old and new reconstruction remain separate groups; the primary accuracy group is grid_qhull_sparse_apply_v1p1.

Latency preserves each checkpoint and physical GPU. ScopeA is CPU input to complete CPU float64 output; B has resident GPU model tensors and complete GPU float64 output with explicit CPU geometry work; C caches only the same geometry/ordered-point-set interpolation map. Per-condition timings and per-checkpoint/GPU pooled tables are retained; raw timings are never pooled across GPUs or training seeds. Between-checkpoint timing differences also include device and CPU-affinity differences and are not labeled pure training randomness.

The benchmark uses three fixed timing cases1001/1094/1198, three sampling seeds for adaptive methods, FP32 networks and complete float64 fields. Memory is measured in separate model/method processes. First-plus-steady cache costs are derived estimates, not contiguous request wall times. backbone_latency_summary.csv retains genuine pooled formal wall-time median/p95 within each checkpoint/training-seed/GPU. latency_summary.csv is the condition table. latency_repeats.csv.gz is a lossless gzip-compressed CSV containing the requested latency_repeats.csv scalar observations.

Pending: {json.dumps(missing)}.

All numeric tables contain observed results only. Verification requires6 model/head pairs,30,360 accuracy requests,4,914 timing/memory conditions and491,400 formal repetitions before final status. Complete CFD truth, weights, feature caches and long logs are not published.

CSV-only verification: python code/validate_results.py --data-dir . [--require-complete]
Figure reproduction: python code/plot_final.py --data-dir . --output-dir figures
"""
    (out/'README.md').write_text(readme)
    if (code/'validate_results.py').exists():
        command=[sys.executable,str(code/'validate_results.py'),'--data-dir',str(out),'--output-dir',str(out/'internal_validation')]
        if complete:command+=['--require-complete']
        subprocess.run(command,check=True)
    if not a.no_plots and (code/'plot_final.py').exists():subprocess.run([sys.executable,str(code/'plot_final.py'),'--data-dir',str(out),'--output-dir',str(out/'figures')],check=True)
    files=[]
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.name=='publication_contract.json' or '__pycache__' in path.parts:continue
        relative=str(path.relative_to(out));destination='figures' if relative.startswith('figures/') else 'reproducibility' if relative.startswith(('code/','config/')) else 'data'
        files.append(dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),allow_public=not relative.startswith('internal_validation/'),destination=destination))
    contract=dict(status=verification['status'],complete=complete,verification='verification.json',summary='README.md',counts=counts,expected_counts=EXPECTED,seed_components=seed_components,files=files,validator=dict(script='code/validate_results.py',base_args=['--data-dir','.'],final_args=['--require-complete']),artifact_root='.')
    temp=out/'publication_contract.json.tmp';temp.write_text(json.dumps(contract,indent=2)+'\n');temp.replace(out/'publication_contract.json')
    print(json.dumps({'status':verification['status'],'counts':counts,'missing':missing,'contract':str(out/'publication_contract.json')},indent=2))


if __name__=='__main__':main()
