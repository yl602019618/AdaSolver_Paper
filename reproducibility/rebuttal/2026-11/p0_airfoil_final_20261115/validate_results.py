#!/usr/bin/env python3
"""Independent CSV-only P0 validation; no model, GPU or figure-path dependency."""
import argparse,collections,csv,gzip,json,math
from pathlib import Path
import numpy as np

EXPECTED=dict(models=6,heads=6,training_seeds=3,accuracy_requests=30360,physical_conditions=4914,formal_repeats=491400,warmups=98280,memory_conditions=4914,parity_pairs=29700,full_reference_requests=660)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path);p.add_argument('--require-complete',action='store_true');a=p.parse_args();base=a.data_dir;output=a.output_dir or base;output.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=base/name
        if not path.exists():return []
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',newline='') as f:return list(csv.DictReader(f))
    def same(x,y):assert np.isclose(float(x),float(y),rtol=1e-10,atol=1e-12),(x,y)
    verification=json.loads((base/'verification.json').read_text())
    registry=read('checkpoint_registry.csv');accuracy=read('accuracy_records.csv.gz');sampling=read('sampling_variability.csv');absolute=read('backbone_absolute_errors.csv');rauec=read('backbone_rauec.csv');training=read('training_seed_summary.csv');conditions=read('latency_summary.csv');repeats=read('latency_repeats.csv.gz');memory=read('memory_summary.csv');parity=read('reconstruction_parity.csv.gz')
    assert len({(r['backbone'],r['training_seed']) for r in registry})==len(registry)
    if registry:
        assert len({r['checkpoint_sha256'] for r in registry})==len(registry)
        assert len({r['head_sha256'] for r in registry})==len(registry)
        assert len({r['normalizer_sha256'] for r in registry})==1
        assert len({r['split_sha256'] for r in registry})==1
    for r in registry:
        assert r['training_seed']==r['head_seed'] and int(r['epochs'])==1000 and int(r['updates'])==113000
        assert r['checkpoint_sha256'] and r['head_sha256'] and r['verified']=='True'
    registry_lookup={(r['backbone'],r['training_seed']):r for r in registry}
    requests={(r['backbone'],r['training_seed'],r['case_id'],r['method'],r['sampling_seed'],r['budget_fraction']) for r in accuracy}
    case_groups=collections.defaultdict(list)
    keys=['backbone','training_seed','checkpoint_sha256','implementation','reconstruction_id','method','metric','budget_fraction','sampling_seed']
    for r in accuracy:
        model=registry_lookup[(r['backbone'],r['training_seed'])]
        assert r['checkpoint_sha256']==model['checkpoint_sha256'] and r['normalizer_sha256']==model['normalizer_sha256']
        if r['method']=='ca':assert r['head_sha256']==model['head_sha256']
        value=float(r['error']);assert math.isfinite(value) and value>=0
        if r['method']=='full':assert r['sampling_seed']==''
        case_groups[tuple(r[k] for k in keys)].append(r)
    for group in case_groups.values():assert len(group)==110 and len({r['case_id'] for r in group})==110
    case_means={k:float(np.mean([float(r['error']) for r in rs])) for k,rs in case_groups.items()}
    sampling_keys=[tuple(r[k] for k in keys) for r in sampling]
    assert len(sampling_keys)==len(set(sampling_keys)) and set(sampling_keys)==set(case_groups),'sampling table must cover every raw case/sampling group'
    absolute_keys=[tuple(r[k] for k in keys[:-1]) for r in absolute]
    assert len(absolute_keys)==len(set(absolute_keys)) and set(absolute_keys)=={k[:-1] for k in case_groups},'absolute table coverage differs from raw case groups'
    ratio_key_names=keys[:7]
    ratio_keys=[tuple(r[k] for k in ratio_key_names) for r in rauec]
    expected_ratio_keys={k[:7] for k in case_groups if k[5]!='full'}
    assert len(ratio_keys)==len(set(ratio_keys)) and set(ratio_keys)==expected_ratio_keys,'rAUEC table must cover every non-Full checkpoint/method/metric group'
    for r in sampling:same(r['case_mean'],case_means[tuple(r[k] for k in keys)])
    for r in absolute:
        prefix=tuple(r[k] for k in keys[:-1]);ss=[''] if r['method']=='full' else ['0','1','2'];values=[case_means[(*prefix,s)] for s in ss]
        same(r['error'],np.mean(values))
        if len(ss)==1:assert r['sampling_sd']==''
        else:same(r['sampling_sd'],np.std(values,ddof=1))
    for r in rauec:
        assert r['method']!='full';values=[]
        for s in ['0','1','2']:
            prefix=(r['backbone'],r['training_seed'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'])
            y=[case_means[(*prefix,r['method'],r['metric'],str(b),s)] for b in [.375,.5,.75]]
            u=[case_means[(*prefix,'uniform',r['metric'],str(b),s)] for b in [.375,.5,.75]]
            den=float(np.trapezoid(u,[.375,.5,.75]));ratio=float(np.trapezoid(y,[.375,.5,.75]))/den
            same(r['matched_uniform_area_seed'+s],den);same(r['sampling_seed'+s],ratio);values.append(ratio)
        same(r['mean'],np.mean(values));same(r['sampling_sd'],np.std(values,ddof=1))
    training_key_names=['backbone','implementation','reconstruction_id','metric','method','beta','quantity','budget']
    training_keys=[tuple(r[k] for k in training_key_names) for r in training]
    expected_training_keys=set()
    for r in absolute:expected_training_keys.add(tuple(r[k] for k in training_key_names[:6])+('absolute_error',r['budget_fraction']))
    for r in rauec:expected_training_keys.add(tuple(r[k] for k in training_key_names[:6])+('rAUEC',r['budget_grid']))
    assert len(training_keys)==len(set(training_keys)) and set(training_keys)==expected_training_keys,'training summary must cover all available per-checkpoint quantities'
    for r in training:
        source=absolute if r['quantity']=='absolute_error' else rauec;values={}
        for x in source:
            if all(x[k]==r[k] for k in ['backbone','implementation','reconstruction_id','metric','method','beta']) and (r['quantity']=='rAUEC' or float(x['budget_fraction'])==float(r['budget'])):
                values[int(x['training_seed'])]=float(x['error'] if r['quantity']=='absolute_error' else x['mean'])
        assert len(values)==int(r['independent_training_count'])
        for s in [0,1,2]:
            if s in values:same(r[f'seed{s}'],values[s])
            else:assert r[f'seed{s}']==''
        same(r['mean'],np.mean(list(values.values())))
        if len(values)==3:same(r['sample_sd'],np.std(list(values.values()),ddof=1));assert r['status']=='VERIFIED_COMPLETE'
        else:assert r['sample_sd']=='' and r['status']=='PARTIAL'
    by_condition=collections.defaultdict(list)
    for r in repeats:by_condition[(r['training_seed'],r['group_id'],r['condition'])].append(r)
    seen=set();gpus=collections.defaultdict(set);pooled=collections.defaultdict(list)
    for r in conditions:
        model=registry_lookup[(r['backbone'],r['training_seed'])]
        assert r['checkpoint_id']==model['checkpoint_sha256']
        if r['method']=='ca':assert r['head_checkpoint_sha256']==model['head_sha256']
        key=(r['training_seed'],r['group_id'],r['condition']);assert key not in seen;seen.add(key)
        rows=by_condition[key];formal=[x for x in rows if x['kind']=='formal'];warm=[x for x in rows if x['kind']=='warmup']
        assert len(formal)==100 and len(warm)==20
        assert {int(x['repeat_id']) for x in formal}==set(range(100)) and {int(x['repeat_id']) for x in warm}==set(range(20))
        times=[float(x['total_ms']) for x in formal];assert np.isfinite(times).all() and min(times)>0
        same(r['direct_median_ms'],np.median(times));same(r['direct_p95_ms'],np.percentile(times,95));same(r['timing_repeat_sd_ms'],np.std(times,ddof=1))
        pooled[(r['training_seed'],r['scope'],r['backbone'],r['method'],r['budget_fraction'],r['implementation'],r['checkpoint_id'],r['physical_gpu'])].extend(times)
        if r['method']=='full':assert r['sampling_seed']==''
        gpus[r['training_seed']].add(r['physical_gpu'])
    assert all(len(v)==1 for v in gpus.values())
    latency_backbone=read('backbone_latency_summary.csv')
    pooled_keys=[(r['training_seed'],r['scope'],r['backbone'],r['method'],r['budget_fraction'],r['implementation'],r['checkpoint_sha256'],r['physical_gpu']) for r in latency_backbone]
    assert len(pooled_keys)==len(set(pooled_keys)) and set(pooled_keys)==set(pooled),'backbone latency summary must cover every checkpoint/GPU group'
    for key,r in zip(pooled_keys,latency_backbone):
        same(r['pooled_median_ms'],np.median(pooled[key]));same(r['pooled_p95_ms'],np.percentile(pooled[key],95));assert int(r['formal_observations'])==len(pooled[key])
    assert len(memory)==len(conditions)
    for r,m in zip(conditions,memory):
        for key in ['training_seed','backbone','checkpoint_id','method','case_id','scope','implementation','physical_gpu','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes']:assert r[key]==m[key]
    for r in parity:
        assert r['passed']=='True' and int(r['elementwise_violations'])==0 and float(r['atol'])==1e-6 and float(r['rtol'])==1e-5
        assert math.isfinite(float(r['max_absolute_difference'])) and float(r['max_absolute_difference'])>=0
    actual=dict(models=len(registry),heads=sum(bool(r['head_sha256']) for r in registry),training_seeds=len({r['training_seed'] for r in registry}),accuracy_requests=len(requests),physical_conditions=len(conditions),formal_repeats=sum(r['kind']=='formal' for r in repeats),warmups=sum(r['kind']=='warmup' for r in repeats),memory_conditions=len(memory),parity_pairs=len(parity),full_reference_requests=sum(k[3]=='full' for k in requests))
    assert actual==verification['counts'],(actual,verification['counts'])
    if a.require_complete:
        assert verification['status']=='VERIFIED_COMPLETE' and verification['complete']
        assert actual==EXPECTED
        assert len(accuracy)==242880 and len(sampling)==2208 and len(absolute)==768 and len(rauec)==240 and len(training)==336
        assert len(latency_backbone)==558
        assert sum(r['implementation']=='new' for r in training)==168
        assert set(verification['accepted_training_seeds'])=={0,1,2}
        assert verification['seed_components']['0']['existing_verified_conditions']==1314
        assert verification['seed_components']['0']['beta0_conditions_verified']==324
        assert all(int(r['independent_training_count'])==3 for r in training)
        expected_keys={(str(s),bb,str(case),method,sampling,str(budget),impl,scope)
            for s in [0,1,2] for bb in ['Transolver3','LinearNO_point_temp'] for case in [1001,1094,1198]
            for method in ['full','uniform','pi','cpg_beta0','cpg','ca']
            for sampling in ([''] if method=='full' else ['0','1','2'])
            for budget in ([1.] if method=='full' else [.375,.5,.75])
            for impl in (['full_shared'] if method=='full' else ['old','new'])
            for scope in ['A_cpu_cpu','B_gpu_gpu','C_cpu_cpu_cached_map']}
        actual_keys={(r['training_seed'],r['backbone'],r['case_id'],r['method'],r['sampling_seed'],r['budget_fraction'],r['implementation'],r['scope']) for r in conditions}
        assert actual_keys==expected_keys
    result=dict(status='passed',complete=verification['complete'],counts=actual,table_coverage=dict(raw_metric_rows=len(accuracy),sampling_rows=len(sampling),absolute_rows=len(absolute),rauec_rows=len(rauec),training_summary_rows=len(training)),checks=['schema and model/head identities','exact summary-key coverage of raw groups','raw case means and sampling SD','matched-seed rAUEC','three-training-seed mean and sample SD','same-checkpoint parity receipt rows','direct100-repeat timing medians/p95 and20warmups','memory joins','one physical GPU per training-seed timing block'],raw_gpu_pooling='none')
    (output/'numeric_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
