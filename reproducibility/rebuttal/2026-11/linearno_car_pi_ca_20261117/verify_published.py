#!/usr/bin/env python3
"""Recompute public Car accuracy and latency tables using only scalar records."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with (gzip.open(path,'rt',newline='') if str(path).endswith('.gz') else Path(path).open(newline='')) as f:return list(csv.DictReader(f))


def area(y):return .0625*y[0]+.1875*y[1]+.125*y[2]


def percentile(values,q):
    x=sorted(values);r=(len(x)-1)*q;lo=math.floor(r);hi=math.ceil(r)
    return x[lo]+(x[hi]-x[lo])*(r-lo)


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-10),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();root=a.data_dir
    protocol=json.loads((root/'protocol.json').read_text());assert protocol['training_seeds']==[0] and protocol['sampling_seeds']==[0,1,2]
    split=json.loads((root/'splits.json').read_text());ids=split['test'];assert len(ids)==len(set(ids))==100
    cfg=json.loads((root/'frozen_policies.json').read_text());records=read_csv(root/'per_case_records.csv.gz')
    assert len(records)==8200
    by_variant={name:[] for name in cfg}
    registry={r['variant']:r for r in read_csv(root/'run_registry.csv')}
    for r in records:
        name=r['variant'];by_variant[name].append(r);c=cfg[name];reg=registry[name]
        assert r['status']==r['cd_status']=='ok' and int(r['training_seed'])==0
        assert r['method']==c['method'] and r['policy_target']==c['selection_target']
        assert r['operator_sha256']==reg['checkpoint_sha256'] and r['normalizer_sha256']==reg['normalizer_sha256']
        assert r['head_sha256']==reg['head_sha256']
        assert all(math.isfinite(float(r[k])) for k in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error','cd_pred','cd_gt'])
        close(r['field_error'],float(r['velocity_rel_l2'])+.5*float(r['pressure_rel_l2']))
        close(r['cd_relative_error'],abs(float(r['cd_pred'])-float(r['cd_gt']))/max(abs(float(r['cd_gt'])),1e-12))
        assert r['reconstruction_id']=='original_physical_knn4_idw_surface_pressure'
        if r['method'] in ['ca','cpg']:assert int(r['coarse_count'])==math.floor(.1*int(r['full_point_count']))
    for name,rows in by_variant.items():
        keys=[(r['sample_id'],int(r['sampling_seed']),float(r['budget'])) for r in rows]
        expected=set(itertools.product(ids,[0] if name=='full' else [0,1,2],[1.] if name=='full' else [.375,.5,.75]))
        assert len(keys)==len(set(keys)) and set(keys)==expected,(name,len(keys))
    summaries=read_csv(root/'accuracy_summary.csv');curves=read_csv(root/'per_sampling_seed_budget.csv');denoms=read_csv(root/'uniform_denominators.csv')
    assert len(summaries)==10 and len(curves)==90 and len(denoms)==6
    for summary in summaries:
        target=summary['target'];metric=summary['metric'];method=summary['method'];assert summary['training_sd']=='' and int(summary['training_runs'])==1
        name='uniform' if method=='uniform' else ('cpg_'+target+'_beta0' if method=='cpg_beta0' else method+'_'+target)
        scores=[]
        for seed in [0,1,2]:
            e=[statistics.mean(float(r[metric]) for r in by_variant[name] if int(r['sampling_seed'])==seed and float(r['budget'])==b) for b in [.375,.5,.75]]
            u=[statistics.mean(float(r[metric]) for r in by_variant['uniform'] if int(r['sampling_seed'])==seed and float(r['budget'])==b) for b in [.375,.5,.75]]
            numerator=area(e);den=area(u);ratio=numerator/den;scores.append(ratio);close(summary[f'sampling_seed_{seed}'],ratio)
            d=next(r for r in denoms if r['target']==target and int(r['sampling_seed'])==seed);close(d['uniform_area'],den)
            rr=[r for r in curves if r['target']==target and r['method']==method and int(r['sampling_seed'])==seed];assert len(rr)==3
            for r in rr:
                close(r['case_mean_error'],e[[.375,.5,.75].index(float(r['budget']))]);close(r['method_area'],numerator);close(r['uniform_area'],den);close(r['rAUEC'],ratio)
        close(summary['mean'],statistics.mean(scores));close(summary['sampling_sd'],statistics.stdev(scores))
    primary=read_csv(root/'primary_metric_records.csv.gz');assert len(primary)==9200
    lookup={(r['variant'],r['sample_id'],int(r['sampling_seed']),float(r['budget'])):r for r in records}
    for r in primary:
        seed=0 if r['sampling_seed']=='' else int(r['sampling_seed']);raw=lookup[r['variant'],r['sample_id'],seed,float(r['budget'])]
        assert r['variant'] in ['full','uniform'] or raw['policy_target']==r['target']
        close(r['error'],raw[r['metric']])
    timing=read_csv(root/'inference_repeats.csv.gz');conditions=read_csv(root/'inference_conditions.csv');assert len(timing)==26640 and len(conditions)==222
    groups={}
    for r in timing:
        key=(r['target'],r['variant'],r['case_id'],r['sampling_seed'],r['budget']);phase=r['phase'];n=int(r['repeat']);v=float(r['wall_ms'])
        assert math.isfinite(v) and v>0
        g=groups.setdefault(key,{'warmup':{},'formal':{}});assert n not in g[phase];g[phase][n]=v
    assert len(groups)==222
    for r in conditions:
        key=(r['target'],r['variant'],r['case_id'],r['sampling_seed'],r['budget']);g=groups[key]
        assert set(g['warmup'])==set(range(20)) and set(g['formal'])==set(range(100))
        v=list(g['formal'].values());close(r['median_ms'],statistics.median(v));close(r['p95_ms'],percentile(v,.95));close(r['mean_ms'],statistics.mean(v));close(r['repeat_sd_ms'],statistics.stdev(v))
    budgets=read_csv(root/'inference_budget_summary.csv')
    for r in budgets:
        rr=[x for x in conditions if x['target']==r['target'] and x['variant']==r['variant'] and float(x['budget'])==float(r['budget'])]
        assert len(rr)==int(r['conditions']);values=[float(x['median_ms']) for x in rr]
        close(r['mean_condition_median_ms'],statistics.mean(values));close(r['condition_median_sd_ms'],statistics.stdev(values))
        full=[float(x['median_ms']) for x in conditions if x['target']==r['target'] and x['variant']=='full'];close(r['same_gpu_full_mean_median_ms'],statistics.mean(full));close(r['full_over_method_latency_ratio'],statistics.mean(full)/statistics.mean(values))
    head=json.loads((root/'head_inference_manifest.json').read_text());assert hashlib.sha256((root/head['checkpoint']).read_bytes()).hexdigest()==head['head_sha256']
    assert head['epochs_completed']==200 and head['input_dim']==256 and head['width']==64 and head['training_seed']==0
    history=read_csv(root/'head_training_history.csv');assert len(history)==200
    best=min(history,key=lambda r:float(r['validation_mse']));assert int(best['epoch'])==head['best_epoch'];close(best['validation_mse'],head['best_validation_mse'])
    result={'passed':True,'native_records':8200,'primary_metric_views':9200,'training_runs':1,'sampling_seeds':3,
            'accuracy_statistics_recomputed':True,'head_weights_hash_and_selection_verified':True,'formal_conditions':222,
            'formal_repetitions':22200,'warmups':4440,'latency_statistics_recomputed':True,'field_cd_policy_identity_preserved':True}
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
