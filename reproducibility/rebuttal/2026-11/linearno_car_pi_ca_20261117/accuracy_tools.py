"""Exact run identity, coverage and paired-Uniform statistics for this extension."""
from pathlib import Path
import json
import itertools
import math
import statistics
from common import ROOT,E3,activate_sources,sha256
activate_sources()
from adasolver.inference.shapenet_car_contract import method_config

BUDGETS=[.375,.5,.75]
SEEDS=[0,1,2]


def label(config):
    method=config['method']
    if method=='fullmesh':return 'full'
    if method=='uniform':return 'uniform'
    suffix='_beta0' if method=='cpg' and config['beta']==0 else ''
    return method+'_'+config['selection_target']+suffix


def load_runs(require_complete=True):
    runs={}
    for part in ['control_full','controls','adaptive']:
        for p in sorted((ROOT/'records/seed0'/part).glob('*/run_manifest.json')):
            meta=json.loads(p.read_text());name=label(meta['config'])
            if name in runs:raise ValueError('duplicate run label '+name)
            if require_complete and meta['status']!='complete':raise ValueError('incomplete run '+str(p))
            raw=[json.loads(l) for l in (p.parent/'raw.jsonl').open() if l.strip()]
            runs[name]={'manifest':meta,'rows':raw,'path':p.parent}
    if require_complete:
        wanted={'full','uniform','pi_field','pi_cd','cpg_field','cpg_field_beta0','cpg_cd','cpg_cd_beta0','ca_field','ca_cd'}
        if set(runs)!=wanted:raise ValueError(f'missing runs: {wanted-set(runs)}')
    return runs


def expected_configs():
    configs=json.loads((ROOT/'controls.json').read_text())+json.loads((ROOT/'seed0_missing_methods.json').read_text())+[{'method':'fullmesh','selection_target':'field'}]
    result={}
    for config in configs:
        resolved=method_config(config['method'],**{k:v for k,v in config.items() if k!='method'})
        result[label(resolved)]=resolved
    return result


def validate_runs(runs):
    split=json.loads((ROOT/'splits/seed0.json').read_text());ids=split['test']
    operator=json.loads((E3/'checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json').read_text())
    head=json.loads((ROOT/'checkpoints/seed0/head/ca_pressure_linearno_trainingseed0_headseed0_manifest.json').read_text())
    configs=expected_configs();findings=[];details={}
    for name,run in runs.items():
        meta,rows=run['manifest'],run['rows']
        expected=set(itertools.product(ids,[0] if name=='full' else SEEDS,[1.] if name=='full' else BUDGETS))
        keys=[(r['sample_id'],r['seed'],r['budget_fraction']) for r in rows]
        errors=[]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:errors.append('coverage')
        if meta['config']!=configs[name]:errors.append('frozen_configuration')
        if meta['split_hash']!=split['split_hash']:errors.append('split_hash')
        if meta['operator_sha256']!=operator['checkpoint_sha256'] or meta['normalizer_sha256']!=operator['normalizer_sha256']:errors.append('operator_normalizer')
        if meta['head_sha256']!=(head['head_sha256'] if name.startswith('ca_') else None):errors.append('head_binding')
        bad=[]
        for r in rows:
            if r['status']!='ok' or r['cd_status']!='ok' or any(not math.isfinite(r[k]) for k in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error','cd_pred','cd_gt']):bad.append((r['sample_id'],r['seed'],r['budget_fraction']))
            if not math.isclose(r['field_error'],r['velocity_rel_l2']+.5*r['pressure_rel_l2'],abs_tol=1e-12,rel_tol=1e-12):errors.append('field_metric_identity')
            if not math.isclose(r['cd_relative_error'],abs(r['cd_pred']-r['cd_gt'])/max(abs(r['cd_gt']),1e-12),abs_tol=1e-12,rel_tol=1e-12):errors.append('cd_metric_identity')
            if name.startswith(('ca_','cpg_')) and r['coarse_count']!=math.floor(.1*r['full_point_count']):errors.append('coarse_count')
            if r['reconstruction_k']!=4:errors.append('reconstruction')
        if bad:errors.append('nonfinite_or_failed_records')
        details[name]={'records':len(rows),'expected_records':len(expected),'cases':len({r['sample_id'] for r in rows}),'errors':sorted(set(errors)),'invalid_records':bad}
        if errors:findings.append(name)
    paired=[]
    for target in ['field','cd']:
        a={ (r['sample_id'],r['seed'],r['budget_fraction']):r for r in runs['cpg_'+target]['rows'] }
        for r in runs['cpg_'+target+'_beta0']['rows']:
            ref=a[r['sample_id'],r['seed'],r['budget_fraction']]
            if (r['selected_indices_sha256'],r['coarse_indices_sha256'])!=(ref['selected_indices_sha256'],ref['coarse_indices_sha256']):paired.append((target,r['sample_id'],r['seed'],r['budget_fraction']))
    if paired:findings.append('cpg_beta_point_pairing')
    return {'passed':not findings,'run_count':len(runs),'total_native_records':sum(len(x['rows']) for x in runs.values()),'details':details,'findings':findings,'cpg_pairing_failures':paired,'test_cases':len(ids),'training_seeds':[0],'sampling_seeds':SEEDS}


def area(curve):return sum((BUDGETS[i+1]-BUDGETS[i])*(curve[i]+curve[i+1])/2 for i in range(2))


def aggregate(runs):
    results=[];curves=[];denominators=[]
    for target,metric in [('field','field_error'),('cd','cd_relative_error')]:
        uniform={seed:[statistics.mean(r[metric] for r in runs['uniform']['rows'] if r['seed']==seed and r['budget_fraction']==budget) for budget in BUDGETS] for seed in SEEDS}
        for seed in SEEDS:denominators.append({'target':target,'metric':metric,'sampling_seed':seed,'uniform_area':area(uniform[seed]),**{f'error_{b}':v for b,v in zip(BUDGETS,uniform[seed])}})
        for method in ['uniform','pi','cpg','ca','cpg_beta0']:
            name='uniform' if method=='uniform' else ('cpg_'+target+'_beta0' if method=='cpg_beta0' else method+'_'+target)
            scores=[]
            for seed in SEEDS:
                curve=[statistics.mean(r[metric] for r in runs[name]['rows'] if r['seed']==seed and r['budget_fraction']==budget) for budget in BUDGETS]
                den=area(uniform[seed]);a=area(curve);score=a/den;scores.append(score)
                for b,value in zip(BUDGETS,curve):curves.append({'backbone':'LinearNO','training_seed':0,'target':target,'metric':metric,'method':method,'sampling_seed':seed,'budget':b,'case_mean_error':value,'method_area':a,'uniform_area':den,'rAUEC':score})
            results.append({'backbone':'LinearNO','training_seed':0,'target':target,'metric':metric,'method':method,**{f'sampling_seed_{s}':v for s,v in zip(SEEDS,scores)},'mean':statistics.mean(scores),'sampling_sd':statistics.stdev(scores),'training_sd':'','training_runs':1})
    return results,curves,denominators
