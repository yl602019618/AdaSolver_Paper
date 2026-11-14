"""Independent seed-first rAUEC aggregation for normalized long records.

The input adapter must provide explicit checkpoint, population and reconstruction
identities. This module never infers a cohort from a method's filename.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS=(.375,.5,.75)
GROUP=('dataset','backbone','checkpoint_id','normalizer_id','training_seed','metric_target',
       'test_population','reconstruction_id','metric','unit')


def aggregate(rows, expected_populations=None):
    cohorts=defaultdict(list)
    for row in rows:
        if any(key not in row for key in GROUP):
            raise ValueError('Missing explicit cohort identity')
        cohorts[tuple(str(row[k]) for k in GROUP)].append(row)
    output=[];curves=[];absolute=[]
    for cohort,records in cohorts.items():
        identity=dict(zip(GROUP,cohort))
        subsets=defaultdict(list)
        full_rows=[]
        for row in records:
            method=row['method'].lower()
            if method in ('full','fullmesh'):
                absolute.append({**identity,**row})
                full_rows.append(row)
                continue
            if int(row['sampling_seed']) not in (0,1,2) or float(row['budget_fraction']) not in BUDGETS:
                raise ValueError(('Unexpected seed or budget',cohort,row['sampling_seed'],row['budget_fraction']))
            beta=str(row.get('beta','')) if method!='uniform' else ''
            subsets[method,beta].append(row)
        full_ids=[str(r['case_id']) for r in full_rows]
        if not full_rows or len(set(full_ids))!=len(full_ids):
            raise ValueError(('Missing or duplicated Full reference',cohort))
        if any(float(r['budget_fraction'])!=1. or r.get('sampling_seed') not in ('',None,'None')
               or r.get('status','ok')!='ok' or not np.isfinite(float(r['value'])) for r in full_rows):
            raise ValueError(('Invalid Full reference identity/metric',cohort))
        if expected_populations is not None:
            expected=expected_populations[identity['dataset'],identity['test_population']]
            if set(full_ids)!=set(map(str,expected)):
                raise ValueError(('Full differs from predeclared population',cohort))
        if not subsets:continue  # Native grid-only Full reference has no rAUEC.
        uniform=subsets.get(('uniform',''))
        if not uniform:raise ValueError(('Missing matched Uniform',cohort))
        reference={}
        populations={}
        for seed in (0,1,2):
            means=[]
            for budget in BUDGETS:
                group=[r for r in uniform if int(r['sampling_seed'])==seed and float(r['budget_fraction'])==budget]
                ids=[str(r['case_id']) for r in group]
                if not group or len(set(ids))!=len(ids):raise ValueError(('Missing/duplicate Uniform case',cohort,seed,budget))
                if any(r.get('status','ok')!='ok' or not np.isfinite(float(r['value'])) for r in group):
                    raise ValueError(('Invalid Uniform reference; retain failures outside aggregation',cohort))
                populations[seed,budget]=set(ids)
                means.append(float(np.mean([float(r['value']) for r in group])))
            reference[seed]=float(np.trapezoid(means,BUDGETS))
            if reference[seed]<=0:raise ValueError(('Nonpositive Uniform area',cohort,seed))
        if len({tuple(sorted(x)) for x in populations.values()})!=1:
            raise ValueError(('Uniform population differs across seed/budget',cohort))
        if set(full_ids)!=populations[0,BUDGETS[0]]:
            raise ValueError(('Missing or duplicated matched Full reference',cohort))
        for (method,beta),method_rows in subsets.items():
            ratios=[]
            for seed in (0,1,2):
                means=[]
                for budget in BUDGETS:
                    group=[r for r in method_rows if int(r['sampling_seed'])==seed and float(r['budget_fraction'])==budget]
                    ids=[str(r['case_id']) for r in group]
                    if len(set(ids))!=len(ids) or set(ids)!=populations[seed,budget]:
                        raise ValueError(('Incomplete/duplicate method population',cohort,method,beta,seed,budget))
                    if any(r.get('status','ok')!='ok' or not np.isfinite(float(r['value'])) for r in group):
                        raise ValueError(('Invalid method metric; do not drop cases',cohort,method,beta))
                    mean=float(np.mean([float(r['value']) for r in group]));means.append(mean)
                    curves.append({**identity,'method':method,'beta':beta,'sampling_seed':seed,
                                   'budget_fraction':budget,'cases':len(ids),'mean_error':mean})
                ratios.append(float(np.trapezoid(means,BUDGETS))/reference[seed])
            output.append({**identity,'method':method,'beta':beta,'cases':len(populations[0,BUDGETS[0]]),
                           'sampling_seeds':3,'seed0_rauec':ratios[0],'seed1_rauec':ratios[1],
                           'seed2_rauec':ratios[2],'mean':float(np.mean(ratios)),
                           'sampling_sd':float(np.std(ratios,ddof=1))})
    return output,curves,absolute


def self_test():
    base=dict(dataset='analytic_fixture',backbone='fixed_model',checkpoint_id='fixed',normalizer_id='fixed',training_seed=0,
              metric_target='field',test_population='two_cases',reconstruction_id='same_rule',metric='field',unit='ratio',status='ok')
    rows=[]
    for seed,expected in enumerate([.5,.6,.7]):
        for bi,budget in enumerate(BUDGETS):
            for case,values in [('a',[1.,3.,5.]),('b',[3.,5.,9.])]:
                rows.append({**base,'method':'uniform','beta':'','sampling_seed':seed,'budget_fraction':budget,'case_id':case,'value':values[bi]})
                rows.append({**base,'method':'cpg','beta':1,'sampling_seed':seed,'budget_fraction':budget,'case_id':case,'value':(1.75/.375)*expected})
    for case in ['a','b']:
        rows.append({**base,'method':'full','sampling_seed':'','budget_fraction':1.,'case_id':case,'value':100000.})
    summary,_,_=aggregate(rows);method=next(r for r in summary if r['method']=='cpg')
    np.testing.assert_allclose([method['seed0_rauec'],method['seed1_rauec'],method['seed2_rauec'],method['mean'],method['sampling_sd']], [.5,.6,.7,.6,.1],atol=1e-14)
    assert not any(r['method']=='full' for r in summary)
    try:aggregate(rows[1:])
    except ValueError:pass
    else:raise AssertionError('Missing case was silently accepted')
    try:aggregate(rows,{('analytic_fixture','two_cases'):['a','b','missing_case']})
    except ValueError:pass
    else:raise AssertionError('Wrong locked population was silently accepted')
    print('Independent aggregation fixture passed: population means, seed-wise areas, sample SD and Full exclusion.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+');p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test();return
    if not args.input or not args.output:p.error('input and output required')
    records=[]
    for path in args.input:
        with path.open() as f:records.extend(csv.DictReader(f))
    summary,curves,_=aggregate(records);args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('independent_rauec.csv',summary),('independent_seed_curves.csv',curves)]:
        with (args.output/name).open('w',newline='') as f:
            if rows:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
