"""Independent scalar rAUEC recomputation; no aggregation implementation reuse."""
from __future__ import annotations
import argparse,csv,json,math,os,statistics
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='4'
from collections import defaultdict
import numpy as np
from common import OUT,read,write,digest

BUDGETS=(.375,.5,.75)
METHODS=('uniform','pi','cpg','ca')


def csv_read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def area(curve):
    return .125*(curve[0]+curve[1])/2+.25*(curve[1]+curve[2])/2


def close(a,b):
    difference=abs(float(a)-float(b))
    assert difference<=2e-12+2e-11*abs(float(b)),(a,b,difference)
    return difference


def verify():
    receipt=read(OUT/'results/aggregation_receipt.json');assert receipt['complete']
    ratios={};budget_means={};fulls={};cases=defaultdict(list);nrows=0
    for n in [85,141,211]:
        acceptance=read(OUT/f'results/r{n}_accuracy_acceptance.json');assert acceptance['passed']
        path=OUT/f'records/r{n}/metrics_records.csv';assert digest(path)==acceptance['raw_csv_sha256']
        for row in csv_read(path):
            grid=row['evaluation_grid'];method=row['method'];budget=float(row['budget_fraction'])
            seed=None if row['sampling_seed']=='' else int(row['sampling_seed']);nrows+=1
            cases[n,grid,method,budget,seed].append((int(row['case_id']),float(row['value'])))
    assert nrows==44400
    for key,values in cases.items():
        assert len(values)==200 and {v[0] for v in values}==set(range(1000,1200))
        average=math.fsum(v[1] for v in values)/200
        if key[2]=='full':
            assert key[3:]==(1.,None);fulls[key[:2]]=average
        else:budget_means[key]=average
    assert len(budget_means)==216 and len(fulls)==6
    for n in [85,141,211]:
        for grid in ['common','native']:
            for method in METHODS:
                for seed in [0,1,2]:
                    numerator=area([budget_means[n,grid,method,b,seed] for b in BUDGETS])
                    denominator=area([budget_means[n,grid,'uniform',b,seed] for b in BUDGETS])
                    ratios[n,grid,method,seed]=(numerator,denominator,numerator/denominator)
    differences=[]
    budget_rows=csv_read(OUT/'results/budget_metrics.csv');assert len(budget_rows)==216
    for row in budget_rows:
        key=(int(row['resolution']),row['evaluation_grid'],row['method'],float(row['budget_fraction']),int(row['sampling_seed']))
        differences.append(close(row['error_mean'],budget_means[key]))
        differences.append(close(row['full_error'],fulls[key[:2]]))
    area_rows=csv_read(OUT/'results/rauec_by_seed.csv');assert len(area_rows)==72
    for row in area_rows:
        key=(int(row['resolution']),row['evaluation_grid'],row['method'],int(row['sampling_seed']))
        for field,value in zip(['auec','matched_uniform_auec','rauec'],ratios[key]):differences.append(close(row[field],value))
        assert row['full_in_integral']=='False' and row['budget_count']=='3'
    summary=csv_read(OUT/'results/rauec_summary.csv');assert len(summary)==24
    for row in summary:
        key=(int(row['resolution']),row['evaluation_grid'],row['method'])
        values=[ratios[key+(s,)][2] for s in [0,1,2]]
        differences.append(close(row['rauec_mean'],statistics.mean(values)))
        differences.append(close(row['sampling_sd'],statistics.stdev(values)))
        assert row['sampling_seed_count']=='3' and row['training_seed_count']=='1'
        if row['method']=='uniform':assert values==[1.,1.,1.]
    absolute=csv_read(OUT/'results/absolute_errors.csv');assert len(absolute)==78
    for row in absolute:
        n=int(row['resolution']);grid=row['evaluation_grid'];method=row['method'];b=float(row['budget_fraction'])
        if method=='full':
            differences.append(close(row['error_mean'],fulls[n,grid]));assert row['sampling_sd']==''
        else:
            values=[budget_means[n,grid,method,b,s] for s in [0,1,2]]
            differences.append(close(row['error_mean'],statistics.mean(values)))
            differences.append(close(row['sampling_sd'],statistics.stdev(values)))
    result=dict(passed=True,complete=True,raw_metric_rows=nrows,verified_budget_means=216,verified_per_seed_area_ratios=72,
        verified_rauec_summary_rows=24,verified_absolute_error_rows=78,max_absolute_difference=max(differences),
        definition='scalar case means -> 3-point trapezoid -> own Uniform ratio per sampling seed -> mean/sample SD',
        raw_source='records/r{85,141,211}/metrics_records.csv',source_sha256=digest(__file__),
        aggregation_receipt_sha256=digest(OUT/'results/aggregation_receipt.json'))
    write(OUT/'results/statistics_acceptance.json',result);return result


def self_check():
    """A counterexample distinguishes mean-of-ratios from ratio-of-mean-curves."""
    import pandas as pd
    from aggregate import metric_tables
    u=[[1.,2.,5.],[4.,8.,6.],[2.,3.,9.]]
    modifiers={'uniform':[[1,1,1]]*3,'pi':[[.6,1.3,.9],[1.2,.5,1.1],[.7,.9,.4]],
        'cpg':[[.5,.9,.7],[1.4,.8,.9],[.8,1.2,.5]],'ca':[[.4,.9,.8],[.8,.6,1.],[1.1,.3,.6]]}
    rows=[]
    for grid in ['common','native']:
        for sid in range(1000,1200):
            base=dict(training_resolution=85,evaluation_grid=grid,case_id=sid,checkpoint_sha256='engineering-fixture',
                normalizer_sha256='engineering-fixture',protocol_sha256='engineering-fixture',reconstruction_id='engineering-fixture')
            rows.append(dict(**base,method='full',budget_fraction=1.,sampling_seed=None,value=.123))
            for method in METHODS:
                for seed in [0,1,2]:
                    for j,b in enumerate(BUDGETS):
                        rows.append(dict(**base,method=method,budget_fraction=b,sampling_seed=seed,value=u[seed][j]*modifiers[method][seed][j]))
    budget,absolute,ratios,summary=metric_tables(pd.DataFrame(rows))
    expected={m:[area([u[s][j]*modifiers[m][s][j] for j in range(3)])/area(u[s]) for s in range(3)] for m in METHODS}
    for row in summary:
        close(row['rauec_mean'],statistics.mean(expected[row['method']]))
        close(row['sampling_sd'],statistics.stdev(expected[row['method']]))
    pi_average_curve=[statistics.mean([u[s][j]*modifiers['pi'][s][j] for s in range(3)]) for j in range(3)]
    uniform_average_curve=[statistics.mean([u[s][j] for s in range(3)]) for j in range(3)]
    incorrect=area(pi_average_curve)/area(uniform_average_curve)
    assert abs(incorrect-statistics.mean(expected['pi']))>1e-3
    assert all(row['method']!='full' for row in ratios)
    result=dict(passed=True,engineering_fixture_only=True,seed_first_ratio_matches_scalar_calculation=True,
        distinguishes_ratio_of_average_curves=True,sample_sd_ddof1_verified=True,full_excluded=True,
        no_experiment_metric_written=True,source_sha256=digest(__file__))
    write(OUT/'evaluation/statistics_self_check.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true')
    print(json.dumps(self_check() if parser.parse_args().self_check else verify(),indent=2))
