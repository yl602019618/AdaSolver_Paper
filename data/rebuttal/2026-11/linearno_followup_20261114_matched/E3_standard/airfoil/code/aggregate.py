"""Airfoil statistics: per-seed error curves first, then matched Uniform area ratios."""
from common import *
import csv,collections,math


def csv_write(path,rows):
    if not rows:return
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


def aggregate_unlocked():
    groups=collections.defaultdict(list);sources=[]
    for arch in ['t3','linearno','linearno_conv']:
        complete=OUT/f'results/{arch}_accuracy_acceptance.json';preliminary=OUT/f'results/{arch}_preliminary_completion.json'
        if complete.exists() and read(complete).get('passed'):receipt=read(complete)
        elif preliminary.exists():receipt=read(preliminary)
        else:continue
        p=Path(receipt['metrics_csv'])
        if p.exists():
            assert sha(p)==receipt['metrics_sha256']
            sources.append(dict(path=str(p),sha256=sha(p)))
            with p.open() as f:
                for row in csv.DictReader(f):
                    key=(row['dataset'],row['backbone'],row['checkpoint_id'],row['metric_target'],row['test_population'],row['reconstruction_id'],row['metric'])
                    groups[key].append(row)
    budgets=[.375,.5,.75];budget_rows=[];area_rows=[];checks=[]
    for key,rows in groups.items():
        dataset,backbone,checkpoint,target,population,reconstruction,metric=key
        common=dict(dataset=dataset,backbone=backbone,checkpoint_id=checkpoint,training_seed=0,metric_target=target,
                    test_population=population,reconstruction_id=reconstruction,metric=metric)
        full=[r for r in rows if r['method']=='full'];assert len(full)==110 and all(not r['sampling_seed'] for r in full)
        full_values=[float(r['value']) for r in full if r['value']]
        full_mean=float(np.mean(full_values)) if len(full_values)==110 else None
        budget_rows.append(dict(**common,method='full',beta=None,budget_fraction=1.,cases=110,sampling_seeds=0,
                                mean_error=full_mean,sampling_sd=None,full_error=full_mean,error_over_full=1. if full_mean else None,
                                valid_records=len(full_values),status='complete' if len(full_values)==110 else 'nonfinite_metric_preserved'))
        methods=sorted({r['method'] for r in rows if r['method']!='full'})
        if not methods:continue
        curves={}
        for method in methods:
            curves[method]={}
            for seed in [0,1,2]:
                values=[]
                for budget in budgets:
                    current=[r for r in rows if r['method']==method and r['sampling_seed']==str(seed) and float(r['budget_fraction'])==budget]
                    assert len(current)==110 and len({r['case_id'] for r in current})==110
                    numeric=[float(r['value']) for r in current if r['value']]
                    values.append(float(np.mean(numeric)) if len(numeric)==110 else None)
                curves[method][seed]=values
            for j,budget in enumerate(budgets):
                v=[curves[method][s][j] for s in [0,1,2]]
                complete=all(x is not None and math.isfinite(x) for x in v)
                m=float(np.mean(v)) if complete else None
                budget_rows.append(dict(**common,method=method,beta=0. if method=='cpg_beta0' else None if method=='uniform' else 1.,
                    budget_fraction=budget,cases=110,sampling_seeds=3,mean_error=m,
                    sampling_sd=float(np.std(v,ddof=1)) if complete else None,full_error=full_mean,
                    error_over_full=m/full_mean if m is not None and full_mean else None,
                    valid_records=sum(110 for x in v if x is not None),status='complete' if complete else 'nonfinite_metric_preserved'))
        for method in methods:
            ratios=[]
            for seed in [0,1,2]:
                a=curves[method][seed];u=curves['uniform'][seed]
                if any(v is None for v in a+u):ratios.append(None);continue
                # Explicit trapezoidal weights for (.375,.5,.75), independently of root's aggregator.
                area=.0625*a[0]+.1875*a[1]+.125*a[2]
                denominator=.0625*u[0]+.1875*u[1]+.125*u[2]
                if denominator<=0:raise ValueError('Uniform area must be positive')
                ratios.append(area/denominator)
            complete=all(x is not None for x in ratios)
            area_rows.append(dict(**common,method=method,beta=0. if method=='cpg_beta0' else None if method=='uniform' else 1.,
                                 mean_rauec=float(np.mean(ratios)) if complete else None,
                                 sampling_sd=float(np.std(ratios,ddof=1)) if complete else None,
                                 seed0=ratios[0],seed1=ratios[1],seed2=ratios[2],sampling_seeds=3,cases=110,
                                 status='complete' if complete else 'nonfinite_metric_preserved'))
        checks.append(dict(**common,methods=methods,full_records=len(full),nonfull_per_method=990,area_definition='per-seed trapezoid divided by matched Uniform, then mean/ddof1 SD'))
    csv_write(OUT/'results/backbone_budget_metrics.csv',budget_rows)
    csv_write(OUT/'results/backbone_absolute_errors.csv',budget_rows)
    csv_write(OUT/'results/backbone_rauec.csv',area_rows)
    write(OUT/'results/statistical_checks.json',dict(groups=checks,source_files=sources,
         full_excluded_from_rauec=True,sampling_seeds_not_training_replicates=True))
    print('budget rows',len(budget_rows),'rAUEC rows',len(area_rows))


def main():
    import fcntl
    with (OUT/'runtime/aggregation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        aggregate_unlocked()


if __name__=='__main__':main()
