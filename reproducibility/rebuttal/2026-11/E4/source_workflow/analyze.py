"""Recompute and visualize the completed E4 Darcy resolution evidence.

No model inference or training. All observations come from the original frozen
experiment. New rAUEC values are explicitly a derived analysis of those records.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CURRENT = OUT.parent
SOURCE = CURRENT / 'rebuttal_20261112/experiments/E4_darcy_resolution'
RESOLUTIONS = [85, 141, 211]
BUDGETS = [.375, .5, .75]
METHODS = ['uniform', 'pi', 'cpg', 'ca']


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(name, records):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0])); writer.writeheader(); writer.writerows(records)


def main():
    raw_path = SOURCE / 'records/metrics_records.csv'
    raw = read_csv(raw_path)
    records = [r for r in raw if r['metric'] == 'field_relative_l2_common']
    assert len(raw) == 54000 and len(records) == 27000
    assert {r['training_seed'] for r in records} == {'0'}
    assert len({r['checkpoint_sha256'] for r in records}) == 1
    groups = defaultdict(list)
    lookup = {}
    for r in records:
        key = (int(r['observation_resolution']), r['method'], int(r['retained_points']), r['sampling_seed'])
        value = float(r['value']); assert np.isfinite(value)
        groups[key].append(r)
        item = key + (int(r['case_id']),)
        assert item not in lookup
        lookup[item] = value
    assert len(groups) == 135
    assert all(len(v) == 200 and {int(r['case_id']) for r in v} == set(range(1000,1200)) for v in groups.values())
    means = {k: float(np.mean([float(r['value']) for r in v])) for k,v in groups.items()}
    summaries = read_csv(SOURCE / 'results/resolution_budget.csv')
    fixed = read_csv(SOURCE / 'results/fixed_points.csv')
    max_delta = 0.
    for row in summaries + fixed:
        n, method, count = int(row['observation_resolution']), row['method'], int(row['retained_points'])
        seeds = [''] if method == 'full' else ['0','1','2']
        vals = [means[n,method,count,seed] for seed in seeds]
        delta = abs(float(np.mean(vals)) - float(row['mean_relative_l2_common']))
        assert delta < 1e-13
        max_delta = max(max_delta, delta)
        if method != 'full':
            np.testing.assert_allclose(np.std(vals,ddof=1),float(row['sampling_seed_std']),rtol=1e-10,atol=1e-14)

    def row_for(n, method, condition, source=summaries):
        return next(r for r in source if int(r['observation_resolution'])==n and r['method']==method and r['condition']==condition)

    def case_means(n, method, count):
        seeds = [''] if method=='full' else ['0','1','2']
        return np.asarray([np.mean([lookup[n,method,count,seed,case] for seed in seeds]) for case in range(1000,1200)])

    comparisons = []; rauc = []; coarse_rows = []
    for n in RESOLUTIONS:
        full = float(row_for(n,'full','full')['mean_relative_l2_common'])
        for condition, source in [('fraction_0.5',summaries),('fixed_5419',fixed),('fraction_0.75',summaries)]:
            uniform = row_for(n,'uniform',condition,source)
            ue = float(uniform['mean_relative_l2_common'])
            k = int(uniform['retained_points'])
            for method in METHODS:
                row = row_for(n,method,condition,source)
                e = float(row['mean_relative_l2_common'])
                wins = int(np.sum(case_means(n,method,k)<case_means(n,'uniform',k))) if method!='uniform' else None
                comparisons.append(dict(observation_resolution=n,condition=condition,method=method,
                    retained_points=k,actual_budget_fraction=k/(n*n),full_error=full,mean_error=e,
                    sampling_sd=float(row['sampling_seed_std']),matched_uniform_error=ue,
                    error_reduction_percent=100*(1-e/ue),cases_better_than_uniform=wins,
                    test_cases=200,training_seeds=1,sampling_seeds=3))
        for method in METHODS:
            ratios=[]
            for seed in ['0','1','2']:
                curve=[means[n,method,int(row_for(n,method,f'fraction_{b}')['retained_points']),seed] for b in BUDGETS]
                base=[means[n,'uniform',int(row_for(n,'uniform',f'fraction_{b}')['retained_points']),seed] for b in BUDGETS]
                ratios.append(float(np.trapezoid(curve,BUDGETS)/np.trapezoid(base,BUDGETS)))
            rauc.append(dict(observation_resolution=n,method=method,seed0_rauec=ratios[0],seed1_rauec=ratios[1],seed2_rauec=ratios[2],
                mean_rauec=float(np.mean(ratios)),sampling_sd=float(np.std(ratios,ddof=1)),training_seeds=1,
                integration='nominal budgets0.375/0.5/0.75; case means then seedwise matched Uniform area ratios',
                origin='derived from existing E4 records; no new experiment'))
        coarse = int(row_for(n,'cpg','fixed_5419',fixed)['coarse_count'])
        coarse_rows.append(dict(observation_resolution=n,final_points=5419,coarse_points=coarse,
            coarse_fraction_of_final=coarse/5419,adaptive_additions=5419-coarse,
            actual_final_fraction=5419/(n*n)))
    full_rows=[]
    baseline=case_means(85,'full',85*85)
    for n in RESOLUTIONS:
        vals=case_means(n,'full',n*n)
        full_rows.append(dict(observation_resolution=n,input_points=n*n,common_grid_error=float(vals.mean()),
            native_grid_error=float(row_for(n,'full','full')['mean_relative_l2_native']),
            error_increase_percent_vs85=100*(vals.mean()/baseline.mean()-1),
            cases_worse_than85=int(np.sum(vals>baseline)),test_cases=200))
    overlaps=[]
    for low, high in [(85,141),(85,211),(141,211)]:
        a=set(range(0,421,{85:5,141:3,211:2}[low])); b=set(range(0,421,{85:5,141:3,211:2}[high]))
        overlaps.append(dict(low_grid=low,high_grid=high,low_is_subset_of_high=a<=b,
                             shared_2d_nodes=len(a&b)**2,low_nodes=low*low,high_nodes=high*high))
    # Verify actual saved nested point sets, not merely the current code text.
    nesting=[]
    for sid in [1000,1099,1199]:
        for n in RESOLUTIONS:
            for method in ['cpg','ca']:
                path=SOURCE/'records/examples'/f'case{sid}_n{n}_{method}_k5419.npz'
                with np.load(path) as sample:
                    coarse=sample['coarse_indices'];selected=sample['indices']
                    assert np.array_equal(selected[:len(coarse)],coarse)
                    nesting.append(dict(case_id=sid,resolution=n,method=method,coarse=len(coarse),final=len(selected),prefix=True))
    write_csv('full_resolution.csv',full_rows)
    write_csv('strategy_comparisons.csv',comparisons)
    write_csv('derived_rauec.csv',rauc)
    write_csv('fixed_budget_composition.csv',coarse_rows)
    source_files=[raw_path,SOURCE/'results/resolution_budget.csv',SOURCE/'results/fixed_points.csv',SOURCE/'results/region_summary.csv',SOURCE/'protocol.md']
    audit=dict(passed=True,source_directory=str(SOURCE),raw_metric_rows=len(raw),physical_requests=len(records),
        condition_seed_groups=len(groups),cases_per_group=200,training_seeds=[0],sampling_seeds=[0,1,2],
        max_summary_recompute_difference=max_delta,full_paired_case_results=full_rows,
        grid_nesting=overlaps,saved_coarse_prefix_checks=nesting,
        timing_scope='single observations attached to accuracy requests; no formal latency inference',
        sources={str(p.relative_to(CURRENT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (OUT/'analysis_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plot(comparisons,full_rows,coarse_rows)
    print(json.dumps(dict(passed=True,physical_requests=len(records),summary_difference=max_delta,artifacts=str(OUT))))


def plot(comparisons,full,coarse):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6),layout='constrained')
    colors={'uniform':'#777777','cpg':'#0072B2','ca':'#009E73'}
    labels={'uniform':'Uniform 50%','cpg':'CPG 50%','ca':'CA 50%'}
    axes[0].plot(RESOLUTIONS,[r['common_grid_error'] for r in full],color='#222222',marker='D',label='Full')
    for method in colors:
        rows=[r for r in comparisons if r['condition']=='fraction_0.5' and r['method']==method]
        axes[0].errorbar(RESOLUTIONS,[r['mean_error'] for r in rows],yerr=[r['sampling_sd'] for r in rows],
            color=colors[method],marker={'uniform':'s','cpg':'o','ca':'^'}[method],capsize=2,label=labels[method])
    axes[0].set_title('(a) Frozen-model error');axes[0].set_ylabel('Physical relative L2 on 211² grid')
    axes[0].legend(frameon=False,fontsize=8)
    for method in ['cpg','ca']:
        for condition,style,text in [('fraction_0.5','-','50%'),('fixed_5419','--','K = 5,419')]:
            rows=[r for r in comparisons if r['condition']==condition and r['method']==method]
            axes[1].plot(RESOLUTIONS,[r['error_reduction_percent'] for r in rows],color=colors[method],
                marker='o' if method=='cpg' else '^',linestyle=style,label=method.upper()+' · '+text)
    axes[1].axhline(0,color='#999999',lw=.8);axes[1].set_title('(b) Benefit depends on budget')
    axes[1].set_ylabel('Error reduction versus Uniform (%)');axes[1].legend(frameon=False,fontsize=8)
    xs=np.arange(3);c=np.asarray([r['coarse_points'] for r in coarse]);extra=5419-c
    axes[2].bar(xs,c,color='#888888',label='Retained coarse points')
    axes[2].bar(xs,extra,bottom=c,color='#56B4E9',label='Adaptive additions')
    for x,low,high in zip(xs,c,extra):
        axes[2].text(x,low/2,f'{low:,}',ha='center',va='center',color='white',fontsize=8)
        axes[2].text(x,low+high/2,f'{high:,}',ha='center',va='center',color='#16384A',fontsize=8)
    axes[2].set_title('(c) Fixed K changes refinement freedom')
    axes[2].set_ylabel('Points in the final set');axes[2].set_xticks(xs,['85²','141²','211²']);axes[2].set_ylim(0,7200)
    axes[2].legend(frameon=False,fontsize=8,loc='upper right')
    for axis in axes[:2]:
        axis.set_xticks(RESOLUTIONS,['85²','141²','211²']);axis.grid(axis='y',alpha=.2)
    for axis in axes:axis.set_xlabel('Observation grid')
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'resolution_analysis.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    (OUT/'resolution_analysis.caption.md').write_text('Existing E4 observations on the same 200 historical cases with one frozen85²-trained Transolver3 checkpoint. '
        'All errors use the211² evaluation grid. (a) Full and50% methods; bars are SD across3 sampling-seed case means, not training SD. '
        '(b) Ratios of population-mean errors relative to same-condition Uniform; not rAUEC. '
        '(c) Both CPG and CA retain the entire coarse prefix while the final point count remains5419. '
        'No new inference, training, or formal timing was performed. Sources and independent recomputation: analysis_audit.json.\n')


if __name__=='__main__':main()
