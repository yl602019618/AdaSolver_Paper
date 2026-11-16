"""Recompute E4's published Darcy analysis from public scalar CSVs.

The default verification uses only Python's standard library. It does not load
weights, private prediction arrays, or the original 421-grid caches. --plot
additionally requires Matplotlib. The original acquisition/point-set audit is
separate: scalar CSVs cannot revalidate its private NPZ prefix checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


RESOLUTIONS = (85, 141, 211)
BUDGETS = (.375, .5, .75)
METHODS = ('uniform', 'pi', 'cpg', 'ca')
REGIONS = ('boundary', 'interface', 'smooth')
CASES = tuple(range(1000, 1200))
METRICS = ('field_relative_l2_common', 'field_relative_l2_native')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return math.fsum(values) / len(values)


def area(curve):
    # Exact trapezoidal weights for the nominal (.375, .5, .75) budgets.
    return .0625 * curve[0] + .1875 * curve[1] + .125 * curve[2]


class Verifier:
    def __init__(self):
        self.numeric_checks = 0
        self.max_difference = 0.

    def close(self, actual, expected, label, *, rtol=1e-10, atol=1e-12):
        actual, expected = float(actual), float(expected)
        require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite value: ' + label)
        difference = abs(actual - expected)
        require(difference <= atol + rtol * abs(expected),
                f'Numeric mismatch: {label}: {actual!r} vs {expected!r}')
        self.numeric_checks += 1
        self.max_difference = max(self.max_difference, difference)


def raw_matrix(data_dir):
    records = read_csv(data_dir / 'metrics_records.csv')
    require(len(records) == 54000, 'Expected 54,000 common/native metric rows')
    require({r['training_seed'] for r in records} == {'0'}, 'Expected the frozen training-seed0 cohort')
    require(len({r['checkpoint_sha256'] for r in records}) == 1, 'Mixed checkpoints')
    values = {}
    for row in records:
        n = int(row['observation_resolution'])
        require(row['dataset'] == 'darcy' and int(row['evaluation_resolution']) == 211,
                'Unexpected dataset or evaluation grid')
        require(row['metric'] in METRICS and row['unit'] == 'ratio', 'Unexpected metric or unit')
        key = (n, row['method'], int(row['retained_points']), row['sampling_seed'],
               int(row['case_id']), row['metric'])
        require(key not in values, 'Duplicate raw identity: ' + repr(key))
        value = float(row['value'])
        require(math.isfinite(value) and value >= 0, 'Invalid field error')
        values[key] = value
    expected = set()
    for n in RESOLUTIONS:
        for sid in CASES:
            for metric in METRICS:
                expected.add((n, 'full', n*n, '', sid, metric))
            keeps = {int(round(n*n*b)) for b in BUDGETS} | {5419}
            for method in METHODS:
                for keep in keeps:
                    for seed in ('0', '1', '2'):
                        for metric in METRICS:
                            expected.add((n, method, keep, seed, sid, metric))
    require(set(values) == expected, 'Raw case/resolution/method/budget/seed matrix differs')
    return records, values


def verify_original_summaries(data_dir, values, check):
    mean_cache = {}

    def seed_mean(n, method, keep, seed, metric='field_relative_l2_common'):
        key = (n, method, keep, seed, metric)
        if key not in mean_cache:
            mean_cache[key] = mean([values[n, method, keep, seed, sid, metric] for sid in CASES])
        return mean_cache[key]

    summary_index = {}
    for filename, count in [('resolution_budget.csv', 39), ('fixed_points.csv', 12)]:
        source = read_csv(data_dir / filename)
        require(len(source) == count, 'Wrong row count: ' + filename)
        seen = set()
        for row in source:
            n, method, condition = int(row['observation_resolution']), row['method'], row['condition']
            key = (n, method, condition)
            require(key not in seen, 'Duplicate summary: ' + repr(key)); seen.add(key)
            if condition == 'full':
                require(method == 'full', 'Full identity mismatch'); keep = n*n
            elif condition == 'fixed_5419':
                keep = 5419
            else:
                budget = float(condition.removeprefix('fraction_'))
                require(budget in BUDGETS, 'Unknown fractional condition'); keep = int(round(n*n*budget))
            require(int(row['retained_points']) == keep, 'Retained count mismatch')
            require(int(row['evaluation_resolution']) == 211 and int(row['cases']) == 200,
                    'Summary grid/population differs')
            seeds = ('',) if method == 'full' else ('0', '1', '2')
            require(int(row['sampling_seeds']) == (0 if method == 'full' else 3), 'Sampling count differs')
            require(int(row['records']) == 200*len(seeds), 'Raw count differs')
            check.close(keep/(n*n), row['actual_budget_fraction'], str(key)+' budget')
            coarse = int(round(n*n*.1)) if method in ('cpg', 'ca') else 0
            require(int(row['coarse_count']) == coarse, 'Coarse count differs')
            means = [seed_mean(n, method, keep, seed) for seed in seeds]
            check.close(mean(means), row['mean_relative_l2_common'], str(key)+' common')
            check.close(statistics.stdev(means) if len(means)>1 else 0., row['sampling_seed_std'], str(key)+' SD')
            native = [seed_mean(n, method, keep, seed, METRICS[1]) for seed in seeds]
            check.close(mean(native), row['mean_relative_l2_native'], str(key)+' native')
            summary_index[key] = row
        expected_conditions = ({(n, m, 'fixed_5419') for n in RESOLUTIONS for m in METHODS}
            if filename == 'fixed_points.csv' else
            {(n, 'full', 'full') for n in RESOLUTIONS} |
            {(n, m, f'fraction_{b}') for n in RESOLUTIONS for m in METHODS for b in BUDGETS})
        require(seen == expected_conditions, 'Unexpected summary condition set: ' + filename)
    return summary_index, seed_mean


def derive_tables(values, summary, seed_mean):
    def case_means(n, method, keep):
        seeds = ('',) if method == 'full' else ('0', '1', '2')
        return [mean([values[n, method, keep, seed, sid, METRICS[0]] for seed in seeds]) for sid in CASES]

    full_rows, coarse_rows, comparisons, rauec = [], [], [], []
    baseline = case_means(85, 'full', 85*85)
    for n in RESOLUTIONS:
        observed = case_means(n, 'full', n*n)
        full_error = mean(observed)
        full_rows.append(dict(observation_resolution=n, input_points=n*n, common_grid_error=full_error,
            native_grid_error=seed_mean(n, 'full', n*n, '', METRICS[1]),
            error_increase_percent_vs85=100*(full_error/mean(baseline)-1),
            cases_worse_than85=sum(a>b for a,b in zip(observed, baseline)), test_cases=200))
        coarse = int(summary[n, 'cpg', 'fixed_5419']['coarse_count'])
        coarse_rows.append(dict(observation_resolution=n, final_points=5419, coarse_points=coarse,
            coarse_fraction_of_final=coarse/5419, adaptive_additions=5419-coarse, actual_final_fraction=5419/(n*n)))
        for condition in ('fraction_0.5', 'fixed_5419', 'fraction_0.75'):
            keep = int(summary[n, 'uniform', condition]['retained_points'])
            uniform_means = [seed_mean(n, 'uniform', keep, s) for s in ('0','1','2')]
            uniform_error = mean(uniform_means)
            for method in METHODS:
                means = [seed_mean(n, method, keep, s) for s in ('0','1','2')]
                error = mean(means)
                wins = None if method == 'uniform' else sum(a<b for a,b in
                    zip(case_means(n, method, keep), case_means(n, 'uniform', keep)))
                comparisons.append(dict(observation_resolution=n, condition=condition, method=method,
                    retained_points=keep, actual_budget_fraction=keep/(n*n), full_error=full_error,
                    mean_error=error, sampling_sd=statistics.stdev(means), matched_uniform_error=uniform_error,
                    error_reduction_percent=100*(1-error/uniform_error), cases_better_than_uniform=wins,
                    test_cases=200, training_seeds=1, sampling_seeds=3))
        for method in METHODS:
            ratios = []
            for seed in ('0','1','2'):
                curve = [seed_mean(n, method, int(round(n*n*b)), seed) for b in BUDGETS]
                reference = [seed_mean(n, 'uniform', int(round(n*n*b)), seed) for b in BUDGETS]
                require(area(reference)>0, 'Nonpositive Uniform area')
                ratios.append(area(curve)/area(reference))
            rauec.append(dict(observation_resolution=n, method=method, seed0_rauec=ratios[0],
                seed1_rauec=ratios[1], seed2_rauec=ratios[2], mean_rauec=mean(ratios),
                sampling_sd=statistics.stdev(ratios), training_seeds=1,
                integration='nominal budgets0.375/0.5/0.75; case means then seedwise matched Uniform area ratios',
                origin='derived from existing E4 records; no new experiment'))
    return {'full_resolution.csv': full_rows, 'fixed_budget_composition.csv': coarse_rows,
            'strategy_comparisons.csv': comparisons, 'derived_rauec.csv': rauec}


def compare_derived(derived_dir, tables, check):
    keys = {'full_resolution.csv': ('observation_resolution',),
            'fixed_budget_composition.csv': ('observation_resolution',),
            'strategy_comparisons.csv': ('observation_resolution', 'condition', 'method'),
            'derived_rauec.csv': ('observation_resolution', 'method')}
    for name, recomputed in tables.items():
        published = read_csv(derived_dir / name)
        fields = keys[name]
        key = lambda row: tuple(str(row[k]) for k in fields)
        index = {key(row): row for row in published}
        require(len(index) == len(published) == len(recomputed), 'Derived row count/duplicates: '+name)
        require(set(index) == {key(r) for r in recomputed}, 'Derived identities differ: '+name)
        for row in recomputed:
            original = index[key(row)]
            for field, value in row.items():
                if isinstance(value, (int, float)):
                    check.close(value, original[field], name+' '+str(key(row))+' '+field)
                elif value is None:
                    require(original.get(field) in ('', 'None'), 'Missing value changed: '+name+' '+field)


def verify_regions(derived_dir, values, check):
    inputs = read_csv(derived_dir / 'region_energy_inputs.csv')
    require(len(inputs) == 1800, 'Expected 1,800 region-energy scalar input rows')
    seen = set(); per_prediction = defaultdict(list); truth_norms = {}; components = defaultdict(list)
    totals = defaultdict(list); output_rows = []
    for row in inputs:
        sid, n, region = int(row['case_id']), int(row['observation_resolution']), row['region']
        key = (sid, n, region)
        require(sid in CASES and n in RESOLUTIONS and region in REGIONS and key not in seen,
                'Invalid/duplicate region input identity: '+repr(key)); seen.add(key)
        region_error = float(row['region_relative_l2'])
        region_norm, full_norm = float(row['truth_region_norm']), float(row['truth_full_norm'])
        require(all(math.isfinite(v) and v>=0 for v in (region_error, region_norm, full_norm)) and full_norm>0,
                'Invalid regional norm')
        original = values[n, 'full', n*n, '', sid, METRICS[0]]
        check.close(row['full_relative_l2'], original, repr(key)+' Full source')
        if (sid, region) in truth_norms:
            require(truth_norms[sid, region] == (region_norm, full_norm), 'Truth norms changed across observation grids')
        else:
            truth_norms[sid, region] = (region_norm, full_norm)
        weight = region_norm**2/full_norm**2
        contribution = region_error**2*weight
        per_prediction[sid,n].append(contribution)
        components[n,region].append((contribution, weight))
        output_rows.append(dict(case_id=sid, observation_resolution=n, region=region,
            truth_energy_fraction=weight, normalized_error_energy=contribution,
            full_squared_relative_l2=original**2))
    require(seen == {(sid,n,r) for sid in CASES for n in RESOLUTIONS for r in REGIONS}, 'Incomplete region inputs')
    max_residual = 0.
    for (sid,n), pieces in per_prediction.items():
        direct = values[n, 'full', n*n, '', sid, METRICS[0]]**2
        rebuilt = math.fsum(pieces)
        # Independent original FP32 truth-norm rounding contract.
        require(abs(rebuilt-direct)<=1e-12+3e-6*abs(direct), 'Region energy does not reconstruct Full metric')
        max_residual = max(max_residual, abs(rebuilt-direct)/max(direct, 1e-30))
        totals[n].append(direct)
    summary=[]
    for n in RESOLUTIONS:
        total = mean(totals[n]); increase = total-mean(totals[85])
        for region in REGIONS:
            energy = mean([x[0] for x in components[n,region]])
            base = mean([x[0] for x in components[85,region]])
            summary.append(dict(observation_resolution=n,region=region,mean_normalized_squared_error=energy,
                share_of_mean_squared_relative_error=energy/total,
                share_of_increase_vs85=None if n==85 else (energy-base)/increase,
                mean_truth_energy_fraction=mean([x[1] for x in components[n,region]])))
    published = read_csv(derived_dir/'region_error_energy.csv')
    index={(int(r['observation_resolution']),r['region']):r for r in published}
    require(len(published)==len(index)==9, 'Expected nine regional summary rows')
    for row in summary:
        original=index[row['observation_resolution'],row['region']]
        for field,value in row.items():
            if field in ('observation_resolution','region'):
                continue
            if value is None:
                require(original[field] in ('','None'),'Baseline region delta must be empty')
            else:
                check.close(value,original[field],'region '+repr((row['observation_resolution'],row['region'],field)))
    return summary, output_rows, max_residual


def plot(output_dir, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    comparisons=tables['strategy_comparisons.csv'];full=tables['full_resolution.csv'];coarse=tables['fixed_budget_composition.csv']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6),layout='constrained')
    colors={'uniform':'#777777','cpg':'#0072B2','ca':'#009E73'}
    labels={'uniform':'Uniform 50%','cpg':'CPG 50%','ca':'CA 50%'}
    axes[0].plot(RESOLUTIONS,[r['common_grid_error'] for r in full],color='#222222',marker='D',label='Full')
    for method in colors:
        subset=[r for r in comparisons if r['condition']=='fraction_0.5' and r['method']==method]
        axes[0].errorbar(RESOLUTIONS,[r['mean_error'] for r in subset],yerr=[r['sampling_sd'] for r in subset],
            color=colors[method],marker={'uniform':'s','cpg':'o','ca':'^'}[method],capsize=2,label=labels[method])
    axes[0].set_title('(a) Frozen-model error');axes[0].set_ylabel('Physical relative L2 on 211² grid')
    axes[0].legend(frameon=False,fontsize=8)
    for method in ('cpg','ca'):
        for condition,style,label in [('fraction_0.5','-','50%'),('fixed_5419','--','K = 5,419')]:
            subset=[r for r in comparisons if r['condition']==condition and r['method']==method]
            axes[1].plot(RESOLUTIONS,[r['error_reduction_percent'] for r in subset],color=colors[method],
                marker='o' if method=='cpg' else '^',linestyle=style,label=method.upper()+' · '+label)
    axes[1].axhline(0,color='#999999',lw=.8);axes[1].set_title('(b) Benefit depends on budget')
    axes[1].set_ylabel('Error reduction versus Uniform (%)');axes[1].legend(frameon=False,fontsize=8)
    xs=list(range(3));counts=[r['coarse_points'] for r in coarse];extra=[5419-v for v in counts]
    axes[2].bar(xs,counts,color='#888888',label='Retained coarse points')
    axes[2].bar(xs,extra,bottom=counts,color='#56B4E9',label='Adaptive additions')
    for x,low,high in zip(xs,counts,extra):
        axes[2].text(x,low/2,f'{low:,}',ha='center',va='center',color='white',fontsize=8)
        axes[2].text(x,low+high/2,f'{high:,}',ha='center',va='center',color='#16384A',fontsize=8)
    axes[2].set_title('(c) Fixed K changes refinement freedom');axes[2].set_ylabel('Points in the final set')
    axes[2].set_xticks(xs,['85²','141²','211²']);axes[2].set_ylim(0,7200);axes[2].legend(frameon=False,fontsize=8,loc='upper right')
    for axis in axes[:2]:
        axis.set_xticks(RESOLUTIONS,['85²','141²','211²']);axis.grid(axis='y',alpha=.2)
    for axis in axes:
        axis.set_xlabel('Observation grid')
    for ext in ('png','svg','pdf'):
        fig.savefig(output_dir/f'resolution_analysis.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Published data/rebuttal/2026-11/E4 directory')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plot',action='store_true',help='Also redraw PDF, SVG and 300 dpi PNG with Matplotlib')
    args=parser.parse_args();data=args.data_dir.resolve();derived=data/'derived_analysis'
    check=Verifier();records,values=raw_matrix(data)
    summary,seed_mean=verify_original_summaries(data,values,check)
    tables=derive_tables(values,summary,seed_mean);compare_derived(derived,tables,check)
    region_summary,region_case_rows,residual=verify_regions(derived,values,check)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():write_csv(args.output_dir/name,rows)
    write_csv(args.output_dir/'region_error_energy.csv',region_summary)
    write_csv(args.output_dir/'region_error_energy_per_case.csv',region_case_rows)
    sources=[data/'metrics_records.csv',data/'resolution_budget.csv',data/'fixed_points.csv',
             derived/'region_energy_inputs.csv',derived/'region_error_energy.csv']
    receipt=dict(passed=True,scope='Public scalar-record recomputation of existing E4 observations and their derived analysis',
        raw_metric_rows=len(records),physical_requests=27000,condition_seed_groups=135,cases_per_group=200,
        training_seeds=[0],sampling_seeds=[0,1,2],original_resolution_summary_rows=39,original_fixed_point_summary_rows=12,
        derived_rauec_rows=len(tables['derived_rauec.csv']),derived_strategy_rows=len(tables['strategy_comparisons.csv']),
        full_resolution_rows=3,fixed_budget_composition_rows=3,region_input_rows=1800,full_region_reconstructions=600,region_summary_rows=9,
        numeric_checks=check.numeric_checks,maximum_numeric_difference=check.max_difference,
        maximum_relative_region_reconstruction_residual=residual,
        private_array_prefix_checks='Not rerun by this public scalar verifier; see the separately published original audit receipt',
        no_training_or_inference=True,formal_latency_verified=False,
        source_sha256={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output_dir/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    if args.plot:plot(args.output_dir,tables)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
