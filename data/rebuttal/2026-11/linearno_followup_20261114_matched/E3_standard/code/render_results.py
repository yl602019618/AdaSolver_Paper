"""Reproducible paper tables and figures from independently checked CSVs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from collect_results import BASE, read, write_csv, write_json

DATASETS = [('darcy', 'Darcy'), ('airfoil2d', 'Airfoil')]
BACKBONES = ['Transolver3', 'LinearNO_point_temp']
BACKBONE_TEXT = {'Transolver3': 'Transolver3', 'LinearNO_point_temp': 'LinearNO (point-temp)',
                 'LinearNO_conv_temp': 'LinearNO (conv-temp, Full)'}
METHODS = ['uniform', 'pi', 'cpg', 'ca']
METHOD_TEXT = {'full': 'Full', 'uniform': 'Uniform', 'pi': 'PI', 'cpg': 'CPG', 'cpg_beta0': 'CPG (beta=0)', 'ca': 'CA'}
COLORS = {'uniform': '#777777', 'pi': '#E69F00', 'cpg': '#0072B2', 'ca': '#009E73'}
STYLES = {'Transolver3': ('-', 'o'), 'LinearNO_point_temp': ('--', 's')}
RECON = {'new': 'grid_qhull_sparse_apply_v1p1', 'old': 'grid_scipy_linear_nearest_native_v1'}
R = BASE / 'results'
FIG = BASE / 'figures'
TABLE = BASE / 'tables'


def rows(name):
    path = R / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def number(row, key):
    return float(row[key]) if row.get(key) not in ('', None, 'None') else float('nan')


def pick(source, **criteria):
    found = [r for r in source if all(str(r.get(k)) == str(v) for k, v in criteria.items())]
    if len(found) > 1:
        raise ValueError(('Ambiguous display cell', criteria))
    return found[0] if found else None


def output_figure(fig, name, source, caption):
    FIG.mkdir(exist_ok=True)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(FIG / f'{name}.{suffix}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    write_csv(FIG / f'{name}.csv', source)
    (FIG / f'{name}.caption.md').write_text(caption + '\n')


def tex_escape(value):
    return str(value).replace('_', r'\_').replace('%', r'\%')


def ranks(source):
    means = sorted(set(round(number(r, 'mean'), 4) for r in source))
    return {v: index + 1 for index, v in enumerate(means)}


def rauec_table(source, mode):
    relevant = [r for r in source if r['reconstruction_id'] == RECON[mode] and r['method'] in METHODS]
    if not relevant:
        return
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{rAUEC ($\downarrow$) of complete physical fields, with means above sampling standard deviations ($n=3$). '
        r'Each backbone is normalized by its own matched Uniform. CPG uses $\beta=1$. '
        r'Best and second means within each backbone and dataset are bold and underlined; ties use displayed precision. '
        r'N/A denotes pending experiments. Airfoil uses the fixed 110-case population with true $C_L>0.02$. '
        + ('Original' if mode == 'old' else 'Sparse-application') + r' reconstruction.}',
        r'\label{tab:linearno-standard-' + mode + '}',
        r'\setlength{\tabcolsep}{8pt}', r'\renewcommand{\arraystretch}{1.15}',
        r'\begin{tabular}{lcc}', r'\toprule', r'Method & Darcy Field & Airfoil Field \\', r'\midrule']
    markdown = ['| Backbone | Method | Darcy Field | Airfoil Field |', '| --- | --- | ---: | ---: |']
    for index, backbone in enumerate(BACKBONES):
        if index:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{3}{l}{\textit{' + tex_escape(BACKBONE_TEXT[backbone]) + r'}} \\')
        rank = {d: ranks([r for r in relevant if r['dataset'] == d and r['backbone'] == backbone]) for d, _ in DATASETS}
        for method in METHODS:
            latex_cells, md_cells = [], []
            for dataset, _ in DATASETS:
                row = pick(relevant, dataset=dataset, backbone=backbone, method=method)
                if row is None:
                    latex_cells.append('N/A'); md_cells.append('N/A (pending)'); continue
                value = f"{number(row, 'mean'):.4f}"
                position = rank[dataset][round(number(row, 'mean'), 4)]
                if position == 1:
                    value = r'\mathbf{' + value + '}'
                elif position == 2:
                    value = r'\underline{' + value + '}'
                latex_cells.append(r'$\begin{gathered}' + value + r'\\[-2pt]\pm ' + f"{number(row, 'sampling_sd'):.4f}" + r'\end{gathered}$')
                md_cells.append(f"{number(row, 'mean'):.4f} ± {number(row, 'sampling_sd'):.4f}")
            lines.append(METHOD_TEXT[method] + ' & ' + ' & '.join(latex_cells) + r' \\')
            markdown.append('| ' + ' | '.join([BACKBONE_TEXT[backbone], METHOD_TEXT[method], *md_cells]) + ' |')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (TABLE / f'rauec_{mode}.tex').write_text('\n'.join(lines))
    (TABLE / f'rauec_{mode}.md').write_text('\n'.join(markdown) + '\n\nSampling seeds n=3. N/A marks a pending cohort or method. Darcy PI uses beta=0.5; Airfoil PI and both CA use beta=1. Full is excluded from rAUEC.\n')
    write_csv(TABLE / f'rauec_{mode}.csv', relevant)


def rauec_plot(source, mode):
    values = [r for r in source if r['reconstruction_id'] == RECON[mode] and r['method'] in METHODS]
    if not values:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), layout='constrained')
    for axis, (dataset, title) in zip(axes, DATASETS):
        axis.set_title(title)
        axis.axhline(1., color='#333333', linestyle=':', lw=1.1, zorder=0)
        for bi, backbone in enumerate(BACKBONES):
            for mi, method in enumerate(METHODS):
                x = bi * 5 + mi
                row = pick(values, dataset=dataset, backbone=backbone, method=method)
                if row is None:
                    axis.text(x, .03, 'N/A', ha='center', va='bottom', rotation=90, fontsize=7)
                    continue
                axis.bar(x, number(row, 'mean'), width=.76, yerr=number(row, 'sampling_sd'),
                         color=COLORS[method], edgecolor='white', linewidth=.5, capsize=2.5)
        axis.set_xticks([1.5, 6.5], ['Transolver3', 'LinearNO\n(point-temp)'])
        axis.set_xlim(-.7, 8.7)
        axis.set_ylim(bottom=0)
        axis.set_ylabel('rAUEC (lower is better)')
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    handles = [Line2D([], [], color=COLORS[m], marker='s', linestyle='', label=METHOD_TEXT[m]) for m in METHODS]
    fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False)
    output_figure(fig, f'rauec_{mode}', values,
        'Relative area under the error–budget curve for three sampling seeds (mean ± sample SD). '
        'The three budgets are 37.5%, 50%, and 75%. Each backbone uses its own matched Uniform denominator. '
        f'Reconstruction: {RECON[mode]}. Airfoil uses the fixed published 110 cases with true Cl>0.02. N/A indicates pending evidence; Full is outside the integral.')


def curve_handles():
    return ([Line2D([], [], color=COLORS[m], label=METHOD_TEXT[m]) for m in METHODS] +
            [Line2D([], [], color='#333333', linestyle=STYLES[b][0], marker=STYLES[b][1], label=BACKBONE_TEXT[b]) for b in BACKBONES] +
            [Line2D([], [], color='#333333', linestyle=':', label='Full reference')])


def absolute_plot(source, mode):
    values = [r for r in source if r['reconstruction_id'] == RECON[mode] and r['method'] in ['full', *METHODS]]
    if not values:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout='constrained')
    for axis, (dataset, title) in zip(axes, DATASETS):
        axis.set_title(title)
        if not any(r['dataset'] == dataset for r in values):
            axis.text(.5, .5, 'Pending ' + title + ' results', transform=axis.transAxes,
                      ha='center', va='center', fontsize=11, color='#666666')
            axis.set_xticks([]); axis.set_yticks([])
            continue
        for bi, backbone in enumerate(BACKBONES):
            style, marker = STYLES[backbone]
            for method in METHODS:
                curve = sorted([r for r in values if r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == method], key=lambda r: number(r, 'budget_fraction'))
                if not curve:
                    continue
                if len(curve) != 3:
                    raise ValueError('Incomplete absolute-error curve')
                axis.errorbar([100 * number(r, 'budget_fraction') for r in curve], [number(r, 'mean_error') for r in curve],
                    yerr=[number(r, 'sampling_sd') for r in curve], color=COLORS[method], linestyle=style,
                    marker=marker, capsize=2, markersize=4, lw=1.3)
            full = pick(values, dataset=dataset, backbone=backbone, method='full')
            if full:
                y = number(full, 'mean_error')
                axis.plot([36.5, 76], [y, y], color='#333333', linestyle=':', lw=1)
                axis.plot([36.5 + bi * 2], [y], color='#333333', marker=marker, markersize=5)
        axis.set_xlabel('Retained points (%)')
        axis.set_xticks([37.5, 50, 75])
        axis.set_ylabel('Physical relative L2')
        axis.grid(alpha=.15)
    fig.legend(handles=curve_handles(), loc='outside upper center', ncol=4, frameon=False)
    output_figure(fig, f'absolute_error_budget_{mode}', values,
        'Absolute full-field errors at three retained budgets. Error bars are sample SD of three sampling-seed means. '
        'Both backbones share each dataset axis and complete test population. Horizontal references are each checkpoint’s Full error. '
        + ('The official conv-temp Full-only reference appears in the absolute-error table. '
         if any(r['backbone'] == 'LinearNO_conv_temp' for r in values)
         else 'The official conv-temp Full-only reference is still pending. ')
        + 'Airfoil uses the fixed published 110 cases with true Cl>0.02. '
        + 'Panels without completed observations are explicitly marked pending. '
        + f'Reconstruction: {RECON[mode]}.')


def mechanism_table(source):
    output = []
    for one in source:
        if one['method'] != 'cpg':
            continue
        criteria = {k: one[k] for k in ['dataset', 'backbone', 'checkpoint_id', 'test_population', 'reconstruction_id']}
        zero = pick(source, **criteria, method='cpg_beta0')
        if not zero:
            raise ValueError('Missing paired CPG beta=0')
        deltas = [number(one, f'seed{s}_rauec') - number(zero, f'seed{s}_rauec') for s in range(3)]
        output.append({**criteria, 'beta0_mean': number(zero, 'mean'), 'beta0_sd': number(zero, 'sampling_sd'),
                       'beta1_mean': number(one, 'mean'), 'beta1_sd': number(one, 'sampling_sd'),
                       'paired_delta_mean': float(np.mean(deltas)), 'paired_delta_sampling_sd': float(np.std(deltas, ddof=1)),
                       **{f'seed{s}_delta': deltas[s] for s in range(3)}})
    write_csv(TABLE / 'cpg_beta_pairs.csv', output)
    if output:
        text = ['| Dataset | Backbone | Reconstruction | beta=0 | beta=1 | Paired delta (1−0) |', '| --- | --- | --- | ---: | ---: | ---: |']
        latex = [r'\begin{table}[t]', r'\centering\small',
                 r'\caption{Paired CPG correction comparison. Entries are rAUEC mean $\pm$ sampling SD ($n=3$). Negative paired $\Delta$ favors $\beta=1$.}',
                 r'\setlength{\tabcolsep}{4pt}',
                 r'\begin{tabular}{lllccc}', r'\toprule',
                 r'Dataset & Backbone & Reconstruction & $\beta=0$ & $\beta=1$ & $\Delta(1-0)$ \\', r'\midrule']
        for row in output:
            text.append(f"| {row['dataset']} | {BACKBONE_TEXT[row['backbone']]} | {row['reconstruction_id']} | {row['beta0_mean']:.4f} ± {row['beta0_sd']:.4f} | {row['beta1_mean']:.4f} ± {row['beta1_sd']:.4f} | {row['paired_delta_mean']:.4f} ± {row['paired_delta_sampling_sd']:.4f} |")
            mode = 'Sparse application' if row['reconstruction_id'] == RECON['new'] else 'Original'
            numeric = [r'$\begin{gathered}' + f"{row[mean]:.4f}" + r'\\[-2pt]\pm ' + f"{row[sd]:.4f}" + r'\end{gathered}$'
                       for mean, sd in [('beta0_mean','beta0_sd'), ('beta1_mean','beta1_sd'), ('paired_delta_mean','paired_delta_sampling_sd')]]
            latex.append(' & '.join([tex_escape(row['dataset']), tex_escape(BACKBONE_TEXT[row['backbone']]), mode,
                *numeric]) + r' \\')
        (TABLE / 'cpg_beta_pairs.md').write_text('\n'.join(text) + '\n\nEach paired comparison shares the coarse prediction and retained indices. Negative delta favors beta=1. SD is across three sampling seeds.\n')
        latex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
        (TABLE / 'cpg_beta_pairs.tex').write_text('\n'.join(latex))


def reconstruction_pairs(latency):
    groups = defaultdict(dict)
    for row in latency:
        if row['method'] != 'full':
            key = tuple(row[k] for k in ['dataset', 'backbone', 'case_id', 'method', 'budget_fraction', 'sampling_seed', 'scope'])
            groups[key][row['implementation']] = row
    output = []
    for key, pair in groups.items():
        if set(pair) != {'old', 'new'}:
            raise ValueError(('Missing old/new timing pair', key))
        old, new = pair['old'], pair['new']
        for field in ['checkpoint_id', 'normalizer_id', 'selected_indices_hash', 'output_dtype', 'output_device', 'metric_target', 'test_population']:
            if old[field] != new[field]:
                raise ValueError(('Old/new timing mismatch', key, field))
        output.append({**dict(zip(['dataset', 'backbone', 'case_id', 'method', 'budget_fraction', 'sampling_seed', 'scope'], key)),
            'checkpoint_id': old['checkpoint_id'], 'normalizer_id': old['normalizer_id'], 'selected_indices_hash': old['selected_indices_hash'],
            'old_reconstruction_id': old['reconstruction_id'], 'new_reconstruction_id': new['reconstruction_id'],
            'old_field_error': old['field_error'], 'new_field_error': new['field_error'],
            'old_reconstruction_ms': old['reconstruction_ms'], 'new_reconstruction_ms': new['reconstruction_ms'],
            'reconstruction_speedup': number(old, 'reconstruction_ms') / number(new, 'reconstruction_ms'),
            'old_direct_total_ms': old['direct_total_ms'], 'new_direct_total_ms': new['direct_total_ms'],
            'complete_request_speedup': number(old, 'direct_total_ms') / number(new, 'direct_total_ms'),
            'old_p95_ms': old['p95_ms'], 'new_p95_ms': new['p95_ms'],
            'output_dtype': old['output_dtype'], 'output_device': old['output_device'],
            'old_first_request_ms': old['first_request_ms'], 'new_first_request_ms': new['first_request_ms'],
            'reuse_assumption': old['reuse_assumption'], 'old_source_json': old['source_json'], 'new_source_json': new['source_json']})
    write_csv(R / 'reconstruction_summary.csv', output)
    return output


def speedup_plot(pairs):
    source = [r for r in pairs if r['scope'] == 'A_cpu_cpu' and r['backbone'] in BACKBONES and r['method'] in METHODS]
    if not source:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    for di, (dataset, title) in enumerate(DATASETS):
        for ci, metric in enumerate(['reconstruction_speedup', 'complete_request_speedup']):
            axis = axes[di, ci]
            axis.set_title(title + (' · reconstruction' if ci == 0 else ' · complete request'))
            axis.axhline(1., color='#333333', linestyle=':', lw=1)
            for backbone in BACKBONES:
                style, marker = STYLES[backbone]
                for method in METHODS:
                    values = [r for r in source if r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == method]
                    if not values:
                        continue
                    y = [np.median([float(r[metric]) for r in values if float(r['budget_fraction']) == b]) for b in [.375, .5, .75]]
                    axis.plot([37.5, 50, 75], y, color=COLORS[method], linestyle=style, marker=marker, markersize=4)
            axis.set_xticks([37.5, 50, 75]); axis.set_xlabel('Retained points (%)')
            axis.set_ylabel('Old / new time'); axis.grid(alpha=.15)
    fig.legend(handles=curve_handles()[:-1], loc='outside upper center', ncol=4, frameon=False)
    output_figure(fig, 'reconstruction_speedup_scope_A', source,
        'Median paired old/new speed ratios across the three fixed timing cases and three sampling seeds at each budget. '
        'Left: reconstruction profiles. Right: directly measured complete request medians. Values below one indicate a slowdown. '
        'All pairs use CPU input and float64 CPU output, the same checkpoint, selected indices and reconstruction rule. '
        'The source CSV retains every case/seed condition and its p95.')


def latency_error_plot(latency, absolute):
    times = [r for r in latency if r['scope'] == 'A_cpu_cpu' and r['backbone'] in BACKBONES]
    if not times:
        return
    used = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.1), layout='constrained')
    for di, (dataset, title) in enumerate(DATASETS):
        for ci, mode in enumerate(['old', 'new']):
            axis = axes[di, ci]
            axis.set_title(title + (' · original reconstruction' if mode == 'old' else ' · sparse application'))
            for backbone in BACKBONES:
                style, marker = STYLES[backbone]
                for method in METHODS:
                    plotted = []
                    for budget in [.375, .5, .75]:
                        conditions = [r for r in times if r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == method
                                      and float(r['budget_fraction']) == budget and r['implementation'] == mode]
                        error = pick(absolute, dataset=dataset, backbone=backbone, method=method,
                                     budget_fraction=str(budget), reconstruction_id=RECON[mode])
                        if not conditions or not error:
                            continue
                        if len(conditions) != 9:
                            raise ValueError('Expected three cases x three sampling seeds per latency point')
                        x = float(np.median([number(r, 'direct_total_ms') for r in conditions]))
                        y = number(error, 'mean_error')
                        plotted.append((x, y))
                        used.append(dict(dataset=dataset, backbone=backbone, method=method, budget_fraction=budget,
                            reconstruction_id=RECON[mode], scope='A_cpu_cpu', median_of_condition_medians_ms=x,
                            mean_full_population_error=y, sampling_sd=error['sampling_sd'], timing_conditions=len(conditions)))
                    if plotted:
                        axis.plot(*zip(*plotted), color=COLORS[method], linestyle=style, marker=marker, markersize=4)
                full_times = [r for r in times if r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == 'full']
                full = pick(absolute, dataset=dataset, backbone=backbone, method='full', reconstruction_id=RECON[mode])
                if full and full_times:
                    full_ms = float(np.median([number(r, 'direct_total_ms') for r in full_times]))
                    used.append(dict(dataset=dataset, backbone=backbone, method='full', budget_fraction=1.,
                        reconstruction_id=RECON[mode], scope='A_cpu_cpu', median_of_condition_medians_ms=full_ms,
                        mean_full_population_error=number(full, 'mean_error'), sampling_sd='', timing_conditions=len(full_times)))
                    axis.plot(full_ms, number(full, 'mean_error'),
                              marker=marker, color='#222222', markerfacecolor='white', markersize=7, linestyle='')
            axis.set_xlabel('Complete request latency (ms)')
            axis.set_ylabel('Physical relative L2'); axis.grid(alpha=.15)
    fig.legend(handles=curve_handles()[:-1] + [Line2D([], [], marker='o', markerfacecolor='white', color='#222222', linestyle='', label='Full')],
               loc='outside upper center', ncol=4, frameon=False)
    output_figure(fig, 'absolute_error_latency_scope_A', used,
        'Absolute field error versus complete CPU-to-CPU request latency. Accuracy is the complete test-population mean; '
        'latency is the median of nine condition medians (three fixed timing cases × three sampling seeds), each based on 100 direct repeats. '
        'Full uses three case conditions and no sampling seed. Native and optimized reconstruction occupy separate panels. '
        'All outputs are float64 CPU fields. Budgets along each method curve are 37.5%, 50%, and 75%.')


def absolute_table(source):
    for dataset, title in DATASETS:
        for mode in RECON:
            current = [r for r in source if r['dataset'] == dataset and r['reconstruction_id'] == RECON[mode]]
            if not current:
                continue
            text = ['| Backbone | Method | Budget | Physical relative L2 | Sampling SD | Full error | Parameters | Epochs / updates |',
                    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            latex = [r'\begin{table}[t]', r'\centering\small',
                     r'\caption{' + title + r' physical field error. Sampling SD uses three seeds. Full has no sampling SD. '
                     + (r'Airfoil uses the fixed published 110 cases with true $C_L>0.02$. ' if dataset == 'airfoil2d' else '')
                     + ('The official conv-temp is a Full-only supplement and is excluded from rAUEC.'
                        if any(r['backbone']=='LinearNO_conv_temp' for r in current) else '') + '}',
                     r'\setlength{\tabcolsep}{6pt}', r'\begin{tabular}{lrrr}', r'\toprule',
                     r'Method & Budget & Error & SD \\', r'\midrule']
            previous_backbone = None
            for row in sorted(current, key=lambda r: (r['backbone'], ['full', *METHODS, 'cpg_beta0'].index(r['method']), number(r, 'budget_fraction'))):
                sd = f"{number(row, 'sampling_sd'):.6f}" if row['method'] != 'full' else 'N/A'
                text.append(f"| {BACKBONE_TEXT[row['backbone']]} | {METHOD_TEXT[row['method']]} | {100 * number(row, 'budget_fraction'):g}% | {number(row, 'mean_error'):.6f} | {sd} | {number(row, 'same_checkpoint_full_error'):.6f} | {row['parameters']} | {row['training_epochs']} / {row['training_updates']} |")
                if row['backbone'] != previous_backbone:
                    if previous_backbone is not None:
                        latex.append(r'\midrule')
                    latex += [r'\multicolumn{4}{l}{\textit{' + tex_escape(BACKBONE_TEXT[row['backbone']]) + r'}} \\',
                        r'\multicolumn{4}{l}{' + f"{int(row['parameters']):,} parameters; {row['training_epochs']} epochs" + r'} \\',
                        r'\multicolumn{4}{l}{' + f"{int(row['training_updates']):,} updates" + r'} \\',
                        r'\multicolumn{4}{l}{Same-checkpoint Full error: ' + f"{number(row, 'same_checkpoint_full_error'):.6f}" + r'} \\']
                    previous_backbone = row['backbone']
                latex.append(' & '.join([tex_escape(METHOD_TEXT[row['method']]),
                    f"{100 * number(row, 'budget_fraction'):g}" + r'\%', f"{number(row, 'mean_error'):.6f}", sd]) + r' \\')
            latex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
            (TABLE / f'absolute_errors_{dataset}_{mode}.md').write_text('\n'.join(text) + '\n')
            (TABLE / f'absolute_errors_{dataset}_{mode}.tex').write_text('\n'.join(latex))
            write_csv(TABLE / f'absolute_errors_{dataset}_{mode}.csv', current)


def efficiency_tables(latency):
    groups = defaultdict(list)
    keys = ['dataset', 'backbone', 'method', 'budget_fraction', 'reconstruction_id', 'implementation', 'scope']
    for row in latency:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, values in groups.items():
        identity = dict(zip(keys, key))
        summary = {**identity, 'conditions': len(values),
            'reconstruction_ms_median': float(np.median([number(r, 'reconstruction_ms') for r in values])),
            'direct_ms_median_of_condition_medians': float(np.median([number(r, 'direct_total_ms') for r in values])),
            'condition_p95_ms_min': min(number(r, 'p95_ms') for r in values), 'condition_p95_ms_max': max(number(r, 'p95_ms') for r in values),
            'peak_allocator_bytes_max': max(number(r, 'peak_cuda_allocated_bytes') for r in values),
            'first_request_ms_median': float(np.median([number(r, 'first_request_ms') for r in values])),
            'output_dtype': values[0]['output_dtype'], 'output_device': values[0]['output_device'],
            'reuse_assumption': values[0]['reuse_assumption'], 'memory_scope': values[0]['allocator_scope'],
            'external_memory_scope': values[0]['external_scope']}
        summaries.append(summary)
    write_csv(TABLE / 'efficiency_display_summary.csv', summaries)
    for dataset, _ in DATASETS:
        for scope in sorted({r['scope'] for r in summaries}):
            values = [r for r in summaries if r['dataset'] == dataset and r['scope'] == scope]
            if not values:
                continue
            lines = ['| Backbone | Method | Budget | Implementation | Reconstruction ms | Direct ms | Condition p95 range ms | Peak allocator GiB | First request ms |',
                     '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
            latex = [r'\begingroup\scriptsize', r'\setlength{\tabcolsep}{3pt}',
                     r'\begin{longtable}{lllrrrrrr}',
                     r'\caption{' + tex_escape(dataset + ', ' + scope) + r'. Times in ms; direct entries are medians of condition medians. The p95 column gives the range of condition p95s. Framework allocator memory is in GiB.}\\',
                     r'\toprule', r'Model & Method & Budget & Version & Recon. & Direct & p95 range & Memory & First \\',
                     r'\midrule\endfirsthead', r'\toprule',
                     r'Model & Method & Budget & Version & Recon. & Direct & p95 range & Memory & First \\', r'\midrule\endhead']
            for row in values:
                lines.append(f"| {BACKBONE_TEXT[row['backbone']]} | {METHOD_TEXT[row['method']]} | {100 * float(row['budget_fraction']):g}% | {row['implementation']} | {row['reconstruction_ms_median']:.3f} | {row['direct_ms_median_of_condition_medians']:.3f} | {row['condition_p95_ms_min']:.3f}–{row['condition_p95_ms_max']:.3f} | {row['peak_allocator_bytes_max'] / 2**30:.3f} | {row['first_request_ms_median']:.3f} |")
                short = {'Transolver3': 'T3', 'LinearNO_point_temp': 'LNO point', 'LinearNO_conv_temp': 'LNO conv'}[row['backbone']]
                latex.append(' & '.join([short, tex_escape(METHOD_TEXT[row['method']]), f"{100 * float(row['budget_fraction']):g}" + r'\%',
                    'shared' if row['implementation'] == 'full_shared' else tex_escape(row['implementation']),
                    f"{row['reconstruction_ms_median']:.3f}", f"{row['direct_ms_median_of_condition_medians']:.3f}",
                    f"{row['condition_p95_ms_min']:.3f}--{row['condition_p95_ms_max']:.3f}",
                    f"{row['peak_allocator_bytes_max'] / 2**30:.3f}", f"{row['first_request_ms_median']:.3f}"]) + r' \\')
            lines += ['', 'Direct ms is the median of condition medians, not a pooled request median. Each condition has 100 formal direct repeats. '
                      'The p95 column reports the range of condition p95s; the condition-level CSV retains every p95. '
                      'Reconstruction profiles are measured separately. Memory includes framework allocator peaks, with the external allocation scope preserved in the CSV. '
                      'Output device/dtype, preparation and reuse assumptions are explicit in the source table.', '']
            (TABLE / f'efficiency_{dataset}_{scope}.md').write_text('\n'.join(lines))
            latex += [r'\bottomrule', r'\end{longtable}', r'\endgroup', '']
            (TABLE / f'efficiency_{dataset}_{scope}.tex').write_text('\n'.join(latex))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    state = read(R / 'collection_status.json')
    check = read(R / 'independent_statistics_check.json')
    if args.require_complete and not (state.get('accuracy_complete') and state.get('latency_complete')):
        raise RuntimeError('Final rendering requires completed accuracy and timing matrices')
    if not check.get('passed'):
        print('Waiting for independently checked real result rows; no numeric figures generated.')
        return
    FIG.mkdir(exist_ok=True); TABLE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    areas = rows('backbone_rauec.csv'); absolute = rows('backbone_absolute_errors.csv'); latency = rows('backbone_latency_summary.csv')
    for mode in RECON:
        rauec_table(areas, mode); rauec_plot(areas, mode); absolute_plot(absolute, mode)
    absolute_table(absolute); mechanism_table(areas)
    if latency:
        pairs = reconstruction_pairs(latency); speedup_plot(pairs); efficiency_tables(latency); latency_error_plot(latency, absolute)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [R / 'backbone_rauec.csv', R / 'backbone_absolute_errors.csv', R / 'backbone_latency_summary.csv'] if path.exists()}
    write_json(FIG / 'render_manifest.json', dict(inputs=inputs, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accuracy_complete=state['accuracy_complete'], latency_complete=state['latency_complete'],
        outputs=[str(p) for p in sorted(FIG.glob('*')) if p.suffix in ['.pdf', '.svg', '.png']],
        render_qa='pending actual image/table inspection', source_number_policy='CSV only; missing values are N/A, never zero'))
    print('Rendered available checked cohorts. Actual image inspection is still required.')


if __name__ == '__main__':
    main()
