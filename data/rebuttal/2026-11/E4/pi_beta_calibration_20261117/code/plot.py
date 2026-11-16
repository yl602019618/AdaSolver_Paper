"""Plot accepted PI beta calibration and fixed-selection final results.

This module only reads the locked selection. It never chooses a beta, changes
the protocol, clips unfavorable observations, or executes a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (85, 141, 211)
BASELINE = .5
COLORS = {'baseline': '#777777', 'shared': '#0072B2', 'per_resolution': '#009E73'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def csv_snapshot(path):
    payload = path.read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode()))), hashlib.sha256(payload).hexdigest()


def write_csv(path, rows):
    require(bool(rows), f'No real observations to export at {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def beta_value(row):
    return None if row['beta'] == '' else float(row['beta'])


def table_index(rows, split):
    result = {}
    for row in rows:
        require(row['split'] == split, 'Display table has an unexpected split')
        resolution = int(row['resolution'])
        require(resolution in RESOLUTIONS, 'Unexpected resolution in display table')
        require(int(row['training_seed_count']) == 1 and int(row['sampling_seed_count']) == 3,
                'The displayed uncertainty must refer to3 sampling seeds of one training model')
        require(int(row['cases']) == (100 if split == 'validation' else 200), 'Unexpected population size')
        value, sd = float(row['mean_rauec']), float(row['sampling_sd'])
        require(math.isfinite(value) and math.isfinite(sd) and value >= 0 and sd >= 0, 'Nonfinite display value')
        key = (resolution, row['method'], beta_value(row))
        require(key not in result, 'Duplicate display condition')
        result[key] = row
    for resolution in RESOLUTIONS:
        uniform = result.get((resolution, 'uniform', None))
        require(uniform is not None and float(uniform['mean_rauec']) == 1., 'Matched Uniform must appear at1')
    return result


def limits(rows, mean_key='mean_rauec', sd_key='sampling_sd', reference=1.):
    low = min([reference] + [float(r[mean_key])-float(r[sd_key]) for r in rows])
    high = max([reference] + [float(r[mean_key])+float(r[sd_key]) for r in rows])
    span = max(high-low, .002)
    return low-.09*span, high+.14*span


def style_axis(axis):
    axis.grid(axis='y', alpha=.16)
    axis.set_axisbelow(True)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)


def export(fig, folder, name, source_rows, caption):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(folder/f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    write_csv(folder/f'{name}.csv', source_rows)
    (folder/f'{name}.caption.md').write_text(caption + '\n')


def plot_validation(rows, index, selection, folder):
    expected_betas = set(float(b) for b in selection['beta_candidates'])
    for resolution in RESOLUTIONS:
        require({beta for r, method, beta in index if r==resolution and method=='pi'} == expected_betas,
                'Calibration curve is missing a predeclared beta candidate')
    pi_rows = [r for r in rows if r['method']=='pi']
    ylimits = limits(pi_rows)
    fig, axes = plt.subplots(1,3,figsize=(10.2,3.65),layout='constrained',sharey=True)
    shared = float(selection['shared_beta'])
    for axis, resolution in zip(axes, RESOLUTIONS):
        ordered = sorted([r for r in pi_rows if int(r['resolution'])==resolution],key=lambda r:float(r['beta']))
        axis.errorbar([float(r['beta']) for r in ordered], [float(r['mean_rauec']) for r in ordered],
            yerr=[float(r['sampling_sd']) for r in ordered], color=COLORS['shared'],marker='o',
            lw=1.5,markersize=4,capsize=2.5)
        axis.axhline(1.,color='#222222',linestyle='--',lw=1.)
        axis.axvline(BASELINE,color=COLORS['baseline'],linestyle=':',lw=1.)
        shared_row=index[resolution,'pi',shared]
        specific=float(selection['per_resolution_beta'][str(resolution)])
        specific_row=index[resolution,'pi',specific]
        axis.plot(shared,float(shared_row['mean_rauec']),marker='*',color='#D55E00',markersize=12,
                  markeredgecolor='white',markeredgewidth=.6,linestyle='',zorder=5)
        axis.plot(specific,float(specific_row['mean_rauec']),marker='D',markerfacecolor='none',
                  markeredgecolor=COLORS['per_resolution'],markersize=8,markeredgewidth=1.5,linestyle='',zorder=6)
        axis.set_title(f'{resolution}² observations')
        axis.set_xticks([0,.5,1,1.5,2]);axis.set_xlabel('Compensation exponent β')
        axis.set_ylim(*ylimits);style_axis(axis)
    axes[0].set_ylabel('Validation rAUEC')
    legend=[Line2D([],[],color=COLORS['shared'],marker='o',label='Mean ± sampling SD'),
            Line2D([],[],color='#222222',linestyle='--',label='Uniform = 1'),
            Line2D([],[],color=COLORS['baseline'],linestyle=':',label='Original β = 0.5'),
            Line2D([],[],color='#D55E00',marker='*',markersize=10,linestyle='',label=f'Shared validation β = {shared:g}'),
            Line2D([],[],color=COLORS['per_resolution'],marker='D',markerfacecolor='none',linestyle='',label='Per-resolution validation β')]
    fig.legend(handles=legend,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    export(fig,folder,'validation_beta_curves',rows,
        'Darcy PI calibration on100 cases disjoint from the final E4 population. All eleven predeclared beta values are displayed, '
        'including exponents1.5 and2 above1. Each point is the mean of three sampling-seed AUEC ratios to its matched Uniform; '
        'error bars are sample SD across those sampling seeds, using one frozen training-seed0 model. The shared beta minimizes '
        'the maximum resolution mean on calibration data. Per-resolution markers are separate calibration-only minima. '
        'Final-test results do not select beta. Values above1 remain visible. Integration uses budgets37.5%,50%,75%; Full is excluded.')


def plot_objective(selection, folder):
    rows=selection['validation_candidates'];ordered=sorted(rows,key=lambda r:float(r['beta']))
    fig,axis=plt.subplots(figsize=(5.5,3.5),layout='constrained')
    axis.plot([float(r['beta']) for r in ordered],[float(r['worst_resolution_mean_rauec']) for r in ordered],
              marker='o',color=COLORS['shared'],lw=1.5,markersize=4)
    chosen=next(r for r in ordered if float(r['beta'])==float(selection['shared_beta']))
    axis.plot(float(chosen['beta']),float(chosen['worst_resolution_mean_rauec']),marker='*',color='#D55E00',markersize=12,
              linestyle='',label=f"Selected β = {float(selection['shared_beta']):g}")
    axis.axhline(1.,color='#222222',ls='--',lw=1,label='Uniform = 1')
    axis.set_xticks([0,.5,1,1.5,2]);axis.set_xlabel('Compensation exponent β')
    axis.set_ylabel('Worst-resolution validation mean rAUEC')
    axis.set_title('Shared-beta calibration objective');style_axis(axis);axis.legend(frameon=False)
    export(fig,folder,'validation_shared_objective',ordered,
        'The predeclared shared-beta objective is the maximum of the three resolution-specific calibration mean rAUECs. '
        'This maximum of means has no sampling-SD bar attached; the underlying sampling SDs appear in validation_beta_curves. '
        'The selected marker is read from immutable selection.json. The plot performs no selection and uses no final-test values.')


def display_test(index, selection, comparisons):
    compare={int(r['resolution']):r for r in comparisons}
    require(set(compare)==set(RESOLUTIONS),'Expected three resolution rows in selected-beta comparison')
    display=[]; paired=[]
    for resolution in RESOLUTIONS:
        baseline=index[resolution,'pi',BASELINE]
        choice=compare[resolution]
        policies=[('baseline',BASELINE,'Original β = 0.5'),
                  ('shared',float(selection['shared_beta']),'Shared validation β'),
                  ('per_resolution',float(selection['per_resolution_beta'][str(resolution)]),'Per-resolution validation β')]
        for policy,beta,label in policies:
            row=index[resolution,'pi',beta]
            value=float(row['mean_rauec']);sd=float(row['sampling_sd'])
            if policy=='shared':
                require(float(choice['shared_beta'])==beta and float(choice['shared_mean_rauec'])==value and float(choice['shared_sampling_sd'])==sd,
                        'Shared display differs from accepted comparison CSV')
            elif policy=='per_resolution':
                require(float(choice['resolution_specific_beta'])==beta and float(choice['resolution_specific_mean_rauec'])==value
                        and float(choice['resolution_specific_sampling_sd'])==sd,'Secondary display differs from accepted comparison CSV')
            else:
                require(float(choice['baseline_mean_rauec'])==value and float(choice['baseline_sampling_sd'])==sd,
                        'Original-beta display differs from accepted comparison CSV')
            display.append(dict(resolution=resolution,policy=policy,policy_label=label,beta=beta,
                mean_rauec=value,sampling_sd=sd,cases=200,training_models=1,sampling_seeds=3,
                alias_of_original_condition=beta==BASELINE,source_method='pi',
                seed0_rauec=float(row['seed0_rauec']),seed1_rauec=float(row['seed1_rauec']),seed2_rauec=float(row['seed2_rauec'])))
            if policy=='baseline':continue
            deltas=[float(row[f'seed{s}_rauec'])-float(baseline[f'seed{s}_rauec']) for s in range(3)]
            delta=statistics.mean(deltas)
            require(math.isclose(delta,value-float(baseline['mean_rauec']),rel_tol=1e-12,abs_tol=1e-14),
                    'Paired beta difference does not reproduce difference of means')
            paired.append(dict(split='test',resolution=resolution,policy=policy,selected_beta=beta,baseline_beta=BASELINE,
                baseline_mean_rauec=float(baseline['mean_rauec']),selected_mean_rauec=value,
                selected_mean_below_uniform=value<1.,mean_delta=delta,paired_sampling_sd=statistics.stdev(deltas),
                seed0_delta=deltas[0],seed1_delta=deltas[1],seed2_delta=deltas[2],
                sampling_seeds=3,training_models=1,cases=200,alias_of_baseline=beta==BASELINE,
                interpretation='negative delta improves on beta0.5; selected mean<1 separately improves on matched Uniform',
                beta_selection='validation only; test values do not select the policy'))
    return display,paired


def plot_test(display, folder):
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.7),sharey=True,layout='constrained')
    ylimits=limits(display)
    for axis,resolution in zip(axes,RESOLUTIONS):
        current=[r for r in display if r['resolution']==resolution]
        for x,row in enumerate(current):
            axis.errorbar(x,row['mean_rauec'],yerr=row['sampling_sd'],color=COLORS[row['policy']],
                          marker=['s','o','^'][x],markersize=7,capsize=4,linestyle='',elinewidth=1.3)
        labels=[f"Original\nβ = {current[0]['beta']:g}",f"Shared\nβ = {current[1]['beta']:g}",f"Per resolution\nβ = {current[2]['beta']:g}"]
        axis.set_xticks(range(3),labels);axis.set_xlim(-.5,2.5)
        axis.axhline(1.,color='#222222',linestyle='--',lw=1.)
        axis.set_ylim(*ylimits);axis.set_title(f'{resolution}² observations');style_axis(axis)
    axes[0].set_ylabel('Final-test rAUEC')
    fig.suptitle('Locked validation choices on the final 200-case population',fontsize=11)
    export(fig,folder,'test_beta_comparison',display,
        'Final-test rAUEC for the original beta0.5, the shared validation-selected beta, and the separately declared '
        'per-resolution validation-selected beta. All choices are read from the immutable calibration selection; no test-best '
        'beta is chosen. Error bars are sample SD across three sampling seeds of one frozen training model. '
        'The dashed line is matched Uniform=1. Repeated beta values in different policy views refer to the same evaluated '
        'condition, not extra independent experiments. The complete y range preserves means and error bars above1.')


def plot_paired(paired,folder):
    fig,axis=plt.subplots(figsize=(6.6,3.7),layout='constrained')
    for policy,offset,marker,label in [('shared',-.07,'o','Shared validation β'),
                                      ('per_resolution',.07,'^','Per-resolution validation β')]:
        current=sorted([r for r in paired if r['policy']==policy],key=lambda r:r['resolution'])
        axis.errorbar([i+offset for i in range(3)],[r['mean_delta'] for r in current],
            yerr=[r['paired_sampling_sd'] for r in current],color=COLORS[policy],marker=marker,
            markersize=6,capsize=4,lw=1.3,label=label)
    axis.axhline(0,color='#222222',ls='--',lw=1)
    axis.set_xticks(range(3),['85²','141²','211²']);axis.set_xlabel('Observation grid')
    axis.set_ylabel('Paired rAUEC change from β = 0.5')
    axis.set_title('Change from the original exponent');style_axis(axis);axis.legend(frameon=False)
    export(fig,folder,'paired_beta_change',paired,
        'For each sampling seed, subtract the beta0.5 rAUEC from the validation-selected beta rAUEC using the same '
        'case population, selected-point group and Uniform denominator. Points show the mean of the three differences; '
        'error bars show their paired sample SD (ddof=1). Negative change indicates improvement relative to beta0.5. '
        'It does not by itself imply that the new rAUEC is below Uniform=1; that is shown separately in test_beta_comparison. '
        'There is one frozen training model. No test-based policy choice or significance test is added.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--require-final',action='store_true');args=parser.parse_args()
    root=args.root.resolve();results=root/'results';folder=root/'figures'
    validation_path=results/'validation_rauec.csv';selection_path=root/'selection.json'
    if not validation_path.exists() or not selection_path.exists() or not(results/'validation_statistics.json').exists():
        require(not args.require_final,'Complete validation selection and final statistics are not available')
        print(json.dumps(dict(status='waiting_for_real_validation_results',figures_generated=0)));return
    selection=read_json(selection_path);validation_receipt=read_json(results/'validation_statistics.json')
    require(validation_receipt.get('passed') is True and validation_receipt['selection_sha256']==sha(selection_path),
            'Calibration receipt does not bind the current immutable selection')
    selection_before=selection_path.read_bytes();protocol_before=(root/'protocol.json').read_bytes()
    validation,validation_hash=csv_snapshot(validation_path);validation_index=table_index(validation,'validation')
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    plot_validation(validation,validation_index,selection,folder);plot_objective(selection,folder)
    inputs={'results/validation_rauec.csv':validation_hash,'selection.json':sha(selection_path),
            'protocol.json':sha(root/'protocol.json')}
    final_path=results/'test_rauec.csv';comparison_path=results/'selected_beta_comparison.csv'
    final_available=final_path.exists() and comparison_path.exists() and (results/'final_statistics.json').exists()
    paired_count=0
    if final_available:
        receipt=read_json(results/'final_statistics.json')
        require(receipt.get('passed') is True and receipt['selection_sha256']==sha(selection_path),
                'Final statistics do not bind the locked selection')
        final,final_hash=csv_snapshot(final_path);comparison,comparison_hash=csv_snapshot(comparison_path)
        final_index=table_index(final,'test')
        for resolution in RESOLUTIONS:
            actual={beta for r,method,beta in final_index if r==resolution and method=='pi'}
            require(actual==set(float(b) for b in selection['test_betas_by_resolution'][str(resolution)]),
                    'Final table includes missing or unselected beta values')
        display,paired=display_test(final_index,selection,comparison)
        for row in paired:
            row['selection_sha256']=sha(selection_path);row['test_rauec_csv_sha256']=final_hash
        write_csv(results/'paired_beta_differences.csv',paired);paired_count=len(paired)
        plot_test(display,folder);plot_paired(paired,folder)
        inputs.update({'results/test_rauec.csv':final_hash,'results/selected_beta_comparison.csv':comparison_hash})
    elif args.require_final:
        raise ValueError('Final test statistics are not yet available')
    require(selection_path.read_bytes()==selection_before and (root/'protocol.json').read_bytes()==protocol_before,
            'Plotting unexpectedly changed locked selection or protocol')
    (folder/'visual-contract.md').write_text('# PI beta calibration visuals\n\n'
        'Quantitative figures distinguish validation-only beta selection, final-test comparison, and paired changes from the original exponent. '
        'Calibration uses100 cases, final testing200 cases; all results come from one frozen training model and three sampling seeds. '
        'All predeclared candidates, including beta>1, remain in the calibration curve. Unfavorable final means and uncertainty bars above1 remain visible. '
        'Primary shared beta and secondary per-resolution beta keep distinct colors, markers and labels. The plots never choose a test-best beta. '
        'Source CSVs are exported beside PDF, editable SVG and300dpiPNG. Selection/protocol byte preservation and input hashes are in render_manifest.json.\n')
    manifest=dict(validation_figures_complete=True,final_figures_complete=final_available,paired_difference_rows=paired_count,
        input_sha256=inputs,plot_code_sha256=sha(Path(__file__)),
        statistics_code_sha256_at_selection=selection.get('statistics_code_sha256'),
        selection_unchanged=True,protocol_unchanged=True,training_models=1,sampling_seeds=3,
        unfavorable_values_preserved=True,test_best_beta_selected=False,
        render_qa='pending actual image inspection',
        files=[str(p.relative_to(root)) for p in sorted(folder.glob('*')) if p.suffix in {'.pdf','.svg','.png','.csv'}])
    (folder/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='figures_generated',validation_complete=True,final_complete=final_available,
                         paired_difference_rows=paired_count,actual_image_inspection_required=True)))


if __name__=='__main__':main()
