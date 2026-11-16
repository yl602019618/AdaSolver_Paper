#!/usr/bin/env python3
"""Integrate the approved Cl > .02 cohort and frozen PI policy into paper data."""
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
PAPER = ROOT / 'paper_git'
OUT = STUDY / 'paper_update'
DATA = PAPER / 'data/airfoil_pressure'
MAP = {
    'field': {'uniform': 'uniform', 'pi': 'pi_voronoi', 'cpg': 'cpg_candidate_11', 'ca': 'ca_candidate_10'},
    'cdcl': {'uniform': 'uniform', 'pi': 'pi_cdcl_tuned', 'cpg': 'cpg_cdcl_budget10', 'ca': 'ca_cdcl_budget10'},
}
METRIC = {'field': 'pressure_field_relative_l2', 'cdcl': 'ratio_abs_error'}


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def preserve(path):
    dest = OUT / 'preintegration' / path.relative_to(ROOT)
    if path.exists() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def csv_write(path, rows, fields=None):
    preserve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def csv_edit(path, edit):
    rows = list(csv.DictReader(path.open()))
    old = copy.deepcopy(rows)
    edit(rows)
    # Every unrelated benchmark must be byte-equivalent at the value level.
    assert [r for r in rows if r.get('benchmark') != 'airfoil2d'] == [r for r in old if r.get('benchmark') != 'airfoil2d']
    csv_write(path, rows)


def main():
    summary = read(STUDY / 'test_summary.json')
    raw = [json.loads(s) for s in (STUDY / 'test_raw.jsonl').read_text().splitlines()]
    full = [json.loads(s) for s in (ROOT / 'airfoil_lift_threshold/fullmesh.jsonl').read_text().splitlines()]
    full.sort(key=lambda r: r['sample_id'])
    ids = [r['sample_id'] for r in full]
    assert len(ids) == len(set(ids)) == 110
    assert all(r['truth']['Cl'] > .02 and r['status'] == 'ok' for r in full + raw)
    denominator = {m: float(np.mean([r[k] for r in full])) for m, k in METRIC.items()}
    for p in DATA.glob('*'):
        if p.is_file() and p.name != 'timing_raw.jsonl':
            preserve(p)
    # Preserve the old field cohort in the published data package as sensitivity history.
    archive = DATA / 'history/cl_gt_0p01'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ['test_records.csv', 'test_summary.json', 'configs.json', 'protocol.json', 'README.md']:
        source = OUT / 'preintegration/paper_git/data/airfoil_pressure' / name
        if source.exists() and not (archive / name).exists():
            shutil.copy2(source, archive / name)

    table_path = ROOT / 'plotting/data/standard_benchmarks.json'
    preserve(table_path)
    table = read(table_path)
    before = copy.deepcopy(table)
    table['record_accounting'] = 'Legacy process/count metadata applies to the original runs; Airfoil replacement records are specified per metric below.'
    for metric, mapping in MAP.items():
        col = next(c for c in table['columns'] if (c['benchmark'], c['metric']) == ('airfoil2d', metric))
        col.update(sample_count=110, sample_ids=ids, inclusion='reference pressure-force Cl > 0.02')
        ranks = {m: i+1 for i, m in enumerate(sorted(['pi', 'cpg', 'ca'], key=lambda m: summary[metric][mapping[m]]['rauec_mean']))}
        sources = {str(s): {} for s in [0, 1, 2]}
        compact = []
        for method, config in mapping.items():
            d = summary[metric][config]
            col['methods'][method] = dict(
                per_seed_rauec=[d['per_seed_rauec'][str(s)] for s in [0, 1, 2]],
                mean=d['rauec_mean'], std=d['rauec_std'], std_ddof=1,
                pooled_curve_rauec=d['pooled_rauec'],
                per_seed_mean_errors=[d['per_seed_mean_errors'][str(s)] for s in [0, 1, 2]],
                **({'rank': ranks[method]} if method != 'uniform' else {}))
            selected = [r for r in raw if r['method'] == config]
            assert len(selected) == 990
            for r in selected:
                compact.append(dict(method=method, sample_id=r['sample_id'], seed=r['seed'],
                    budget_fraction=r['budget_fraction'], selected_count=r['selected_count'],
                    **{METRIC[metric]: r[METRIC[metric]]}))
            for seed in [0, 1, 2]:
                rel = f'updates/airfoil_cl002/{metric}/seed_{seed}/{method}_raw.jsonl'
                p = ROOT / 'AdaSolver_official/results/standard_benchmarks_rauec' / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(''.join(json.dumps(r) + '\n' for r in selected if r['seed'] == seed))
                sources[str(seed)][method] = rel
        col['sources'] = sources
        col['policy_update'] = dict(selection='airfoil_pi_tuning/frozen_selection.json',
            cohort='Cl > 0.02 post-hoc threshold revision', config_mapping=mapping)
        prefix = '' if metric == 'field' else 'cdcl_'
        write(DATA / f'{prefix}test_summary.json', {m: summary[metric][c] for m, c in mapping.items()})
        write(DATA / f'{prefix}configs.json', {m: summary[metric][c]['config'] for m, c in mapping.items()})
        csv_write(DATA / f'{prefix}test_records.csv', compact)
    assert [c for c in table['columns'] if c['benchmark'] != 'airfoil2d'] == [c for c in before['columns'] if c['benchmark'] != 'airfoil2d']
    write(table_path, table)
    columns = {c['metric']: c for c in table['columns'] if c['benchmark'] == 'airfoil2d'}

    def update_table(rows):
        for r in rows:
            if r['benchmark'] != 'airfoil2d':
                continue
            d = columns[r['metric']]['methods'][r['method'].lower()]
            r.update(samples=110, mean=d['mean'], std_ddof1=d['std'], pooled_curve_rauec=d['pooled_curve_rauec'], rank=d['rank'])
            for i in [0, 1, 2]:
                r[f'seed_{i}_rauec'] = d['per_seed_rauec'][i]
    csv_edit(ROOT / 'plotting/data/standard_benchmarks.csv', update_table)

    def update_curves(rows):
        for r in rows:
            if r['benchmark'] != 'airfoil2d':
                continue
            metric = r['metric']
            d = summary[metric][MAP[metric][r['method']]]
            i = [37.5, 50., 75.].index(float(r['budget_percent']))
            mean, sd = d['mean'][i], d['std'][i]
            if 'normalized_mean' in r:
                den = denominator[metric]
                r.update(raw_mean_error=mean, raw_std_error=sd, fullmesh_mean_error=den,
                    normalized_mean=mean/den, normalized_std=sd/den, sample_count=110,
                    mean_at_or_below_fullmesh=mean <= den)
            else:
                r.update(mean=mean, std=sd)
    for f in ['fig3/data/curves.csv', 'fig3/data/normalized_curves.csv', 'paper_git/figures/appendix/normalized_curve_points.csv']:
        csv_edit(ROOT / f, update_curves)

    p = ROOT / 'fig3/data/fullmesh_references.json'
    preserve(p)
    refs = read(p)
    for c in refs['columns']:
        if c['benchmark'] == 'airfoil2d':
            c.update(sample_count=110, sample_ids=ids, fullmesh_mean_error=denominator[c['metric']],
                per_sample_errors=[r[METRIC[c['metric']]] for r in full])
            c['provenance']['inclusion'] = 'reference pressure-force Cl > 0.02'
    write(p, refs)
    def update_ref(rows):
        for r in rows:
            if r['benchmark'] == 'airfoil2d':
                r.update(sample_count=110, fullmesh_mean_error=denominator[r['metric']])
    csv_edit(ROOT / 'fig3/data/fullmesh_references.csv', update_ref)
    p = ROOT / 'fig3/data/fullmesh_replays/airfoil2d.json'
    preserve(p)
    replay = read(p)
    replay.update(sample_ids=ids, sample_count=110, records=[dict(sample_id=r['sample_id'],
        pressure_field_relative_l2=r['pressure_field_relative_l2'], ratio_abs_error=r['ratio_abs_error'],
        full_point_count=r['selected_count']) for r in full])
    replay['provenance']['inclusion'] = 'reference pressure-force Cl > 0.02'
    write(p, replay)
    csv_write(DATA / 'fullmesh_records.csv', [dict(sample_id=r['sample_id'], true_cl=r['truth']['Cl'],
        pressure_field_relative_l2=r['pressure_field_relative_l2'], ratio_abs_error=r['ratio_abs_error']) for r in full])

    baseline = ROOT / 'fig3/fullpoint_budget_comparison/data/fullpoint_budget_curves.csv'
    rows = list(csv.DictReader(baseline.open()))
    invalid = [r for r in rows if r['benchmark'] == 'airfoil2d']
    if invalid:
        assert len(invalid) == 8
        csv_write(DATA / 'history/invalid_mach_reference.csv', invalid)
    valid = [r for r in rows if r['benchmark'] != 'airfoil2d']
    assert len(valid) == 40
    csv_write(baseline, valid)
    csv_write(PAPER / 'figures/appendix/fullmesh_training_curve_points.csv', valid)
    audit = read(ROOT / 'airfoil_lift_threshold/reference_target_audit.json')
    audit['handling'] = 'Excluded from both Airfoil appendix panels: no matched full-mesh-trained pressure checkpoint is available.'
    write(DATA / 'reference_target_audit.json', audit)
    protocol = read(STUDY / 'protocol.json')
    protocol.update(paper_integration='Table 1 and appendix budget curves',
        threshold_history='Post-hoc change from true Cl > 0.01 (112 cases) to > 0.02 (110 cases).',
        sample_ids=ids, removed_case_ids=[1041, 1139], metric_mapping=MAP,
        field_selection_protocol='history/cl_gt_0p01/protocol.json',
        field_policies='Previously selected under Cl > 0.01; unchanged for the Cl > 0.02 test cohort.',
        fullmesh_denominators=denominator)
    write(DATA / 'protocol.json', protocol)
    write(DATA / 'pi_frozen_selection.json', read(STUDY / 'frozen_selection.json'))
    write(DATA / 'pi_confirmation_summary.json', read(STUDY / 'confirmation_summary.json'))
    csv_write(DATA / 'budget_summary.csv', [dict(metric=metric, method=method,
        budget_fraction=budget, mean=d['mean'][i], std=d['std'][i],
        normalized_mean=d['mean'][i]/denominator[metric], normalized_std=d['std'][i]/denominator[metric],
        fullmesh_mean_error=denominator[metric], samples=110, seeds=3)
        for metric, mapping in MAP.items() for method, cfg in mapping.items()
        for d in [summary[metric][cfg]] for i, budget in enumerate([.375, .5, .75])])
    p = ROOT / 'AdaSolver_official/configs/methods/airfoil2d_pi_cdcl_tuned.yaml'
    write(p, summary['cdcl']['pi_cdcl_tuned']['config'])

    table_tex = PAPER / 'tables/standard_benchmarks.tex'
    preserve(table_tex)
    lines = table_tex.read_text().splitlines()
    for i, line in enumerate(lines):
        if not any(line.strip().startswith(m.upper() + ' &') for m in ['pi', 'cpg', 'ca']):
            continue
        cells = line.split(' & ')
        method = cells[0].strip().lower()
        for j, metric in enumerate(['field', 'cdcl'], 1):
            d = columns[metric]['methods'][method]
            mean = f"{d['mean']:.4f}"
            if d['rank'] <= 2:
                mean = '\\' + ('mathbf' if d['rank'] == 1 else 'underline') + '{' + mean + '}'
            cells[j] = '$\\begin{gathered}' + mean + '\\\\[-2pt]\\pm ' + f"{d['std']:.4f}" + '\\end{gathered}$'
        lines[i] = ' & '.join(cells)
    table_tex.write_text('\n'.join(lines) + '\n')
    write(OUT / 'result_binding.json', dict(methods=MAP, sample_count=110, seeds=[0, 1, 2],
        budgets=[.375, .5, .75], fullmesh_denominators=denominator,
        adaptive_points=144, fullmesh_trained_points=40, excluded_airfoil_mach_points=8,
        other_benchmarks_unchanged=True, source='airfoil_pi_tuning/test_raw.jsonl'))
    print('Installed Table 1, 24 Airfoil curve points, both denominators and reproducible records.')
    print('CA field at 75%:', summary['field'][MAP['field']['ca']]['mean'][2] / denominator['field'])


if __name__ == '__main__':
    main()
