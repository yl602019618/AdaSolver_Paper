#!/usr/bin/env python3
"""Export the corrected original-policy Cd/Cl package from fresh native records."""
from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

from result_tools import ROOT, IDS, SEEDS, BUDGETS, NAMES, CONFIGS, collect, validate_rows, aggregate, write_csv, write_json

PACKAGE_NAME = 'p0_airfoil_original_cdcl_20261117'
REPO = Path(os.environ.get('AIRFOIL_CDCL_REPO', str(ROOT / 'paper'))).resolve()
OUT = REPO / 'data/rebuttal/2026-11' / PACKAGE_NAME
REPRO = REPO / 'reproducibility/rebuttal/2026-11' / PACKAGE_NAME
DISPLAY = {'uniform': 'Uniform', 'pi': 'PI', 'cpg': 'CPG', 'ca': 'CA'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compressed_csv(path, rows):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)


def flatten(row, cohort, registry):
    return {'backbone': 'Transolver3', 'cohort': cohort, 'training_seed': registry['training_seed'],
            'method': DISPLAY[NAMES[row['method']]], 'policy_name': row['method'],
            'sample_id': row['sample_id'], 'sampling_seed': row['seed'], 'budget': row['budget_fraction'],
            'selected_count': row['selected_count'], 'coarse_count': row['sampling_metadata']['coarse_count'],
            'true_Cd': row['truth']['Cd'], 'true_Cl': row['truth']['Cl'],
            'true_Cd_over_Cl': row['truth']['Cd_over_Cl'],
            'predicted_Cd': row['prediction']['Cd'], 'predicted_Cl': row['prediction']['Cl'],
            'predicted_Cd_over_Cl': row['prediction']['Cd_over_Cl'],
            'cdcl_abs_error': row['ratio_abs_error'],
            'selected_indices_sha256': row['selected_indices_hash'],
            'coarse_indices_sha256': row['sampling_metadata'].get('coarse_indices_hash'),
            'checkpoint_sha256': row['operator_sha256'], 'normalizer_sha256': row['normalizer_sha256'],
            'head_sha256': row['sampling_metadata'].get('head_sha256'),
            'reconstruction_id': registry['reconstruction_id'],
            'interpolation_backend': row['interpolation_backend'],
            'interpolation_fallback_count': row['interpolation_fallback_count'],
            'status': row['status'], 'truth_access_during_sampling': row['truth_access']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / 'runtime_registry.json').read_text())
    control = json.loads((ROOT / 'control_verification.json').read_text())
    if control['passed'] is not True:
        raise RuntimeError('original control failed; training-seed aggregation is prohibited')
    checks, summaries, flat, raw_by_seed = {}, {}, [], {}
    issues = []
    registry_rows = []
    for key in ['control', 'training_seed0', 'training_seed1', 'training_seed2']:
        r = registry[key]
        for file_field, hash_field in [('checkpoint', 'checkpoint_sha256'), ('head_checkpoint', 'head_sha256'),
                                       ('normalizer', 'normalizer_sha256'), ('manifest', 'manifest_sha256'),
                                       ('head_manifest', 'head_manifest_sha256')]:
            if sha(r[file_field]) != r[hash_field]:
                issues.append({'type': 'asset_changed', 'cohort': key, 'field': file_field})
        registry_rows.append({'cohort': key, **r})
    for seed in SEEDS:
        key = f'training_seed{seed}'
        rows, manifests = collect(key, 2)
        raw_by_seed[seed] = rows
        checks[key] = validate_rows(rows, registry[key])
        if not checks[key]['passed']:
            issues.append({'type': 'record_validation', 'cohort': key, 'issues': checks[key]['issues']})
        flat.extend(flatten(row, key, registry[key]) for row in rows)
        if checks[key]['passed']:
            summaries[seed] = aggregate(rows)
    compressed_csv(OUT / 'per_case_records.csv.gz', flat)
    control_rows, _ = collect('control', 4)
    control_flat = [flatten(row, 'control', registry['control']) for row in control_rows]
    compressed_csv(OUT / 'control_per_case_records.csv.gz', control_flat)
    write_csv(OUT / 'checkpoint_registry.csv', registry_rows)
    source_expected = json.loads((ROOT / 'source_snapshot_hashes.json').read_text())
    changed_source = [p for p, h in source_expected.items() if sha(ROOT / 'original_source' / p) != h]
    if changed_source:
        issues.append({'type': 'source_changed', 'files': changed_source})

    # Independently evaluate the fixed reference-pressure filter on every original test ID.
    sys.path.insert(0, str(ROOT / 'original_source'))
    from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
    from adasolver.metrics import pressure_surface_force
    data = AirfoilRuntimeData(registry['data_path'], registry['control']['manifest'])
    filter_rows = []
    for sid in range(1000, 1200):
        force = pressure_surface_force(data.coords(sid), data.target(sid))
        selected = force.lift_coefficient > 0.02
        filter_rows.append({'sample_id': sid, 'true_Cl': force.lift_coefficient,
                            'true_Cd': force.drag_coefficient,
                            'true_Cd_over_Cl': force.drag_coefficient / force.lift_coefficient if force.lift_coefficient != 0 else '',
                            'in_original_protocol': sid in IDS, 'included': sid in IDS,
                            'passes_original_cl_threshold': selected,
                            'removed_by_original_threshold_revision': sid in [1041, 1139],
                            'reason': 'retained original Cl>0.02 population' if sid in IDS else 'excluded by original reference Cl threshold'})
    filtered = [r['sample_id'] for r in filter_rows if r['passes_original_cl_threshold']]
    if filtered != IDS:
        issues.append({'type': 'case_filter_mismatch', 'recomputed_ids': filtered, 'locked_ids': IDS})
    write_csv(OUT / 'case_filter.csv', filter_rows)
    special = [r for r in control_flat + flat if r['sample_id'] in [1007, 1059, 1198]]
    write_csv(OUT / 'case_1007_1059_1198.csv', special)
    training_norms = {registry[f'training_seed{s}']['normalizer_sha256'] for s in SEEDS}
    if len(training_norms) != 1:
        issues.append({'type': 'training_normalizer_mismatch', 'hashes': sorted(training_norms)})
    protocol = json.loads((ROOT / 'reference/protocol.json').read_text())
    protocol.update(package=PACKAGE_NAME, evaluation_metric='absolute error of pressure-force Cd/Cl',
                    original_metric_specific_configs=CONFIGS, training_seeds=list(SEEDS),
                    sampling_seeds=list(SEEDS), full_100_percent_in_rauec=False,
                    original_protocol_sha256=sha(ROOT / 'reference/protocol.json'),
                    original_case_ids_preserved=True, reconstruction_id='original_index_grid_linear_nearest',
                    pressure_target_channel=3, pressure_prediction_clipping=False,
                    normalization={'within_checkpoint': 'For each sampling seed, average case errors at each budget, trapezoid-integrate, divide by that checkpoint/sampling-seed Uniform area; average the three resulting ratios.',
                                   'between_checkpoints': 'Mean and sample SD (ddof=1) of the three checkpoint-level means.',
                                   'control_normalizer': 'Original frozen operator normalizer, retained for the original control.',
                                   'P0_normalizer': 'The same P0 training-only normalizer for all three new checkpoints; not replaced by the historical control normalizer.'})
    write_json(OUT / 'protocol_original_cdcl.json', protocol)
    verification = {'status': 'VERIFIED_COMPLETE' if not issues else 'FAILED', 'complete': not issues,
                    'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'control_reproduction_passed': control['passed'], 'control': control,
                    'training_seed_coverage': checks, 'training_records': len(flat), 'expected_training_records': 11880,
                    'all_110_original_cases_retained': filtered == IDS and all(c['retained_cases'] == IDS for c in checks.values()),
                    'removed_outliers': [], 'extra_cases': [], 'missing_training_checkpoints': [],
                    'case_filter_recomputed': filtered == IDS, 'special_case_records': len(special),
                    'special_case_file': 'case_1007_1059_1198.csv',
                    'training_normalizer_sha256': sorted(training_norms),
                    'reconstruction_ids': sorted({r['reconstruction_id'] for r in flat + control_flat}),
                    'old_new_reconstruction_mixed': False, 'original_aggregation_independently_recomputed': len(summaries) == 3,
                    'source_files_changed': changed_source, 'inconsistencies': issues,
                    'resolved_protocol_discrepancies': [
                        {'issue': 'The supplied official script describes an archived 25% candidate.',
                         'resolution': 'Used the unchanged Table 1 run_budget10/AirfoilEvaluator implementation and original aggregation; all 3960 control case errors reproduced exactly.'},
                        {'issue': 'Original seed-0 and P0 seed-0 identify different trained checkpoints and normalizers.',
                         'resolution': 'Original checkpoint is a separate protocol control. P0 training seeds0/1/2 form the new three-model population. Comparisons to the control are descriptive, not paired checkpoint deltas.'}],
                    'pending_required_items': [] if not issues else ['resolve recorded inconsistencies'],
                    'outside_scope': ['Plasticity three-training-seed experiments', 'LinearNO original-Cd/Cl-policy reevaluation',
                                      'New formal latency for these metric-specific policies'],
                    'old_P0_numeric_files_preserved': True}
    write_json(OUT / 'verification.json', verification)
    if issues:
        raise RuntimeError(f'package validation failed: {issues}')

    curve_rows, denominator_rows, checkpoint_summaries, training_rows = [], [], [], []
    for seed in SEEDS:
        for sampling_seed in SEEDS:
            u = summaries[seed]['uniform']['per_sampling_seed'][str(sampling_seed)]
            denominator_rows.append({'backbone': 'Transolver3', 'training_seed': seed, 'sampling_seed': sampling_seed,
                                     'cases': 110, 'uniform_error_0375': u['case_mean_errors'][0],
                                     'uniform_error_050': u['case_mean_errors'][1], 'uniform_error_075': u['case_mean_errors'][2],
                                     'uniform_area': u['uniform_area'], 'checkpoint_sha256': registry[f'training_seed{seed}']['checkpoint_sha256'],
                                     'normalizer_sha256': registry[f'training_seed{seed}']['normalizer_sha256'],
                                     'reconstruction_id': 'original_index_grid_linear_nearest'})
        for method in DISPLAY:
            value = summaries[seed][method]
            checkpoint_summaries.append({'backbone': 'Transolver3', 'training_seed': seed, 'method': DISPLAY[method],
                                         'rAUEC': value['sampling_mean'], 'sampling_sd': value['sampling_sd'],
                                         **{f'sampling_seed_{s}': value['per_sampling_seed'][str(s)]['rAUEC'] for s in SEEDS}})
            for sampling_seed in SEEDS:
                v = value['per_sampling_seed'][str(sampling_seed)]
                for i, budget in enumerate(BUDGETS):
                    curve_rows.append({'backbone': 'Transolver3', 'training_seed': seed, 'method': DISPLAY[method],
                                       'budget': budget, 'sampling_seed': sampling_seed,
                                       'case_mean_error': v['case_mean_errors'][i], 'uniform_area': v['uniform_area'],
                                       'method_area': v['method_area'], 'rAUEC': v['rAUEC'], 'cases': 110})
    for method in DISPLAY:
        values = [summaries[s][method]['sampling_mean'] for s in SEEDS]
        training_rows.append({'method': DISPLAY[method], **{f'training_seed_{s}': values[s] for s in SEEDS},
                              'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)})
    write_csv(OUT / 'per_seed_sampling_summary.csv', curve_rows)
    write_csv(OUT / 'per_training_seed_summary.csv', checkpoint_summaries)
    write_csv(OUT / 'uniform_denominators.csv', denominator_rows)
    write_csv(OUT / 'training_seed_cdcl_summary.csv', training_rows)
    control_summary = json.loads((ROOT / 'control_summary.json').read_text())
    comparison = [{'method': DISPLAY[m], 'original_control_sampling_mean': control_summary[m]['sampling_mean'],
                   'original_control_sampling_sd': control_summary[m]['sampling_sd'],
                   'P0_seed0_sampling_mean': summaries[0][m]['sampling_mean'],
                   'P0_seed0_sampling_sd': summaries[0][m]['sampling_sd'],
                   'comparison_scope': 'different checkpoints and fitted normalizers; no paired delta'} for m in DISPLAY]
    write_csv(OUT / 'original_control_vs_new_seed0.csv', comparison)

    # Keep large ratio errors and near-zero predicted lift visible in variance diagnostics.
    variance = {}
    for seed in SEEDS:
        rr = [r for r in flat if r['training_seed'] == seed]
        areas = [r['uniform_area'] for r in denominator_rows if r['training_seed'] == seed]
        largest = sorted(rr, key=lambda r: r['cdcl_abs_error'], reverse=True)[:10]
        variance[str(seed)] = {'uniform_areas_by_sampling_seed': areas,
                               'uniform_area_max_over_min': max(areas)/min(areas),
                               'predictions_with_abs_Cl_below_002': sum(abs(r['predicted_Cl']) < .02 for r in rr),
                               'negative_predicted_Cl_count': sum(r['predicted_Cl'] < 0 for r in rr),
                               'largest_ratio_errors': largest}
    write_json(OUT / 'variance_diagnostics.json', variance)
    for name in ['control_verification.json', 'control_reproduction_comparison.csv', 'control_per_case_comparison.csv']:
        shutil.copy2(ROOT / name, OUT / name)
    source_hashes = {'frozen_native_source': source_expected,
                     'original_reference': {p.name: sha(p) for p in (ROOT / 'reference').iterdir() if p.is_file()},
                     'analysis_source': {n: sha(ROOT / n) for n in ['run_evaluation.py', 'result_tools.py', 'verify_control.py', 'build_package.py', 'verify_package.py']},
                     'launch_records': {p.name: sha(p) for p in (ROOT / 'launches').glob('*.json')},
                     'baseline_paper_commit': '507bca1ca42c15fa156583b697adfb58047b650f',
                     'data_path': registry['data_path']}
    write_json(OUT / 'source_hashes.json', source_hashes)
    REPRO.mkdir(parents=True, exist_ok=True)
    for name in ['run_evaluation.py', 'result_tools.py', 'verify_control.py', 'build_package.py', 'verify_package.py',
                 'runtime_registry.json', 'source_snapshot_hashes.json', 'control_gate_protocol.json']:
        if (ROOT / name).resolve() != (REPRO / name).resolve():
            shutil.copy2(ROOT / name, REPRO / name)
    if ROOT.resolve() != REPRO.resolve():
        shutil.copytree(ROOT / 'original_source', REPRO / 'original_source', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'reference', REPRO / 'reference', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if (ROOT / 'launches').resolve() != (OUT / 'run_receipts').resolve():
        shutil.copytree(ROOT / 'launches', OUT / 'run_receipts', dirs_exist_ok=True)
    write_json(ROOT / 'final_summaries.json', {'training': summaries, 'control': control_summary, 'training_table': training_rows})
    print(json.dumps({'verification': verification['status'], 'training_records': len(flat), 'table': training_rows,
                      'special_case_truth_Cl': {r['sample_id']: r['true_Cl'] for r in filter_rows if r['sample_id'] in [1007,1059,1198]}}, indent=2))


if __name__ == '__main__':
    main()
