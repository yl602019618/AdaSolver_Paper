"""Independently assemble standard-task evidence from accepted raw CSVs.

Can publish an explicitly partial table after a complete preliminary cohort.
Never imports training modules, opens a GPU, or substitutes historical results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

from aggregate_rauec import aggregate, GROUP

BASE = Path(__file__).resolve().parents[1]
FOLLOWUP = BASE.parent
MODELS = {'darcy': ['transolver3', 'linearno', 'linearno_conv'],
          'airfoil': ['t3', 'linearno', 'linearno_conv']}
LABELS = {'transolver3': 'Transolver3', 't3': 'Transolver3',
          'linearno': 'LinearNO_point_temp', 'linearno_conv': 'LinearNO_conv_temp'}
METHODS = ['full', 'uniform', 'pi', 'cpg_beta0', 'cpg', 'ca']


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        # Clear our generated view so an earlier cohort cannot survive a newer
        # empty collection. Preserve its schema when one was already available.
        header = next(csv.reader(path.open()), []) if path.exists() else []
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            if header:
                csv.writer(f).writerow(header)
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_snapshot(path):
    payload = Path(path).read_bytes()
    return list(csv.DictReader(io.StringIO(payload.decode()))), hashlib.sha256(payload).hexdigest()


def cohort_status(dataset, architecture):
    folder = BASE / dataset
    if dataset == 'darcy':
        final = read(folder / 'records' / architecture / 'complete_completion.json')
        if final.get('execution_status') == 'completed':
            return True, True, final
        baseline = read(folder / 'records' / architecture / 'baseline_completion.json')
        return baseline.get('execution_status') == 'completed', False, baseline
    final = read(folder / 'results' / f'{architecture}_accuracy_acceptance.json')
    if final.get('passed'):
        return True, True, read(folder / 'results' / f'{architecture}_complete_completion.json')
    preliminary = read(folder / 'results' / f'{architecture}_preliminary_completion.json')
    return preliminary.get('execution_status') == 'completed', False, preliminary


def collect_accuracy():
    split = read(BASE / 'split_manifest.json')
    populations = {('darcy', 'historical_smooth2_200'): split['darcy']['test_ids'],
                   ('airfoil2d', 'published_cl_gt_0p02_110'): split['airfoil']['primary_test_ids']}
    field_rows, auxiliary, registry, sources, states = [], [], [], [], []
    for dataset, architectures in MODELS.items():
        for arch in architectures:
            folder = BASE / dataset
            ready, complete, receipt = cohort_status(dataset, arch)
            model = read(folder / 'checkpoints' / arch / 'manifest.json')
            state = dict(dataset=dataset, backbone=LABELS[arch],
                         expected_methods=['full'] if arch == 'linearno_conv' else METHODS,
                         accuracy_status='complete' if complete else 'preliminary' if ready else 'pending',
                         reason='' if ready else 'Training or complete preliminary evaluation pending')
            states.append(state)
            if not ready:
                continue
            if model.get('execution_status') != 'completed' or model.get('epochs_completed') != 1000:
                raise ValueError(('Evaluation lacks completed locked training', dataset, arch))
            path = Path(receipt['metrics_csv'])
            raw, source_sha = csv_snapshot(path)
            if source_sha != receipt.get('metrics_sha256', receipt.get('metrics_csv_sha256')):
                raise ValueError(('Immutable phase CSV does not match receipt', path))
            sources.append(dict(path=str(path), sha256=source_sha, rows=len(raw), dataset=dataset, architecture=arch))
            state['source_csv'] = str(path)
            checkpoint = model['checkpoint_sha256']
            normalizer = model['normalizer_sha256']
            reconstruction = read(folder / 'config' / ('evaluation_gate.json' if dataset == 'airfoil' else 'evaluation_lock.json'))
            expected_ids = {reconstruction[k] for k in (('old_id', 'new_id') if dataset == 'airfoil' else ('old_reconstruction_id', 'new_reconstruction_id'))}
            current = []
            for index, record in enumerate(raw, 2):
                row = dict(record)
                row['normalizer_id'] = row.get('normalizer_id') or row.get('normalizer_sha256')
                row['backbone'] = LABELS[arch]
                row['source_csv'] = str(path)
                row['source_line'] = index
                row['source_csv_sha256'] = source_sha
                if row['checkpoint_id'] != checkpoint or row['normalizer_id'] != normalizer:
                    raise ValueError(('CSV/checkpoint/normalizer mismatch', path, index))
                if row['training_seed'] != '0' or row['metric_target'] != 'field' or row['reconstruction_id'] not in expected_ids:
                    raise ValueError(('Unexpected cohort identity', path, index))
                if row['dataset'] != ('airfoil2d' if dataset == 'airfoil' else 'darcy'):
                    raise ValueError(('Incorrect task identity', path, index))
                if row['metric'] != 'field_relative_l2':
                    auxiliary.append(row)
                    continue
                if row['unit'] != 'ratio' or row['status'] != 'ok' or not np.isfinite(float(row['value'])) or float(row['value']) < 0:
                    raise ValueError(('Invalid field observation', path, index))
                current.append(row)
            actual_methods = {r['method'] for r in current}
            wanted = {'full'} if arch == 'linearno_conv' else set(METHODS if complete else METHODS[:-1])
            if actual_methods != wanted:
                raise ValueError(('Method matrix differs from completion state', path, actual_methods, wanted))
            population_key = (current[0]['dataset'], current[0]['test_population'])
            ids = populations[population_key]
            expected = {(str(sid), method, '' if method == 'full' else str(seed), 1. if method == 'full' else budget, rid)
                        for sid in ids for method in wanted
                        for seed in ([None] if method == 'full' else [0, 1, 2])
                        for budget in ([1.] if method == 'full' else [.375, .5, .75]) for rid in expected_ids}
            keyed = {(r['case_id'], r['method'], r['sampling_seed'], float(r['budget_fraction']), r['reconstruction_id']): r for r in current}
            if len(keyed) != len(current) or set(keyed) != expected:
                raise ValueError(('Raw field matrix incomplete or duplicated', path, len(current), len(expected)))
            for key, row in keyed.items():
                beta = None if row['method'] in ('full', 'uniform') else .5 if row['method'] == 'pi' and dataset == 'darcy' else 0. if row['method'] == 'cpg_beta0' else 1.
                if (None if row['beta'] == '' else float(row['beta'])) != beta:
                    raise ValueError(('Incorrect locked beta', path, key))
                if row['method'] == 'cpg':
                    zero = keyed[(key[0], 'cpg_beta0', *key[2:])]
                    for column in ['selected_indices_hash', 'coarse_indices_hash', 'coarse_prediction_hash']:
                        if not row.get(column):
                            raise ValueError(('Missing CPG identity', path, key, column))
                        if row[column] != zero[column]:
                            raise ValueError(('CPG beta-pair point identity mismatch', path, key, column))
            state.update(actual_field_rows=len(current), expected_field_rows=len(expected), population_cases=len(ids))
            field_rows.extend(current)
            registry.append(dict(dataset=current[0]['dataset'], backbone=LABELS[arch], checkpoint_id=checkpoint,
                normalizer_id=normalizer, training_seed=0, training_cases=900, validation_cases=100, epochs=1000,
                updates=113000 if dataset == 'airfoil' else 225000, parameters=model['parameter_count'],
                training_seconds=model.get('training_seconds', model.get('training_wall_seconds')),
                best_validation=model['best_validation_relative_l2'], model_class=model.get('native_model_class', model.get('model_class')),
                training_points='full grid' if dataset == 'darcy' or arch == 'linearno_conv' else 'shared uniform K in [4508,11271]',
                checkpoint_path=model['checkpoint'], accuracy_status=state['accuracy_status'],
                model_manifest=str(folder / 'checkpoints' / arch / 'manifest.json')))
    return field_rows, auxiliary, registry, sources, states, populations


def absolute_tables(records, curves, registry):
    groups = defaultdict(list)
    for row in curves:
        groups[tuple(row[k] for k in GROUP) + (row['method'], row['beta'], float(row['budget_fraction']))].append(row)
    full_groups = defaultdict(list)
    for row in records:
        if row['method'] == 'full':
            full_groups[tuple(str(row[k]) for k in GROUP)].append(row)
    models = {(r['dataset'], r['backbone'], r['checkpoint_id']): r for r in registry}
    output = []
    for key, raw in full_groups.items():
        identity = dict(zip(GROUP, key))
        full = float(np.mean([float(r['value']) for r in raw]))
        model = models[identity['dataset'], identity['backbone'], identity['checkpoint_id']]
        output.append({**identity, 'method': 'full', 'beta': '', 'budget_fraction': 1., 'cases': len(raw),
                       'sampling_seeds': 0, 'mean_error': full, 'sampling_sd': '', 'same_checkpoint_full_error': full,
                       'parameters': model['parameters'], 'training_epochs': model['epochs'], 'training_updates': model['updates']})
    for key, rows in groups.items():
        identity = dict(zip(GROUP, key[:len(GROUP)]))
        values = [float(r['mean_error']) for r in rows]
        assert {int(r['sampling_seed']) for r in rows} == {0, 1, 2}
        full = float(np.mean([float(r['value']) for r in full_groups[key[:len(GROUP)]]]))
        model = models[identity['dataset'], identity['backbone'], identity['checkpoint_id']]
        output.append({**identity, 'method': key[-3], 'beta': key[-2], 'budget_fraction': key[-1],
                       'cases': rows[0]['cases'], 'sampling_seeds': 3, 'mean_error': float(np.mean(values)),
                       'sampling_sd': float(np.std(values, ddof=1)), 'same_checkpoint_full_error': full,
                       'parameters': model['parameters'], 'training_epochs': model['epochs'], 'training_updates': model['updates']})
    return output


def compare_owner_tables(independent):
    checks = []
    for dataset in MODELS:
        path = BASE / dataset / 'results/backbone_rauec.csv'
        if not path.exists():
            continue
        rows, digest = csv_snapshot(path)
        for row in rows:
            if row['metric'] != 'field_relative_l2':
                continue
            backbone = row['backbone'].replace('_fullgrid', '')
            match = [r for r in independent if all(str(r[k]) == str(row[k]) for k in
                     ('dataset', 'checkpoint_id', 'training_seed', 'metric_target', 'test_population', 'reconstruction_id', 'metric', 'method'))
                     and r['backbone'] == backbone]
            if not match:
                continue  # A newly completed cohort may appear after our atomic raw snapshot.
            if len(match) != 1:
                raise ValueError(('Ambiguous independent match', path, row))
            reference = match[0]
            actual = [float(row.get('mean', row.get('mean_rauec'))), float(row['sampling_sd'])]
            wanted = [reference['mean'], reference['sampling_sd']]
            np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14)
            for seed in range(3):
                np.testing.assert_allclose(float(row[f'seed{seed}']), reference[f'seed{seed}_rauec'], rtol=1e-12, atol=1e-14)
            checks.append(dict(dataset=row['dataset'], backbone=backbone, method=row['method'],
                reconstruction_id=row['reconstruction_id'], source=str(path), source_sha256=digest,
                mean_absolute_difference=abs(actual[0] - wanted[0]), sd_absolute_difference=abs(actual[1] - wanted[1])))
    return checks


def collect_latency(registry):
    results, receipts = [], []
    models = {(r['dataset'], r['backbone']): r for r in registry}
    for dataset in MODELS:
        folder = BASE / dataset
        receipt = read(folder / 'results/latency_completion.json')
        if receipt.get('execution_status') != 'completed':
            continue
        path = folder / 'results/backbone_latency_summary.csv'
        rows, digest = csv_snapshot(path)
        if len(rows) != 1323:
            raise ValueError(('Latency condition count differs', path, len(rows)))
        for row in rows:
            row['backbone'] = row['backbone'].replace('_fullgrid', '')
            model = models[row['dataset'], row['backbone']]
            if row['checkpoint_id'] != model['checkpoint_id'] or row['normalizer_id'] != model['normalizer_id']:
                raise ValueError(('Accuracy/timing model mismatch', path))
            if int(row['formal_repeats']) != 100 or int(row['warmups']) != 20 or row['output_dtype'] != 'float64':
                raise ValueError(('Timing contract mismatch', path))
            row['condition_source_csv'] = str(path)
            row['condition_source_csv_sha256'] = digest
            results.append(row)
        receipts.append(dict(dataset=dataset, source=str(path), sha256=digest, conditions=len(rows)))
    return results, receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    rows, auxiliary, registry, sources, states, populations = collect_accuracy()
    destination = BASE / 'results'
    destination.mkdir(exist_ok=True)
    complete = all(r['accuracy_status'] == 'complete' for r in states)
    if args.require_complete and not complete:
        raise RuntimeError('New-task accuracy matrices are not all complete')
    areas, curves, _ = aggregate(rows, populations)
    absolute = absolute_tables(rows, curves, registry)
    checks = compare_owner_tables(areas)
    if args.require_complete and len(checks) != len(areas):
        raise RuntimeError('Independent rAUEC rows do not all have matching owner summaries')
    latency, latency_sources = collect_latency(registry)
    if args.require_complete and len(latency) != 2646:
        raise RuntimeError('Both new-task latency matrices are not complete')
    for name, values in [('field_records.csv', rows), ('auxiliary_metric_records.csv', auxiliary),
                         ('checkpoint_registry.csv', registry), ('backbone_rauec.csv', areas),
                         ('backbone_budget_metrics.csv', absolute), ('backbone_absolute_errors.csv', absolute),
                         ('seed_budget_errors.csv', curves), ('backbone_latency_summary.csv', latency)]:
        write_csv(destination / name, values)
    write_json(destination / 'independent_statistics_check.json', dict(
        passed=bool(rows), status='checked_available_complete_cohorts' if rows else 'waiting_for_real_cohort',
        validation_scope='Explicit locked populations; exact raw matrix; identity and beta checks; independent seed-first trapezoid and ddof=1 SD',
        accuracy_complete=complete, field_rows=len(rows), auxiliary_rows_preserved=len(auxiliary),
        method_area_rows=len(areas), owner_comparisons=checks, sources=sources,
        split_manifest_sha256=sha(BASE / 'split_manifest.json'), collector_sha256=sha(__file__),
        aggregator_sha256=sha(Path(__file__).with_name('aggregate_rauec.py'))))
    write_json(destination / 'collection_status.json', dict(
        accuracy_complete=complete, latency_complete=len(latency) == 2646, models=states,
        field_rows=len(rows), rauec_rows=len(areas), latency_conditions=len(latency),
        latency_sources=latency_sources, source_records=sources))
    print(json.dumps(dict(accuracy_complete=complete, field_rows=len(rows), rauec_rows=len(areas),
                         owner_comparisons=len(checks), latency_conditions=len(latency))))


if __name__ == '__main__':
    main()
