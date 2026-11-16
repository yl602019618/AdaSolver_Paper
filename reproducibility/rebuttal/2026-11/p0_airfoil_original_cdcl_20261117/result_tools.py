"""Coverage checks and original paired-Uniform aggregation for Airfoil Cd/Cl."""
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
BUDGETS = (0.375, 0.5, 0.75)
SEEDS = (0, 1, 2)
CONFIGS = json.loads((ROOT / 'reference/cdcl_configs.json').read_text())
NAMES = {cfg['name']: method for method, cfg in CONFIGS.items()}
IDS = json.loads((ROOT / 'reference/protocol.json').read_text())['sample_ids']


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def area(curve):
    return sum((BUDGETS[i+1] - BUDGETS[i]) * (curve[i] + curve[i+1]) / 2 for i in range(2))


def collect(cohort, shards):
    rows, manifests = [], []
    for shard in range(shards):
        path = ROOT / 'runs' / cohort / f'shard{shard}'
        m = json.loads((path / 'run_manifest.json').read_text())
        manifests.append(m)
        if m['status'] != 'complete':
            raise RuntimeError(f'{path} is incomplete')
        if m['sample_ids'] != IDS[shard::shards]:
            raise RuntimeError(f'{path} case list changed')
        for name in NAMES:
            for line in (path / f'{name}_raw.jsonl').open():
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (NAMES[r['method']], r['seed'], r['budget_fraction'], r['sample_id']))
    return rows, manifests


def validate_rows(rows, registry):
    expected = set(itertools.product(NAMES, IDS, SEEDS, BUDGETS))
    keys = [(r['method'], r['sample_id'], r['seed'], r['budget_fraction']) for r in rows]
    actual = set(keys)
    issues = []
    metrics = ('ratio_abs_error', 'pressure_field_relative_l2')
    invalid, binding, policies, thresholds = [], [], [], []
    for row in rows:
        key = (row['method'], row['sample_id'], row['seed'], row['budget_fraction'])
        values = [row.get(k) for k in metrics]
        values += [row[role].get(k) for role in ['truth', 'prediction'] for k in ['Cd', 'Cl', 'Cd_over_Cl']]
        if row['status'] != 'ok' or any(x is None or not math.isfinite(x) for x in values):
            invalid.append(key)
        if row['operator_sha256'] != registry['checkpoint_sha256'] or row['normalizer_sha256'] != registry['normalizer_sha256']:
            binding.append(key)
        if row['interpolation_backend'] != 'index_grid_linear_nearest':
            binding.append(key)
        if row['truth']['Cl'] <= .02 or row['sample_id'] in [1041, 1139] or row['truth_access'] is not False:
            thresholds.append(key)
        method = NAMES.get(row['method'])
        if method is not None:
            got = dict(row['resolved_method_config'])
            wanted = dict(CONFIGS[method])
            for k in ['head_checkpoint', 'head_manifest']:
                got.pop(k, None); wanted.pop(k, None)
            if got != wanted:
                policies.append(key)
            head_hash = row['sampling_metadata'].get('head_sha256')
            if (method == 'ca' and head_hash != registry['head_sha256']) or (method != 'ca' and head_hash is not None):
                binding.append(key)
    counts = {'expected_records': len(expected), 'actual_records': len(rows),
              'unique_records': len(actual), 'missing': sorted(expected-actual),
              'unexpected': sorted(actual-expected), 'duplicate_count': len(keys)-len(actual),
              'invalid_metrics': invalid, 'binding_mismatches': binding,
              'policy_mismatches': policies, 'filter_or_truth_access_mismatches': thresholds,
              'retained_cases': sorted({r['sample_id'] for r in rows}),
              'reconstruction_backends': sorted({r['interpolation_backend'] for r in rows})}
    for field in ['missing', 'unexpected', 'duplicate_count', 'invalid_metrics', 'binding_mismatches',
                  'policy_mismatches', 'filter_or_truth_access_mismatches']:
        if counts[field]:
            issues.append(field)
    counts['passed'] = not issues and counts['actual_records'] == 3960
    counts['issues'] = issues
    return counts


def aggregate(rows):
    curves = {}
    for name in NAMES:
        for seed in SEEDS:
            curve = [statistics.mean(r['ratio_abs_error'] for r in rows
                                    if r['method'] == name and r['seed'] == seed and r['budget_fraction'] == budget)
                     for budget in BUDGETS]
            curves[name, seed] = curve
    result = {}
    for name, method in NAMES.items():
        per_seed = {}
        for seed in SEEDS:
            den = area(curves['uniform', seed])
            if not math.isfinite(den) or den <= 0:
                raise RuntimeError(f'invalid Uniform area for sampling seed {seed}')
            per_seed[str(seed)] = {'case_mean_errors': curves[name, seed],
                                   'uniform_area': den, 'method_area': area(curves[name, seed]),
                                   'rAUEC': area(curves[name, seed]) / den}
        values = [per_seed[str(s)]['rAUEC'] for s in SEEDS]
        result[method] = {'policy_name': name, 'per_sampling_seed': per_seed,
                          'sampling_mean': statistics.mean(values),
                          'sampling_sd': statistics.stdev(values)}
    # Independently reuse the original Table 1 aggregator and compare all means/SDs.
    spec = importlib.util.spec_from_file_location('original_table1_aggregation', ROOT / 'reference/original_table1_aggregation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.summarize(rows, 'ratio_abs_error')
    for name, method in NAMES.items():
        assert math.isclose(result[method]['sampling_mean'], original[name]['rauec_mean'], abs_tol=1e-12)
        assert math.isclose(result[method]['sampling_sd'], original[name]['rauec_std'], abs_tol=1e-12)
    return result
