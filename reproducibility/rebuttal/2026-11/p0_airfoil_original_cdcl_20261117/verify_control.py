#!/usr/bin/env python3
"""Verify fresh original-checkpoint inference before opening the training-seed gate."""
import csv
import datetime
import json

from result_tools import ROOT, NAMES, collect, validate_rows, aggregate, write_json, write_csv

rows, manifests = collect('control', 4)
registry = json.loads((ROOT / 'runtime_registry.json').read_text())['control']
gate = json.loads((ROOT / 'control_gate_protocol.json').read_text())
coverage = validate_rows(rows, registry)
reference_rows = list(csv.DictReader((ROOT / 'reference/cdcl_test_records.csv').open()))
reference = {(r['method'], int(r['sample_id']), int(r['seed']), float(r['budget_fraction'])): r for r in reference_rows}
comparison, differences = [], []
passed = coverage['passed']
summary = aggregate(rows) if passed else None
if summary:
    for method, value in summary.items():
        expected = gate['expected'][method]
        mean_diff = value['sampling_mean'] - expected['rauec_mean']
        sd_diff = value['sampling_sd'] - expected['rauec_std']
        seed_diffs = {s: p['rAUEC'] - expected['per_seed_rauec'][s] for s, p in value['per_sampling_seed'].items()}
        ok = (abs(mean_diff) <= gate['mean_sd_absolute_tolerance'] and
              abs(sd_diff) <= gate['mean_sd_absolute_tolerance'] and
              max(map(abs, seed_diffs.values())) <= gate['per_sampling_seed_absolute_tolerance'])
        passed &= ok
        comparison.append({'method': method, 'original_sampling_mean': expected['rauec_mean'],
                           'reproduced_sampling_mean': value['sampling_mean'], 'mean_difference': mean_diff,
                           'original_sampling_sd': expected['rauec_std'], 'reproduced_sampling_sd': value['sampling_sd'],
                           'sd_difference': sd_diff, 'max_sampling_seed_difference': max(map(abs, seed_diffs.values())),
                           'passed': ok})
    for row in rows:
        key = (NAMES[row['method']], row['sample_id'], row['seed'], row['budget_fraction'])
        old = float(reference[key]['ratio_abs_error'])
        differences.append({'method': key[0], 'sample_id': key[1], 'sampling_seed': key[2], 'budget': key[3],
                            'original_error': old, 'reproduced_error': row['ratio_abs_error'],
                            'difference': row['ratio_abs_error'] - old})
    write_csv(ROOT / 'control_reproduction_comparison.csv', comparison)
    write_csv(ROOT / 'control_per_case_comparison.csv', differences)
    write_json(ROOT / 'control_summary.json', summary)
verification = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'passed': bool(passed), 'coverage': coverage, 'comparison': comparison,
                'max_absolute_per_case_error_difference': max((abs(r['difference']) for r in differences), default=None),
                'tolerance_predeclared': gate, 'fresh_inference': True,
                'control_checkpoint_sha256': registry['checkpoint_sha256'],
                'original_aggregation_independently_recomputed': summary is not None,
                'entrypoint_resolution': gate['entrypoint_resolution'],
                'not_the_P0_training_seed0_checkpoint': True}
write_json(ROOT / 'control_verification.json', verification)
print(json.dumps(verification, indent=2))
raise SystemExit(0 if passed else 2)
