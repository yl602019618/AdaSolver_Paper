"""Recompute this release's tables from compressed raw records; no GPU needed."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'E3_standard/code'))
from aggregate_rauec import aggregate, GROUP


def load_csv(path):
    payload = path.read_bytes()
    if path.suffix == '.gz':
        payload = gzip.decompress(payload)
    return list(csv.DictReader(io.StringIO(payload.decode()))), hashlib.sha256(payload).hexdigest()


def write_csv(path, records):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, list(records[0])); w.writeheader(); w.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'recomputed')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'snapshot_manifest.json').read_text())
    file_index = {r['path']: r for r in manifest['files']}
    records = []; sources = []; all_rows = 0
    for arch in ['t3', 'linearno', 'linearno_conv']:
        relative = f'E3_standard/airfoil/records/{arch}/metrics_complete.csv'
        if (ROOT / (relative + '.gz')).exists():
            relative += '.gz'
        rows, raw_hash = load_csv(ROOT / relative)
        assert raw_hash == file_index[relative]['source_sha256']
        assert len(rows) == (880 if arch == 'linearno_conv' else 40480)
        all_rows += len(rows)
        sources.append(dict(path=relative, uncompressed_sha256=raw_hash, rows=len(rows)))
        for row in rows:
            if row['metric'] != 'field_relative_l2':
                continue
            row['normalizer_id'] = row['normalizer_sha256']
            row['backbone'] = row['backbone'].replace('_fullgrid', '')
            assert row['dataset'] == 'airfoil2d' and row['training_seed'] == '0'
            records.append(row)
    split = json.loads((ROOT / 'E3_standard/split_manifest.json').read_text())
    expected = {('airfoil2d', 'published_cl_gt_0p02_110'): split['airfoil']['primary_test_ids']}
    areas, curves, _ = aggregate(records, expected)
    published, _ = load_csv(ROOT / 'E3_standard/results/backbone_rauec.csv')
    key = lambda r: tuple(str(r[k]) for k in GROUP) + (r['method'],)
    area_index = {key(r): r for r in published}
    assert len(areas) == len(area_index) == 20
    maximum = 0.
    for row in areas:
        original = area_index[key(row)]
        for field in ['mean', 'sampling_sd', 'seed0_rauec', 'seed1_rauec', 'seed2_rauec']:
            a, b = float(row[field]), float(original[field])
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
            maximum = max(maximum, abs(a-b))
    absolute, _ = load_csv(ROOT / 'E3_standard/results/backbone_absolute_errors.csv')
    raw_groups = defaultdict(list)
    for row in records:
        raw_groups[key(row) + (float(row['budget_fraction']),)].append(row)
    assert len(raw_groups) == len(absolute) == 66
    for row in absolute:
        current = raw_groups[key(row) + (float(row['budget_fraction']),)]
        if row['method'] == 'full':
            observed = float(np.mean([float(r['value']) for r in current]))
            wanted = float(row['mean_error'])
            np.testing.assert_allclose(observed, wanted, rtol=1e-12, atol=1e-14)
            maximum = max(maximum, abs(observed-wanted))
        else:
            means = [np.mean([float(r['value']) for r in current if r['sampling_seed'] == str(seed)]) for seed in [0,1,2]]
            for observed, wanted in [(float(np.mean(means)), float(row['mean_error'])),
                                     (float(np.std(means, ddof=1)), float(row['sampling_sd']))]:
                np.testing.assert_allclose(observed, wanted, rtol=1e-12, atol=1e-14)
                maximum = max(maximum, abs(observed-wanted))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'independent_rauec.csv', areas)
    write_csv(args.output / 'independent_seed_curves.csv', curves)
    result = dict(passed=True, scope='Airfoil training-seed0 point backbones and separate conv-temp Full supplement, completed field accuracy only',
        raw_rows=all_rows, field_rows=len(records), rauec_rows=len(areas), absolute_rows=len(absolute),
        maximum_statistic_difference=maximum, sources=sources, formal_latency_verified=False)
    (args.output / 'recompute_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
