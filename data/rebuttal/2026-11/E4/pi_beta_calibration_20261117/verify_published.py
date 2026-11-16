"""Recompute the published Darcy PI beta calibration from lossless raw CSVs.

The frozen summarizer is verified before execution. Raw CSVs are materialized
only in a temporary directory; the package, protocol and selected beta are not
modified. No model, private NPZ, MAT data, GPU or third-party Python library is
needed. This checks numerical reproduction, not new experimental measurements.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SUMMARIZER_SHA256 = 'b8c758cfa1e9071127ba8f005534c71d1d8e19bca3913d93b76632adeb91f908'
PROTOCOL_SHA256 = '965da495c0ce60f3de5044a4de38f82f0af2c3719f941e3896b05709cf9b8f96'
SELECTION_SHA256 = 'fa8d4f43cfc8b2a617865abeb6aa13cb4e4e5b5c0413fe686731833c26a4db08'
RESOLUTIONS = (85, 141, 211)
ORIGINAL_BETA_HALF = {85: .9966218517535795, 141: 1.000775870643035, 211: 1.0034479928795303}
TABLES = {
    'validation_rauec.csv': 36,
    'validation_budget_curves_by_seed.csv': 324,
    'validation_absolute_errors.csv': 111,
    'validation_candidate_objectives.csv': 11,
    'test_rauec.csv': 12,
    'test_budget_curves_by_seed.csv': 108,
    'absolute_errors.csv': 39,
    'selected_beta_comparison.csv': 3,
}
IDENTITY_FIELDS = ('split', 'resolution', 'method', 'beta', 'budget_fraction', 'sampling_seed')
TEXT_FIELDS = {'split', 'method', 'checkpoint_sha256', 'normalizer_id', 'protocol_sha256',
               'uncertainty', 'roles'}
COUNT_FIELDS = {'resolution', 'cases', 'training_seed_count', 'sampling_seed_count',
                'retained_points', 'sampling_seed', 'sampling_seeds_below_one',
                'shared_sampling_seeds_below_one'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_table(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric_equal(left, right):
    return math.isfinite(left) and math.isfinite(right) and abs(left-right) <= 1e-14 + 1e-12*abs(right)


def compare_table(actual, published, name, expected_count):
    require(len(actual) == len(published) == expected_count,
            f'{name}: incorrect summary row count')
    require(bool(actual) and set(actual[0]) == set(published[0]), f'{name}: summary schema differs')
    fields = list(actual[0])
    keys = [k for k in IDENTITY_FIELDS if k in fields]
    require(bool(keys), f'{name}: missing row identities')
    identity = lambda row: tuple(row[k] for k in keys)
    expected = {identity(row): row for row in published}
    computed = {identity(row): row for row in actual}
    require(len(expected) == len(published) and len(computed) == len(actual), f'{name}: duplicate summary identity')
    require(set(computed) == set(expected), f'{name}: missing or extra summary identities')
    comparisons = 0; maximum = 0.
    for key, row in computed.items():
        original = expected[key]
        for field in fields:
            a, b = row[field], original[field]
            if a in ('', 'True', 'False') or b in ('', 'True', 'False') or field in TEXT_FIELDS or 'sha256' in field:
                require(a == b, f'{name}/{key}/{field}: metadata or result flag differs')
                continue
            try:
                x, y = float(a), float(b)
            except ValueError:
                require(a == b, f'{name}/{key}/{field}: text differs')
                continue
            if field in COUNT_FIELDS:
                require(x == y and x.is_integer(), f'{name}/{key}/{field}: count differs')
            else:
                require(numeric_equal(x, y), f'{name}/{key}/{field}: numeric value differs')
            maximum = max(maximum, abs(x-y)); comparisons += 1
    return {'table': name, 'rows': len(actual), 'numeric_comparisons': comparisons,
            'maximum_absolute_difference': maximum}


def raw_payload(package, logical_path):
    plain = package / logical_path
    compressed = package / (logical_path + '.gz')
    if compressed.is_file():
        stored = compressed.read_bytes()
        payload = gzip.decompress(stored)
        if plain.exists():
            require(digest(plain.read_bytes()) == digest(payload), f'Ambiguous plain/gzip records: {logical_path}')
        return payload, {'path': logical_path + '.gz', 'encoding': 'gzip',
                         'stored_bytes': len(stored), 'stored_sha256': digest(stored)}
    require(plain.is_file(), f'Missing raw record file: {logical_path}(.gz)')
    payload = plain.read_bytes()
    return payload, {'path': logical_path, 'encoding': 'identity',
                     'stored_bytes': len(payload), 'stored_sha256': digest(payload)}


def verify(package):
    package = package.resolve()
    protocol_path = package / 'protocol.json'
    selection_path = package / 'selection.json'
    script_path = package / 'code/summarize.py'
    require(digest(protocol_path.read_bytes()) == PROTOCOL_SHA256, 'Frozen protocol bytes differ')
    require(digest(selection_path.read_bytes()) == SELECTION_SHA256, 'Frozen calibration selection bytes differ')
    require(digest(script_path.read_bytes()) == SUMMARIZER_SHA256, 'Frozen statistics source differs; refusing execution')
    protocol = read_json(protocol_path); selection = read_json(selection_path)
    final_receipt = read_json(package / 'results/final_statistics.json')
    require(selection['protocol_sha256'] == PROTOCOL_SHA256 and selection['statistics_code_sha256'] == SUMMARIZER_SHA256,
            'Selection is not bound to the frozen protocol and code')
    require(selection['shared_beta'] == 1.0 and selection['per_resolution_beta'] == {'85': .625, '141': 1.5, '211': 2.0},
            'Calibration choices differ from this publication')
    require(selection['final_test_used_for_selection'] is False, 'Test-based selection is not this protocol')
    require(final_receipt['passed'] is True and final_receipt['selection_sha256'] == SELECTION_SHA256,
            'Final completion receipt differs from the fixed selection')
    require(final_receipt['protocol_sha256'] == PROTOCOL_SHA256 and final_receipt['statistics_code_sha256'] == SUMMARIZER_SHA256,
            'Final statistics identity mismatch')
    source_records = []; counts = {'validation': 0, 'test': 0}; comparisons = []
    with tempfile.TemporaryDirectory(prefix='darcy_pi_beta_verify_') as directory:
        scratch = Path(directory)
        (scratch / 'records').mkdir()
        shutil.copyfile(protocol_path, scratch / 'protocol.json')
        shutil.copyfile(selection_path, scratch / 'selection.json')
        for split in ['validation', 'test']:
            hashes = selection['validation_sources'] if split == 'validation' else final_receipt['sources']
            for resolution in RESOLUTIONS:
                logical = f'records/{split}_r{resolution}.csv'
                payload, item = raw_payload(package, logical)
                h = digest(payload)
                require(h == hashes[logical], f'Uncompressed record hash differs: {logical}')
                row_count = 0
                for row in csv.DictReader(io.StringIO(payload.decode('utf-8'), newline='')):
                    require(row['protocol_sha256'] == PROTOCOL_SHA256, f'Raw protocol identity differs: {logical}')
                    if split == 'test':
                        require(row['selection_sha256'] == SELECTION_SHA256, f'Raw test selection identity differs: {logical}')
                    row_count += 1
                require(row_count == (10900 if split == 'validation' else 7400), f'Raw count differs: {logical}')
                counts[split] += row_count
                item.update(logical_path=logical, uncompressed_bytes=len(payload), uncompressed_sha256=h, rows=row_count)
                source_records.append(item)
                (scratch / logical).write_bytes(payload)
        require(counts == {'validation': 32700, 'test': 22200}, 'Incomplete raw matrices')
        env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
        execution = subprocess.run([sys.executable, str(script_path), '--root', str(scratch), '--finalize'],
                                   cwd=scratch, env=env, text=True, capture_output=True)
        require(execution.returncode == 0,
                'Frozen summarizer failed: ' + execution.stderr[-4000:])
        require(digest((scratch/'selection.json').read_bytes()) == SELECTION_SHA256, 'Recomputation changed selection')
        for name, expected_rows in TABLES.items():
            comparisons.append(compare_table(read_table(scratch/'results'/name),
                                             read_table(package/'results'/name), name, expected_rows))
        recomputed = read_json(scratch / 'results/final_statistics.json')
        require(recomputed['selection_recomputed_from_validation'] is True and recomputed['final_test_used_for_selection'] is False,
                'Selection was not independently rebuilt from calibration')
        for field in ['rows','expected_rows','resolution_matrices','primary_shared_beta',
                      'primary_all_resolution_means_below_one','secondary_all_resolution_means_below_one',
                      'training_seed_count','sampling_seed_count','all_candidate_test_rows_retained']:
            require(recomputed[field] == final_receipt[field], f'Final receipt differs in {field}')
        final = read_table(scratch / 'results/test_rauec.csv')
        primary = []; secondary = []; baseline = []
        for resolution in RESOLUTIONS:
            pi = [r for r in final if int(r['resolution']) == resolution and r['method'] == 'pi']
            get = lambda beta: next(r for r in pi if float(r['beta']) == beta)
            old = get(.5); shared = get(1.); specific = get(selection['per_resolution_beta'][str(resolution)])
            require(numeric_equal(float(old['mean_rauec']), ORIGINAL_BETA_HALF[resolution]),
                    f'Original beta0.5 reference changed at {resolution}')
            for target, row in [(baseline,old),(primary,shared),(secondary,specific)]:
                target.append({'resolution': resolution, 'beta': float(row['beta']),
                    'mean_rauec': float(row['mean_rauec']), 'sampling_sd': float(row['sampling_sd']),
                    'sampling_seed_rauec': [float(row[f'seed{s}_rauec']) for s in range(3)],
                    'sampling_seeds_below_one': int(row['sampling_seeds_below_one'])})
        primary_means = all(r['mean_rauec'] < 1 for r in primary)
        all_seeds = all(all(v < 1 for v in r['sampling_seed_rauec']) for r in primary)
        require(primary_means and not all_seeds, 'Published mean/individual-seed interpretation differs')
        require(primary[0]['mean_rauec'] > baseline[0]['mean_rauec'], '85² regression relative to beta0.5 was lost')
    # The original package is never passed as the writable --root.
    require(digest(protocol_path.read_bytes()) == PROTOCOL_SHA256 and digest(selection_path.read_bytes()) == SELECTION_SHA256,
            'Source package changed during verification')
    return dict(passed=True, scope='Reproduction of published PI beta scalar results and calibration-only choice; no new inference',
        checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        protocol_sha256=PROTOCOL_SHA256, selection_sha256=SELECTION_SHA256,
        summarizer_sha256=SUMMARIZER_SHA256, verifier_sha256=digest(Path(__file__).read_bytes()),
        calibration_rows=counts['validation'], test_rows=counts['test'], source_files=source_records,
        table_checks=comparisons, numeric_comparisons=sum(r['numeric_comparisons'] for r in comparisons),
        maximum_summary_difference=max(r['maximum_absolute_difference'] for r in comparisons),
        shared_beta=1.0, primary_all_resolution_means_below_one=primary_means,
        primary_all_individual_sampling_seeds_below_one=all_seeds,
        secondary_all_resolution_means_below_one=all(r['mean_rauec'] < 1 for r in secondary),
        secondary_all_individual_sampling_seeds_below_one=all(all(v < 1 for v in r['sampling_seed_rauec']) for r in secondary),
        baseline=baseline, primary=primary, secondary=secondary,
        selection_recomputed_from_calibration=True, final_test_used_for_selection=False,
        package_inputs_unchanged=True, temporary_records_removed=True,
        model_weights_loaded=False, private_arrays_checked=False, formal_latency_verified=False)


def default_package():
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate/'protocol.json').is_file() and (candidate/'selection.json').is_file() and (candidate/'code/summarize.py').is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=default_package(), help='Published package root containing protocol.json and records/')
    parser.add_argument('--output', type=Path, help='Optional path for the JSON verification receipt')
    args = parser.parse_args()
    if args.package is None:
        parser.error('--package is required when this script is outside the package')
    result = verify(args.package)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
