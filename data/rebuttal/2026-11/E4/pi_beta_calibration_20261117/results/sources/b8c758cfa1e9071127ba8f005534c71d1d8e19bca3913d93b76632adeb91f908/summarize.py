"""Validation-only beta selection and independent Darcy PI field statistics.

The public numerical operation is case mean -> per-sampling-seed AUEC ratio
to matched Uniform -> mean/sample SD over the three sampling seeds. Full is
excluded from the integral. No test value selects a beta or a policy variant.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (85, 141, 211)
BUDGETS = (.375, .5, .75)
SEEDS = (0, 1, 2)
BETAS = (0., .125, .25, .375, .5, .625, .75, .875, 1., 1.5, 2.)
BASELINE = .5
TIE_TOLERANCE = 1e-12
FIELDS = ('split', 'case_id', 'source_row', 'resolution', 'method', 'beta',
          'budget_fraction', 'retained_points', 'sampling_seed',
          'field_relative_l2', 'selected_indices_sha256',
          'checkpoint_sha256', 'normalizer_id', 'status')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    require(bool(values), 'Cannot average an empty group')
    return math.fsum(values) / len(values)


def area(values):
    require(len(values) == 3, 'AUEC requires the three locked budgets')
    return .0625 * values[0] + .1875 * values[1] + .125 * values[2]


def csv_write(path, rows):
    require(bool(rows), f'No numeric rows for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_protocol(path):
    doc = json.loads(path.read_text())
    expected = dict(resolutions=list(RESOLUTIONS), budgets=list(BUDGETS),
                    sampling_seeds=list(SEEDS), beta_candidates=list(BETAS),
                    calibration_case_ids=list(range(1400, 1500)),
                    test_case_ids=list(range(1000, 1200)))
    for key, value in expected.items():
        require(doc.get(key) == value, f'Protocol {key} differs from the predeclared matrix')
    for key in ['checkpoint_sha256', 'normalizer_id']:
        require(isinstance(doc.get(key), str) and bool(doc[key]), f'Protocol lacks {key}')
    require(not set(doc['calibration_case_ids']) & set(doc['test_case_ids']),
            'Calibration and final populations overlap')
    return doc


def normalized_beta(value):
    return None if value in ('', None) else float(value)


def expected_count(cases, betas):
    return len(cases) * (1 + len(BUDGETS) * len(SEEDS) * (1 + len(betas)))


def read_matrix(root, protocol, split, beta_sets, selection_hash=None):
    cases = protocol['calibration_case_ids'] if split == 'validation' else protocol['test_case_ids']
    expected_ids = set(cases)
    all_records = []; sources = {}; audits = []
    for resolution in RESOLUTIONS:
        path = root / 'records' / f'{split}_r{resolution}.csv'
        payload = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
        reader = csv.DictReader(io.StringIO(payload.decode()))
        require(reader.fieldnames is not None and set(FIELDS) <= set(reader.fieldnames),
                f'Missing CSV fields in {path}')
        records = []; keyed = {}; paired_indices = defaultdict(set)
        expected_betas = set(beta_sets[resolution])
        for line, source in enumerate(reader, 2):
            row = dict(source)
            require(row['split'] == split and int(row['resolution']) == resolution,
                    f'Incorrect split/resolution at {path}:{line}')
            case = int(row['case_id']); source_row = int(row['source_row'])
            require(case in expected_ids and source_row == case - 1000,
                    f'Incorrect fixed population at {path}:{line}')
            require(row['checkpoint_sha256'] == protocol['checkpoint_sha256'],
                    f'Checkpoint identity mismatch at {path}:{line}')
            require(row['normalizer_id'] == protocol['normalizer_id'],
                    f'Normalizer identity mismatch at {path}:{line}')
            require(row['status'] == 'ok', f'Failed observation retained at {path}:{line}; matrix cannot be accepted')
            value = float(row['field_relative_l2'])
            require(math.isfinite(value) and value >= 0, f'Invalid field error at {path}:{line}')
            method = row['method']; beta = normalized_beta(row['beta'])
            budget = float(row['budget_fraction'])
            seed = None if row['sampling_seed'] == '' else int(row['sampling_seed'])
            require(method in {'full', 'uniform', 'pi'}, f'Unknown method at {path}:{line}')
            if method == 'full':
                require(beta is None and budget == 1. and seed is None,
                        f'Full must have budget1 and no beta/sampling seed at {path}:{line}')
            else:
                require(budget in BUDGETS and seed in SEEDS, f'Unexpected budget/seed at {path}:{line}')
                require((method == 'uniform' and beta is None) or (method == 'pi' and beta in expected_betas),
                        f'Unexpected beta at {path}:{line}')
            count = int(row['retained_points'])
            require(count == round(resolution * resolution * budget), f'Retained point count mismatch at {path}:{line}')
            index_hash = row['selected_indices_sha256']
            require(len(index_hash) == 64 and all(c in '0123456789abcdef' for c in index_hash.lower()),
                    f'Missing selected-index hash at {path}:{line}')
            if selection_hash and row.get('selection_sha256'):
                require(row['selection_sha256'] == selection_hash, f'Test record uses a different selection at {path}:{line}')
            key = (case, method, beta, budget, seed)
            require(key not in keyed, f'Duplicate observation at {path}:{line}')
            row.update(case_id=case, source_row=source_row, resolution=resolution,
                       beta=beta, budget_fraction=budget, sampling_seed=seed,
                       retained_points=count, field_relative_l2=value)
            records.append(row); keyed[key] = row
            if method == 'pi':
                paired_indices[case, budget, seed].add(index_hash)
        expected = {(case, 'full', None, 1., None) for case in cases}
        expected.update((case, 'uniform', None, budget, seed)
                        for case in cases for budget in BUDGETS for seed in SEEDS)
        expected.update((case, 'pi', beta, budget, seed)
                        for case in cases for beta in expected_betas for budget in BUDGETS for seed in SEEDS)
        require(set(keyed) == expected, f'Incomplete/extra matrix in {path}: observed {len(keyed)}, expected {len(expected)}')
        require(len(records) == expected_count(cases, expected_betas), f'Unexpected row count in {path}')
        require(len(paired_indices) == len(cases)*9 and all(len(hashes) == 1 for hashes in paired_indices.values()),
                f'PI beta candidates do not retain identical selected indices in {path}')
        audits.append(dict(resolution=resolution, cases=len(cases), rows=len(records),
                           expected_rows=len(expected), beta_values=sorted(expected_betas),
                           paired_index_groups=len(paired_indices), paired_index_hashes_match=True))
        all_records.extend(records)
    return all_records, sources, audits


def aggregate(records, protocol_hash):
    groups = defaultdict(list)
    for row in records:
        groups[row['resolution'], row['method'], row['beta'], row['budget_fraction'], row['sampling_seed']].append(row)
    split = records[0]['split']; rauec = []; curves = []; absolute = []
    for resolution in RESOLUTIONS:
        full = groups[resolution, 'full', None, 1., None]
        full_mean = mean(r['field_relative_l2'] for r in full)
        common = dict(split=split, resolution=resolution, checkpoint_sha256=full[0]['checkpoint_sha256'],
                      normalizer_id=full[0]['normalizer_id'], protocol_sha256=protocol_hash,
                      cases=len(full), training_seed_count=1)
        absolute.append(dict(**common, method='full', beta='', budget_fraction=1.,
                             retained_points=resolution*resolution, mean_error=full_mean, sampling_sd='',
                             sampling_seed_count=0, same_checkpoint_full_error=full_mean,
                             seed0_error='', seed1_error='', seed2_error=''))
        candidates = [('uniform', None)] + [('pi', beta) for beta in sorted({r['beta'] for r in records if r['resolution']==resolution and r['method']=='pi'})]
        uniform_areas = {}
        for seed in SEEDS:
            values = [mean(r['field_relative_l2'] for r in groups[resolution,'uniform',None,b,seed]) for b in BUDGETS]
            uniform_areas[seed] = area(values)
            require(uniform_areas[seed] > 0, f'Nonpositive Uniform area at resolution{resolution}/seed{seed}')
        for method, beta in candidates:
            ratios = []; means_by_budget = defaultdict(list)
            for seed in SEEDS:
                values = []
                for budget in BUDGETS:
                    group = groups[resolution, method, beta, budget, seed]
                    value = mean(r['field_relative_l2'] for r in group)
                    values.append(value); means_by_budget[budget].append(value)
                    curves.append(dict(**common, method=method, beta='' if beta is None else beta,
                                       budget_fraction=budget, sampling_seed=seed, mean_error=value,
                                       retained_points=group[0]['retained_points']))
                ratios.append(area(values)/uniform_areas[seed])
            rauec.append(dict(**common, method=method, beta='' if beta is None else beta,
                              mean_rauec=mean(ratios), sampling_sd=statistics.stdev(ratios),
                              seed0_rauec=ratios[0], seed1_rauec=ratios[1], seed2_rauec=ratios[2],
                              sampling_seed_count=3, mean_below_one=mean(ratios)<1.,
                              sampling_seeds_below_one=sum(value<1. for value in ratios),
                              uncertainty='sample SD across3 sampling seeds; one frozen training model'))
            for budget in BUDGETS:
                vals = means_by_budget[budget]
                absolute.append(dict(**common, method=method, beta='' if beta is None else beta,
                    budget_fraction=budget, retained_points=round(resolution*resolution*budget),
                    mean_error=mean(vals), sampling_sd=statistics.stdev(vals), sampling_seed_count=3,
                    same_checkpoint_full_error=full_mean, seed0_error=vals[0], seed1_error=vals[1], seed2_error=vals[2]))
    return rauec, curves, absolute


def best_beta(scores):
    minimum = min(scores.values())
    tied = [beta for beta, value in scores.items() if value <= minimum + TIE_TOLERANCE]
    return min(tied, key=lambda beta: (abs(beta-BASELINE), beta))


def build_selection(rauec, protocol_hash, sources, audits):
    means = {(r['resolution'], float(r['beta'])): r['mean_rauec'] for r in rauec if r['method']=='pi'}
    scores = {beta:max(means[resolution,beta] for resolution in RESOLUTIONS) for beta in BETAS}
    shared = best_beta(scores)
    per_resolution = {str(resolution):best_beta({beta:means[resolution,beta] for beta in BETAS}) for resolution in RESOLUTIONS}
    test_betas = {str(resolution):sorted({BASELINE,shared,per_resolution[str(resolution)]}) for resolution in RESOLUTIONS}
    objectives = [dict(beta=beta, resolution85_mean=means[85,beta], resolution141_mean=means[141,beta],
                       resolution211_mean=means[211,beta], worst_resolution_mean_rauec=scores[beta],
                       mean_over_resolutions=mean(means[resolution,beta] for resolution in RESOLUTIONS)) for beta in BETAS]
    return dict(status='locked_on_validation', protocol_sha256=protocol_hash,
        beta_candidates=list(BETAS), baseline_beta=BASELINE,
        selection_rule='minimize maximum resolution mean rAUEC on validation; within1e-12 choose closest to0.5, then smaller beta',
        tie_tolerance=TIE_TOLERANCE, shared_beta=shared,
        shared_validation_objective=scores[shared], shared_validation_all_resolutions_below_one=scores[shared]<1.,
        per_resolution_beta=per_resolution, test_betas_by_resolution=test_betas,
        secondary_policy='per-resolution validation minimum; reported separately and never selected against the primary policy using test data',
        calibration_case_ids=list(range(1400,1500)), test_case_ids=list(range(1000,1200)),
        validation_sources=sources, validation_matrix=audits, validation_candidates=objectives,
        expected_counts=dict(validation_by_resolution={str(r):10900 for r in RESOLUTIONS},validation_total=32700,
            test_by_resolution={str(r):expected_count(range(200),test_betas[str(r)]) for r in RESOLUTIONS},
            test_total=sum(expected_count(range(200),test_betas[str(r)]) for r in RESOLUTIONS)),
        final_test_used_for_selection=False, beta_above_one_note='1.5 and2.0 are exponent candidates above1, not tempered exponents within0–1')


def audit_selection(existing, computed):
    for key, value in computed.items():
        require(existing.get(key) == value, f'Immutable selection differs in {key}; test results must not revise beta selection')


def selected_views(rauec, selection):
    index={(r['resolution'],r['method'],normalized_beta(r['beta'])):r for r in rauec}
    result=[]
    for resolution in RESOLUTIONS:
        shared=selection['shared_beta']; specific=selection['per_resolution_beta'][str(resolution)]
        base=index[resolution,'pi',BASELINE]; chosen=index[resolution,'pi',shared]; separate=index[resolution,'pi',specific]
        result.append(dict(resolution=resolution, shared_beta=shared, shared_mean_rauec=chosen['mean_rauec'],
            shared_sampling_sd=chosen['sampling_sd'], shared_mean_below_one=chosen['mean_rauec']<1.,
            shared_sampling_seeds_below_one=chosen['sampling_seeds_below_one'], baseline_beta=BASELINE,
            baseline_mean_rauec=base['mean_rauec'], baseline_sampling_sd=base['sampling_sd'],
            shared_minus_baseline=chosen['mean_rauec']-base['mean_rauec'],
            resolution_specific_beta=specific, resolution_specific_mean_rauec=separate['mean_rauec'],
            resolution_specific_sampling_sd=separate['sampling_sd'],
            resolution_specific_mean_below_one=separate['mean_rauec']<1.))
    return result


def report(root, selection, views=None):
    complete=views is not None
    lines=['# Darcy PI β calibration', '',
        'Status: '+('final test matrix verified' if complete else 'validation complete; beta locked; final test pending')+'.', '',
        'The original E4 checkpoint, normalizer, PI prior and reconstruction rule are retained. Only the compensation exponent changes. '
        'All beta variants for one case/budget/sampling seed share the same selected-index hash. '
        'The disjoint calibration cohort is smooth2 rows400–499 (IDs1400–1499); the final E4 cohort is rows0–199 (IDs1000–1199). '
        'The original checkpoint and historic policies had test-informed development, so this is not described as a never-viewed holdout.', '',
        'Candidates: '+', '.join(f'{b:g}' for b in BETAS)+'. Exponents1.5 and2 lie above1; the whole search is not called0–1 tempering.', '',
        f"The primary shared beta is **{selection['shared_beta']:g}**, selected by minimizing the worst resolution’s validation mean rAUEC. "
        'Ties within1e-12 choose the value closest to the original0.5, then the lower beta. '
        'Per-resolution minima are secondary validation-selected diagnostics. Final-test values do not choose beta or select between these two policies.', '',
        'rAUEC first averages case errors separately for each sampling seed and budget, then forms the three-budget trapezoid ratio to the same resolution’s Uniform, '
        'then reports mean and sample SD (ddof=1) over sampling seeds0/1/2. Full100% is excluded from the integral. '
        'The experiment uses one frozen training model; sampling SD is not training SD.', '']
    if complete:
        success=all(row['shared_mean_below_one'] for row in views)
        lines += [f'Observed final primary result: shared-beta mean rAUEC below1 at all three resolutions = **{success}**. '
                  'This status concerns the three means; individual sampling-seed values remain in test_rauec.csv.', '',
                  '| Resolution | Original beta0.5 | Shared beta | Shared rAUEC ± sampling SD | Secondary validation beta | Secondary rAUEC ± sampling SD |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in views:
            lines.append(f"| {row['resolution']}² | {row['baseline_mean_rauec']:.6f} ± {row['baseline_sampling_sd']:.6f} | {row['shared_beta']:g} | {row['shared_mean_rauec']:.6f} ± {row['shared_sampling_sd']:.6f} | {row['resolution_specific_beta']:g} | {row['resolution_specific_mean_rauec']:.6f} ± {row['resolution_specific_sampling_sd']:.6f} |")
        lines += ['', 'All validation candidates and all prespecified test conditions are retained, including values at or above1 and regressions relative to beta0.5. '
                  'No test-best beta is computed. The old published E4 table is a separate fixed-beta result and is not overwritten by this calibration study.', '']
    else:
        lines += ['No final-test benefit is claimed yet. The validation objective and all candidate curves are saved before final evaluation.', '']
    lines += ['Files: `selection.json`, `results/validation_rauec.csv`, `results/validation_candidate_objectives.csv`, '
              '`results/test_rauec.csv` and `results/absolute_errors.csv` when complete. '
              'Raw observations are retained in `records/`; matrix and identity checks are in the statistical receipts. '
              'No formal latency conclusion follows from these accuracy runs.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'read.me').write_text('Darcy PI beta calibration: '+('final test verified' if complete else 'validation complete; test pending')+'.\n'
        f"Primary shared beta={selection['shared_beta']:g}, selected only on calibration IDs1400–1499.\n"
        'Final population is the historical E4 IDs1000–1199; results never change the locked selection.\n'
        'See REPORT.md, selection.json and results/* for candidate values, mean/sampleSD and any unfavorable outcomes.\n')


def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--select',action='store_true'); action.add_argument('--finalize',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--protocol',type=Path)
    args=parser.parse_args();root=args.root.resolve();protocol_path=args.protocol or root/'protocol.json'
    protocol=read_protocol(protocol_path);protocol_hash=sha(protocol_path);result=root/'results';result.mkdir(exist_ok=True)
    validation,sources,audits=read_matrix(root,protocol,'validation',{n:BETAS for n in RESOLUTIONS})
    rauec,curves,absolute=aggregate(validation,protocol_hash)
    selection=build_selection(rauec,protocol_hash,sources,audits)
    selection_path=root/'selection.json'
    if args.select:
        if selection_path.exists():
            existing=json.loads(selection_path.read_text());audit_selection(existing,selection);selection=existing
        else:
            require(not any((root/'records').glob('test_r*.csv')), 'Cannot first select beta after final-test files already exist')
            selection.update(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),statistics_code_sha256=sha(__file__))
            source=result/'sources'/selection['statistics_code_sha256'];source.mkdir(parents=True,exist_ok=True)
            (source/'summarize.py').write_bytes(Path(__file__).read_bytes())
            json_write(selection_path,selection)
    else:
        require(selection_path.exists(), 'Finalization requires a prior immutable validation selection')
        locked=json.loads(selection_path.read_text());audit_selection(locked,selection);selection=locked
    csv_write(result/'validation_rauec.csv',rauec)
    csv_write(result/'validation_budget_curves_by_seed.csv',curves)
    csv_write(result/'validation_absolute_errors.csv',absolute)
    csv_write(result/'validation_candidate_objectives.csv',selection['validation_candidates'])
    json_write(result/'validation_statistics.json',dict(passed=True,rows=len(validation),resolution_matrices=audits,
        source_files=sources,protocol_sha256=protocol_hash,selection_sha256=sha(selection_path),
        mean_before_sampling_seed_ratio=False,full_in_rauec=False))
    if args.select:
        report(root,selection)
        print(json.dumps({k:selection[k] for k in ['status','shared_beta','per_resolution_beta','test_betas_by_resolution','expected_counts']}))
        return
    beta_sets={n:selection['test_betas_by_resolution'][str(n)] for n in RESOLUTIONS}
    final,test_sources,test_audits=read_matrix(root,protocol,'test',beta_sets,sha(selection_path))
    final_rauec,final_curves,final_absolute=aggregate(final,protocol_hash)
    for row in final_rauec+final_absolute+final_curves:
        if row['method']=='pi':
            beta=float(row['beta']);roles=[]
            if beta==BASELINE:roles.append('original_beta0p5_control')
            if beta==selection['shared_beta']:roles.append('primary_shared_validation_beta')
            if beta==selection['per_resolution_beta'][str(row['resolution'])]:roles.append('secondary_resolution_validation_beta')
            row['roles']=';'.join(roles)
        else:row['roles']='matched_uniform' if row['method']=='uniform' else 'full_reference'
    views=selected_views(final_rauec,selection)
    csv_write(result/'test_rauec.csv',final_rauec)
    csv_write(result/'test_budget_curves_by_seed.csv',final_curves)
    csv_write(result/'absolute_errors.csv',final_absolute)
    csv_write(result/'selected_beta_comparison.csv',views)
    json_write(result/'final_statistics.json',dict(passed=True,status='complete_for_prespecified_final_accuracy_matrix',
        rows=len(final),expected_rows=selection['expected_counts']['test_total'],resolution_matrices=test_audits,
        sources=test_sources,protocol_sha256=protocol_hash,selection_sha256=sha(selection_path),
        selection_recomputed_from_validation=True,final_test_used_for_selection=False,
        primary_shared_beta=selection['shared_beta'],primary_all_resolution_means_below_one=all(r['shared_mean_below_one'] for r in views),
        secondary_all_resolution_means_below_one=all(r['resolution_specific_mean_below_one'] for r in views),
        all_candidate_test_rows_retained=True,training_seed_count=1,sampling_seed_count=3,
        statistics_code_sha256=sha(__file__),formal_latency_verified=False))
    report(root,selection,views)
    print(json.dumps(dict(status='finalized',rows=len(final),shared_beta=selection['shared_beta'],
        primary_all_resolution_means_below_one=all(r['shared_mean_below_one'] for r in views))))


if __name__=='__main__':main()
