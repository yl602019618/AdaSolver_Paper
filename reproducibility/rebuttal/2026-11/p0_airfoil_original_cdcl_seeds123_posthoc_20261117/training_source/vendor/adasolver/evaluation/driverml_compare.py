"""Independent raw-record validation and DriverML rAUEC aggregation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from adasolver.inference.driverml_contract import BUDGETS, TARGETS, coarse_count
from adasolver.metrics.rauec import relative_area_under_error_curve
from adasolver.utils.budget import budget_count


def read_records(paths):
    return [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]


def compare(records, *, sample_ids, seeds=(0,1,2), methods=('uniform','pi','cpg','ca')):
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('expected sample IDs and seeds must be nonempty and unique')
    if 'uniform' not in methods:
        raise ValueError('the Uniform denominator is required')
    expected = {(rid,seed,budget) for rid in sample_ids for seed in seeds for budget in BUDGETS}
    summary = {'budgets': list(BUDGETS), 'sample_ids': list(sample_ids), 'seeds': list(seeds), 'targets': {}}
    targets = sorted({r['metric_target'] for r in records})
    if not targets or any(t not in TARGETS for t in targets):
        raise ValueError('unknown or missing metric target')
    signatures = {r['protocol_signature'] for r in records}
    if len(signatures) != 1:
        raise ValueError('cannot mix incompatible operators or evaluation protocols')
    summary['protocol_signature'] = next(iter(signatures))
    stages={r.get('stage','unspecified') for r in records}
    if len(stages)!=1:
        raise ValueError('cannot mix diagnostic, development and official records')
    summary['stage']=next(iter(stages))
    point_counts={rid:{r['full_count'] for r in records if r['run_id']==rid} for rid in sample_ids}
    if any(len(counts)!=1 for counts in point_counts.values()):
        raise ValueError('all methods must use the same full reference surface for each car')
    for target in targets:
        error_key = 'pressure_field_relative_l2' if target == 'pressure_field' else 'cd_relative_error'
        rows = {}
        curves = {}
        for method in methods:
            selected = [r for r in records if r['method'] == method and r['metric_target'] == target]
            if not selected or len({r['config_id'] for r in selected}) != 1:
                raise ValueError(f'{target}/{method}: missing records or multiple method configs')
            indexed = {}
            for rec in selected:
                key = (rec['run_id'],rec['seed'],rec['budget_fraction'])
                if key in indexed:
                    raise ValueError(f'duplicate DriverML record: {key}')
                if rec['status'] != 'ok' or not np.isfinite(rec[error_key]) or rec[error_key] < 0:
                    raise ValueError('invalid metric in DriverML record')
                if rec['selected_count'] != budget_count(rec['full_count'],rec['budget_fraction']):
                    raise ValueError('incorrect final point count')
                if rec.get('budget_count',rec['selected_count'])!=rec['selected_count']:
                    raise ValueError('declared budget count differs from selected points')
                if method in ('cpg','ca') and rec['coarse_count'] != coarse_count(rec['full_count']):
                    raise ValueError('incorrect coarse point count')
                if target == 'cd' and rec.get('wss_source') != 'predicted':
                    raise ValueError('Cd primary results require predicted WSS')
                indexed[key] = rec
            if set(indexed) != expected:
                raise ValueError(f'{target}/{method}: missing or unexpected sample/seed/budget records')
            array = np.array([[[indexed[(rid,s,c)][error_key] for c in BUDGETS] for rid in sample_ids] for s in seeds])
            per_seed = array.mean(axis=1)
            mean = per_seed.mean(axis=0)
            curves[method] = (mean,per_seed)
            rows[method] = dict(config_id=selected[0]['config_id'], record_count=len(selected),
                                mean_error=mean.tolist(), per_seed_error=per_seed.tolist(),
                                error_std_across_seeds=per_seed.std(axis=0,ddof=1 if len(seeds)>1 else 0).tolist(),
                                auec=float(np.trapezoid(mean,BUDGETS)),
                                mean_acquisition_seconds=float(np.mean([r.get('acquisition_seconds',0) for r in selected])),
                                mean_total_seconds=float(np.mean([r.get('total_seconds',0) for r in selected])))
        uniform, uniform_seeds = curves['uniform']
        for method, (mean,per_seed) in curves.items():
            ratios = [relative_area_under_error_curve(BUDGETS,e,u) for e,u in zip(per_seed,uniform_seeds)]
            rows[method].update(rauec=relative_area_under_error_curve(BUDGETS,mean,uniform),
                                per_seed_rauec=ratios, rauec_std_across_seeds=float(np.std(ratios,ddof=1 if len(seeds)>1 else 0)))
        summary['targets'][target] = rows
    return summary
