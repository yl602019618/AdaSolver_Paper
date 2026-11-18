"""Recompute curves and both rAUECs with a shared Uniform denominator."""
from __future__ import annotations

import numpy as np

from adasolver.inference.superwing_contract import BUDGETS, NFULL, NCOARSE
from adasolver.metrics.rauec import relative_area_under_error_curve
from adasolver.utils.budget import budget_count


def compare(records, sample_ids, seeds=(0,1,2)):
    signatures={r['protocol_signature'] for r in records}
    stages={r['stage'] for r in records}
    if len(signatures)!=1 or len(stages)!=1:
        raise ValueError('cannot compare different operators/protocols or evaluation stages')
    expected={(int(i),int(s),float(b)) for i in sample_ids for s in seeds for b in BUDGETS}
    groups={}
    for r in records:
        groups.setdefault(r['config_id'],[]).append(r)
    uniform=[cid for cid,rows in groups.items() if rows[0]['method']=='uniform']
    if len(uniform)!=1:
        raise ValueError('exactly one shared Uniform policy is required')
    arrays={}
    summary={}
    for cid,rows in groups.items():
        by={}
        for r in rows:
            key=r['sample_id'],r['seed'],r['budget_fraction']
            if key in by or r['status']!='ok':
                raise ValueError('duplicate or failed evaluation record')
            if r['full_count']!=NFULL or r['selected_count']!=budget_count(NFULL,r['budget_fraction']):
                raise ValueError('incorrect SuperWing final budget')
            if r['method'] in ('cpg','ca') and r['coarse_count']!=NCOARSE:
                raise ValueError('incorrect SuperWing coarse budget')
            by[key]=r
        if set(by)!=expected:
            raise ValueError(f'{cid}: missing/unexpected sample-seed-budget records')
        arrays[cid]={}
        metric_fields=[('field','field_relative_l2'),('cd','cd_relative_error')]
        if 'cdcl_relative_error' in rows[0]:
            metric_fields.extend([('cl','cl_relative_error'),('cdcl','cdcl_relative_error')])
        for metric,error in metric_fields:
            values=np.array([[[by[(i,s,b)][error] for b in BUDGETS] for i in sample_ids] for s in seeds])
            if not np.isfinite(values).all() or np.any(values<0):
                raise ValueError('invalid metric values')
            arrays[cid][metric]=values
        summary[cid]=dict(method=rows[0]['method'],target=rows[0]['metric_target'],record_count=len(rows),metrics={},
                          mean_timing={k:float(np.mean([r[k] for r in rows])) for k in
                                       ('acquisition_seconds','sampling_seconds','compensation_seconds','final_seconds',
                                        'reconstruction_seconds','metric_seconds','geometry_seconds','total_seconds')},
                          mean_peak_gpu_memory_bytes=float(np.mean([r['peak_gpu_memory_bytes'] for r in rows])))
    for cid,metrics in arrays.items():
        for metric,values in metrics.items():
            per_seed=values.mean(axis=1)
            u=arrays[uniform[0]][metric].mean(axis=1)
            ratios=[relative_area_under_error_curve(BUDGETS,e,d) for e,d in zip(per_seed,u)]
            summary[cid]['metrics'][metric]=dict(mean_curve=per_seed.mean(axis=0).tolist(),
                per_seed_curve=per_seed.tolist(),seed_std=per_seed.std(axis=0,ddof=int(len(seeds)>1)).tolist(),
                rauec=relative_area_under_error_curve(BUDGETS,per_seed.mean(axis=0),u.mean(axis=0)),
                per_seed_rauec=ratios)
    return dict(protocol_signature=next(iter(signatures)),stage=next(iter(stages)),budgets=list(BUDGETS),
                sample_ids=list(sample_ids),seeds=list(seeds),uniform_config_id=uniform[0],policies=summary)
