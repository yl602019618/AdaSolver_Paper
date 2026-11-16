"""Frozen SuperWing evaluation protocol and portable configuration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from adasolver.utils.budget import budget_count

ROOT = Path(__file__).resolve().parents[2]
GRID = (128, 256)
NFULL = 32768
BUDGETS = (0.375, 0.5, 0.75)
SEEDS = (0, 1, 2)
COARSE_FRACTION = 0.1
NCOARSE = budget_count(NFULL, COARSE_FRACTION)
PROTOCOL = 'superwing_lattice_field_cd_budget10_v1'
TARGETS = ('field', 'cd', 'cdcl')
MODEL = dict(space_dim=3, fun_dim=2, out_dim=3, hidden_dim=128, heads=8,
             layers=8, mlp_ratio=2, slices=64, dropout=0.0, activation='gelu', ref=8)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def indices_identity(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def rng_for(sample_id, seed, stream, budget_index=0):
    return np.random.default_rng(np.random.SeedSequence([int(sample_id), int(seed), int(stream), int(budget_index)]))


def coarse_indices(sample_id, seed):
    return rng_for(sample_id, seed, 10).choice(NFULL, NCOARSE, replace=False)


def validate_indices(indices, expected_count):
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in 'iu' or len(idx) != expected_count:
        raise ValueError('SuperWing indices must be a one-dimensional integer array of the exact budget')
    if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= NFULL):
        raise ValueError('SuperWing indices must be unique and within the full reference grid')


def validate_policy(policy):
    method=policy.get('method')
    if method not in ('uniform','pi','cpg','ca'):
        raise ValueError('unknown SuperWing sampling method')
    if method=='uniform':
        if policy!=dict(method='uniform',target='shared'):
            raise ValueError('Uniform must be the single shared unmodified baseline')
        return
    if policy.get('target') not in TARGETS:
        raise ValueError('adaptive policy requires a field, cd or cdcl target')
    # Budget and metric changes belong to a new benchmark protocol, not a policy.
    forbidden={'r0','coarse_fraction','coarse_count','budgets','metric','reconstruction','operator'}
    if forbidden & set(policy):
        raise ValueError('sampling policies cannot override the frozen benchmark protocol')
    if not (0<=float(policy.get('beta',1))<=1) or float(policy.get('power',1))<=0:
        raise ValueError('invalid compensation strength or score power')
    if policy.get('budget_parameters') is not None:
        from adasolver.inference.superwing_density import for_budget
        for i in range(3):validate_policy(for_budget(policy,i))
    if method=='ca':
        target=policy.get('head_target','Cp_GT')
        if target=='Cp_GT':
            if policy.get('feature') not in ('cp_grad','cp_lap'):
                raise ValueError('pressure heads require an explicit pressure variation transform')
        else:
            expected={'log1p_joint_lap_GT':'field','log1p_cd_lap_GT':'cd', 'log1p_cdcl_lap_GT':'cdcl'}
            proxy=(target=='log1p_joint_lap_GT' and policy.get('head_role')=='field_variation_proxy' and
                   ((policy['target']=='cd' and policy.get('cd_revision')=='v2') or
                    (policy['target']=='cdcl' and policy.get('density_revision')=='v2')))
            if target not in expected or (policy['target']!=expected[target] and not proxy) or policy.get('feature')!='direct_score':
                raise ValueError('score-head objective/transform mismatch')


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path=None):
    config = yaml.safe_load(resolve(path or 'configs/benchmarks/superwing.yaml').read_text())
    if config['name'] != 'superwing' or config['model'] != MODEL:
        raise ValueError('SuperWing configuration must describe the frozen 3/2/3 Transolver3')
    expected = dict(id=PROTOCOL, nfull=NFULL, grid=list(GRID), budgets=list(BUDGETS),
                    seeds=list(SEEDS), coarse_fraction=COARSE_FRACTION, coarse_count=NCOARSE,
                    reconstruction='native_lattice_linear_nearest', field_scale='dataset_1_150_300',
                    field_metric='unweighted_full_relative_l2', cd_truth='index_column_10_structured',
                    cd_metric='per_sample_relative_absolute_error')
    from adasolver.metrics.superwing_cdcl import PROTOCOL as CDCL_PROTOCOL
    if config['protocol'].get('id') == CDCL_PROTOCOL:
        expected.update(id=CDCL_PROTOCOL, cdcl_truth='index_column_10_divided_by_9_structured',
                        cdcl_metric='per_sample_relative_absolute_error',
                        cdcl_filter='absolute_Cl_ge_strategy_calibration_q10',
                        cdcl_cl_threshold=0.07103351890721582)
    if config['protocol'] != expected:
        raise ValueError('SuperWing protocol differs from the frozen field/Cd budget10 contract')
    for key, variable in [('data_path', 'SUPERWING_DATA_ROOT'), ('split_path', 'SUPERWING_SPLITS'),
                          ('normalizer_path', 'SUPERWING_NORMALIZER'), ('checkpoint', 'SUPERWING_CHECKPOINT'),
                          ('operator_manifest', 'SUPERWING_OPERATOR_MANIFEST')]:
        config[key] = str(resolve(os.environ.get(variable, config[key])))
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
