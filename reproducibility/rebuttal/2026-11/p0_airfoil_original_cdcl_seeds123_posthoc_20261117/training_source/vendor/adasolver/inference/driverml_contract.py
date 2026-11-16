"""DriverML protocol and configuration shared by all method entry points."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from adasolver.utils.budget import budget_count

ROOT = Path(__file__).resolve().parents[2]
BUDGETS = (0.375, 0.5, 0.75)
COARSE_FRACTION = 0.10
SEEDS = (0, 1, 2)
TARGETS = ("pressure_field", "cd")
PROTOCOL = "driverml_full_surface_budget10_v1"


def coarse_count(nfull: int) -> int:
    return budget_count(nfull, COARSE_FRACTION)


def validate_coarse_indices(indices, nfull):
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) != coarse_count(nfull):
        raise ValueError("DriverML coarse set must contain exactly 10% of the full surface")
    if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= nfull):
        raise ValueError("coarse indices must be unique and within the full surface")


def rng_for(seed, run_id, stream=0, budget_id=0):
    return np.random.default_rng(np.random.SeedSequence([seed, run_id, stream, budget_id]))


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/benchmarks/driverml.yaml"
    config = yaml.safe_load(path.read_text())
    expected_model = dict(architecture='Transolver3',hidden_dim=256,heads=8,layers=8,slices=32,
                          mlp_ratio=2,space_dim=6,fun_dim=0,pressure_out_dim=1,wss_out_dim=3)
    if config.get('name')!='driverml' or config['model']!=expected_model:
        raise ValueError('DriverML configuration must describe the frozen pressure/WSS architecture')
    protocol = config["protocol"]
    if tuple(protocol["budgets"]) != BUDGETS or protocol["coarse_fraction"] != COARSE_FRACTION:
        raise ValueError("DriverML requires three official budgets and a fixed 10% coarse set")
    if tuple(protocol["seeds"]) != SEEDS or protocol["pool_final"] != 0:
        raise ValueError("DriverML requires seeds 0/1/2 and a full reference surface")
    if (protocol['final_reference']!='full_mesh' or config['geometry']['area_weighted']
            or config['geometry']['reconstruction']!='ckdtree_nn'
            or config['metrics']['pressure_error']!='physical_unweighted_relative_l2'
            or config['metrics']['cd_error']!='relative_absolute_error'
            or config['metrics']['wss_source']!='predicted'):
        raise ValueError('DriverML field, reconstruction and Cd metric conventions are fixed')
    overrides = {"data_path": "DRIVERML_DATA_ROOT", "normalizer_path": "DRIVERML_NORMALIZER",
                 "split_path": "DRIVERML_SPLITS", "labels_path": "DRIVERML_LABELS"}
    for key, env in overrides.items():
        value = Path(os.environ.get(env, config[key])).expanduser()
        config[key] = str(value if value.is_absolute() else ROOT / value)
    for field in ("pressure", "wss"):
        value = Path(os.environ.get(f"DRIVERML_{field.upper()}_CHECKPOINT", config["checkpoints"][field])).expanduser()
        config["checkpoints"][field] = str(value if value.is_absolute() else ROOT / value)
    return config
