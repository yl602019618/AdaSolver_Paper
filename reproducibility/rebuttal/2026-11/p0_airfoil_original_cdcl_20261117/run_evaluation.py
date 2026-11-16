#!/usr/bin/env python3
"""Run the original Airfoil evaluator with a pinned checkpoint and Cd/Cl policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['control', 'training'], required=True)
    parser.add_argument('--training-seed', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--physical-gpu', type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    if args.cohort == 'training':
        gate = json.loads((ROOT / 'control_verification.json').read_text())
        if gate.get('passed') is not True:
            raise RuntimeError('original Table 1 control reproduction has not passed')
    registry = json.loads((ROOT / 'runtime_registry.json').read_text())
    key = 'control' if args.cohort == 'control' else f'training_seed{args.training_seed}'
    record = registry[key]
    for field, hash_field in [('checkpoint', 'checkpoint_sha256'), ('head_checkpoint', 'head_sha256'),
                              ('normalizer', 'normalizer_sha256')]:
        if digest(record[field]) != record[hash_field]:
            raise RuntimeError(f'{field} changed for {key}')
    source_hashes = json.loads((ROOT / 'source_snapshot_hashes.json').read_text())
    for relative, expected in source_hashes.items():
        if digest(ROOT / 'original_source' / relative) != expected:
            raise RuntimeError(f'pinned source changed: {relative}')
    sys.path.insert(0, str(ROOT / 'original_source'))
    import torch
    import numpy as np
    from scripts.airfoil2d.run_budget10 import run
    torch.set_num_threads(1)
    torch.manual_seed(0)
    protocol = json.loads((ROOT / 'reference/protocol.json').read_text())
    ids = protocol['sample_ids'][args.shard::args.shards]
    configs = json.loads((ROOT / 'reference/cdcl_configs.json').read_text())
    configs['ca']['head_checkpoint'] = record['head_checkpoint']
    configs['ca']['head_manifest'] = record['head_manifest']
    output = ROOT / 'runs' / key / f'shard{args.shard}'
    launch_path = ROOT / 'launches' / f'{key}_shard{args.shard}.json'
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    if launch_path.exists():
        raise FileExistsError(launch_path)
    launch = {'cohort': key, 'training_seed': record['training_seed'],
              'shard': args.shard, 'shards': args.shards, 'sample_ids': ids,
              'sampling_seeds': [0, 1, 2], 'budgets': protocol['budgets'],
              'physical_gpu': args.physical_gpu, 'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'command': sys.argv, 'pid': os.getpid(), 'started_at_unix': time.time(),
              'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
              'cuda': torch.version.cuda, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'status': 'running', 'registry': record,
              'reconstruction_id': 'original_index_grid_linear_nearest',
              'evaluator': 'scripts.airfoil2d.run_budget10.run / AirfoilEvaluator.record'}
    write(launch_path, launch)
    try:
        run(list(configs.values()), ids, output, device='cuda:0', seeds=(0, 1, 2),
            protocol='positive_lift', min_true_cl=0.02, role='official',
            checkpoint=record['checkpoint'], manifest=record['manifest'], data_path=registry['data_path'])
        manifest = json.loads((output / 'run_manifest.json').read_text())
        if manifest['sample_ids'] != ids:
            raise RuntimeError('the original filter removed or reordered a locked case')
        launch['status'] = 'complete'
    except BaseException as error:
        launch['status'] = 'failed'
        launch['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        launch['finished_at_unix'] = time.time()
        launch['elapsed_seconds'] = launch['finished_at_unix'] - launch['started_at_unix']
        write(launch_path, launch)


if __name__ == '__main__':
    main()
