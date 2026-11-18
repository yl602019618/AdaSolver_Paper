"""Build each CA bank from its own completed backbone and native resolution."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from common import load_model
from head_support import (OUT, CODE, array_digest, coarse_count, coarse_grad_lap_score,
                          digest, head_paths, model_core, read, tensor_digest, write)


def source_paths(resolution):
    data = OUT / f'data/r{resolution}'
    return {key: data / name for key, name in {
        'train_coeff': 'train_coeff.npy', 'validation_coeff': 'validation_coeff.npy',
        'positions': 'positions.npy', 'reference': 'reference_coords.npy',
        'normalizer': 'normalizer.npz'}.items()}


@torch.no_grad()
def build(resolution, device, resume=False):
    bank, _ = head_paths(resolution)
    bank.mkdir(parents=True, exist_ok=True)
    with (bank / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(OUT / 'protocol.json')
        assert protocol['head']['epochs'] == 1000
        assert protocol['head']['training_masks_per_case'] == 1
        assert protocol['head']['coarse_fraction'] == .1
        split = read(OUT / 'split_manifest.json')
        assert split['training_ids'] == list(range(900))
        assert split['validation_ids'] == list(range(900, 1000))
        paths = source_paths(resolution)
        normalizer_sha = digest(paths['normalizer'])
        input_hashes = {name: digest(path) for name, path in paths.items()}
        operator, model_manifest = load_model(resolution, device=device)
        assert model_manifest['resolution'] == resolution
        assert model_manifest['normalizer_sha256'] == normalizer_sha
        operator.requires_grad_(False).eval()
        model_state = tensor_digest(operator)
        checkpoint_sha = model_manifest['checkpoint_sha256']
        manifest_path = bank / 'manifest.json'
        previous = read(manifest_path) if manifest_path.exists() else {}
        if previous:
            if previous['operator_sha256'] != checkpoint_sha or previous['normalizer_sha256'] != normalizer_sha:
                raise ValueError('Existing bank belongs to different frozen inputs; preserve it before changing source')
            if previous['resolution'] != resolution or previous['operator_tensor_sha256'] != model_state:
                raise ValueError('Bank native resolution or operator tensors differ')
            if previous['accessed_data_sha256'] != input_hashes:
                raise ValueError('Bank train/validation inputs or native coordinates changed')
            if previous.get('execution_status') == 'completed':
                print(f'r{resolution} head bank already completed', flush=True)
                return previous
            if not resume:
                raise FileExistsError('Partial head bank exists; use --resume')

        nfull = resolution * resolution
        ncoarse = coarse_count(resolution)
        positions = np.load(paths['positions'], allow_pickle=False)
        reference = np.load(paths['reference'], allow_pickle=False)
        if positions.shape != (nfull, 2) or reference.shape != (nfull, 2):
            raise ValueError('Head reference is not the complete declared native grid')
        assert positions.dtype == np.float32 and reference.dtype == np.float64
        assert np.array_equal(reference, positions.astype(np.float64))
        assert len(np.unique(positions[:, 0])) == len(np.unique(positions[:, 1])) == resolution
        train = np.load(paths['train_coeff'], mmap_mode='r', allow_pickle=False)
        validation = np.load(paths['validation_coeff'], mmap_mode='r', allow_pickle=False)
        assert train.shape == (900, nfull, 1) and validation.shape == (100, nfull, 1)
        assert train.dtype == validation.dtype == np.float32
        with np.load(paths['normalizer'], allow_pickle=False) as norm:
            mean = torch.as_tensor(norm['x_mean'].copy(), dtype=torch.float32).reshape(1, 1, 1)
            std = torch.as_tensor(norm['x_std'].copy(), dtype=torch.float32).reshape(1, 1, 1)
        assert bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all())

        shapes = {
            'features.npy': ((1000, ncoarse, 128), np.float32),
            'teacher_predictions.npy': ((1000, ncoarse, 1), np.float32),
            'raw_scores.npy': ((1000, ncoarse), np.float64),
            'coarse_indices.npy': ((1000, ncoarse), np.int64),
        }
        arrays = {}
        for name, (shape, dtype) in shapes.items():
            if previous:
                arrays[name] = np.load(bank / name, mmap_mode='r+')
                assert arrays[name].shape == shape and arrays[name].dtype == dtype
            else:
                arrays[name] = np.lib.format.open_memmap(bank / name, mode='w+', shape=shape, dtype=dtype)
        ids = np.arange(1000, dtype=np.int64)
        if not previous:
            np.save(bank / 'sample_ids.npy', ids)
        else:
            np.testing.assert_array_equal(np.load(bank / 'sample_ids.npy'), ids)

        first = int(previous.get('completed_rows', 0))
        assert 0 <= first <= 1000
        prior_seconds = float(previous.get('bank_seconds', 0.))
        started = time.perf_counter()
        source_hashes = {name: digest(CODE / name) for name in ['prepare_head.py', 'head_support.py', 'common.py']}
        archive = bank / 'source_at_start'
        archive.mkdir(exist_ok=True)
        for name in source_hashes:
            target = archive / name
            if target.exists() and digest(target) != source_hashes[name]:
                raise ValueError('Bank source implementation changed across resume')
            if not target.exists():
                shutil.copyfile(CODE / name, target)
        state = dict(
            execution_status='running', benchmark='darcy', resolution=resolution,
            backbone='Transolver3', operator_sha256=checkpoint_sha,
            operator_checkpoint=model_manifest.get('checkpoint', str(OUT / f'checkpoints/r{resolution}/best.pt')),
            operator_manifest=str(OUT / f'checkpoints/r{resolution}/manifest.json'),
            operator_tensor_sha256=model_state, normalizer_sha256=normalizer_sha,
            training_ids=list(range(900)), validation_ids=list(range(900, 1000)), test_ids_used=[],
            sample_ids_sha256=digest(bank / 'sample_ids.npy'),
            feature_source='own frozen backbone block0 output on own native-resolution coarse subset',
            teacher='own frozen normalized coarse prediction; original raw grad_lap at native resolution',
            teacher_grid_shape=[resolution, resolution], nfull=nfull, coarse_count=ncoarse,
            hidden_dim=128, head_seed=0, masks_per_case=1,
            mask_rng='numpy SeedSequence([0, case_id, 20261114])',
            target='zscore_log1p_grad_lap', target_normalizer_fit='training rows0:900 only',
            accessed_data_files={key: str(path) for key, path in paths.items()},
            accessed_data_sha256=input_hashes,
            source_sha256=source_hashes, completed_rows=first, bank_seconds=prior_seconds,
            device=str(device), precision='FP32 operator and feature tensors; FP64 teacher arithmetic',
            actual_input_shapes=list(previous.get('actual_input_shapes', [])), execution_pid=os.getpid())
        write(manifest_path, state)
        capture = {}

        def hook(module, inputs, output):
            capture['features'] = output.detach().cpu().numpy().copy()

        handle = model_core(operator).blocks[0].register_forward_hook(hook)
        try:
            for lower in range(first, 1000, 8):
                upper = min(lower + 8, 1000)
                current_ids = ids[lower:upper]
                selected = np.stack([
                    np.random.default_rng(np.random.SeedSequence([0, int(sid), 20261114])).choice(nfull, ncoarse, replace=False)
                    for sid in current_ids]).astype(np.int64)
                coefficient = np.stack([np.asarray(train[sid]) if sid < 900 else np.asarray(validation[sid-900]) for sid in current_ids])
                normalized = (torch.from_numpy(coefficient.copy()) - mean) / std
                coarse_pos = torch.from_numpy(positions[selected]).to(device)
                coarse_fx = torch.stack([normalized[index, points] for index, points in enumerate(selected)]).to(device)
                predicted = operator(coarse_pos, coarse_fx).detach().cpu().numpy()
                hidden = capture['features']
                assert predicted.shape == (len(current_ids), ncoarse, 1)
                assert hidden.shape == (len(current_ids), ncoarse, 128)
                assert predicted.dtype == hidden.dtype == np.float32
                assert np.isfinite(predicted).all() and np.isfinite(hidden).all()
                scores = np.stack([coarse_grad_lap_score(reference, indices, values)
                                   for indices, values in zip(selected, predicted)])
                assert scores.shape == (len(current_ids), ncoarse)
                assert np.isfinite(scores).all() and (scores >= 0).all()
                arrays['features.npy'][lower:upper] = hidden
                arrays['teacher_predictions.npy'][lower:upper] = predicted
                arrays['raw_scores.npy'][lower:upper] = scores
                arrays['coarse_indices.npy'][lower:upper] = selected
                for array in arrays.values():
                    array.flush()
                if lower == first or upper == 1000 or lower <= 900 < upper:
                    state['actual_input_shapes'].append(dict(
                        sample_ids=current_ids.tolist(), full_source_coeff=list(coefficient.shape),
                        coarse_coordinates=list(coarse_pos.shape), coarse_coefficients=list(coarse_fx.shape),
                        normalized_prediction=list(predicted.shape), frozen_features=list(hidden.shape),
                        teacher_full_grid=[resolution, resolution], teacher_coarse_scores=list(scores.shape)))
                state.update(completed_rows=upper, bank_seconds=prior_seconds + time.perf_counter()-started)
                write(manifest_path, state)
                write(OUT / f'runtime/head_bank_r{resolution}.json', state)
                if upper % 100 < 8 or upper == 1000:
                    print(f'head bank r{resolution}: {upper}/1000, coarse={ncoarse}', flush=True)
        finally:
            handle.remove()
        assert state['completed_rows'] == 1000
        log_training = np.log1p(np.asarray(arrays['raw_scores.npy'][:900], dtype=np.float64))
        target_mean = float(log_training.mean())
        target_std = float(log_training.std() + 1e-8)
        targets = np.lib.format.open_memmap(bank / 'targets.npy', mode='w+',
                                           shape=(1000, ncoarse, 1), dtype=np.float32)
        for lower in range(0, 1000, 32):
            logs = np.log1p(np.asarray(arrays['raw_scores.npy'][lower:lower+32], dtype=np.float64))
            targets[lower:lower+32, :, 0] = (logs-target_mean)/target_std
        targets.flush()
        assert tensor_digest(operator) == model_state, 'Frozen backbone state changed during head bank generation'
        names = [*shapes, 'sample_ids.npy', 'targets.npy']
        state.update(execution_status='completed', target_mean=target_mean, target_std=target_std,
                     bank_seconds=prior_seconds+time.perf_counter()-started,
                     files={name: digest(bank/name) for name in names},
                     file_shapes={**{name: list(shape) for name, (shape, _) in shapes.items()},
                                  'targets.npy': [1000, ncoarse, 1], 'sample_ids.npy': [1000]},
                     frozen_operator_unchanged=True)
        write(manifest_path, state)
        write(OUT / f'runtime/head_bank_r{resolution}.json', state)
        print(f'head bank r{resolution} completed', flush=True)
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, choices=[85, 141, 211], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    build(args.resolution, args.device, args.resume)


if __name__ == '__main__':
    main()
