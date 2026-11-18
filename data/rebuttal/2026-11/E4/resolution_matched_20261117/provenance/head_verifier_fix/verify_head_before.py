"""Verify complete head training, resolution identity, bank targets and selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model
from head_support import (OUT, CODE, coarse_count, coarse_grad_lap_score, digest,
                          frozen_features, head_paths, load_head, read, tensor_digest, write)


def compare(actual, expected, name):
    actual = np.asarray(actual); expected = np.asarray(expected)
    difference = np.abs(actual - expected)
    violations = int(np.count_nonzero(difference > 1e-6 + 1e-5*np.abs(expected)))
    if violations:
        raise ValueError((name, 'feature/prediction reproduction failed', violations, float(difference.max())))
    return dict(name=name, max_abs=float(difference.max()), violations=violations,
                relative_norm=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30)))


@torch.no_grad()
def verify(resolution, device='cpu'):
    bank, folder = head_paths(resolution)
    manifest = read(folder/'manifest.json'); bm = read(bank/'manifest.json')
    backbone = read(OUT/f'checkpoints/r{resolution}/manifest.json')
    protocol = read(OUT/'protocol.json')
    assert manifest['execution_status'] == bm['execution_status'] == backbone['execution_status'] == 'completed'
    assert manifest['resolution'] == bm['resolution'] == backbone['resolution'] == resolution
    assert manifest['epochs_completed'] == manifest['epochs'] == 1000
    assert manifest['total_optimization_steps'] == 225000
    assert manifest['world_size'] == protocol['head']['world_size']
    assert manifest['settings']['effective_batch'] == 4
    assert manifest['training_ids'] == bm['training_ids'] == list(range(900))
    assert manifest['validation_ids'] == bm['validation_ids'] == list(range(900,1000))
    assert not manifest['test_used_for_selection'] and bm['test_ids_used'] == []
    assert manifest['test_inputs_opened'] == []
    assert all('test_' not in Path(path).name for path in bm['accessed_data_files'].values())
    for name, path in bm['accessed_data_files'].items():
        assert digest(path) == bm['accessed_data_sha256'][name], ('Head source data changed', name)
    n = resolution*resolution; count = coarse_count(resolution)
    assert manifest['coarse_count'] == bm['coarse_count'] == count
    assert bm['teacher_grid_shape'] == [resolution,resolution]
    assert manifest['operator_sha256'] == bm['operator_sha256'] == backbone['checkpoint_sha256']
    assert digest(OUT/f'checkpoints/r{resolution}/best.pt') == manifest['operator_sha256']
    assert manifest['normalizer_sha256'] == bm['normalizer_sha256'] == backbone['normalizer_sha256']
    assert digest(OUT/f'data/r{resolution}/normalizer.npz') == manifest['normalizer_sha256']
    assert digest(folder/'best.pt') == manifest['checkpoint_sha256']
    assert manifest['head_sha256'] == manifest['checkpoint_sha256']
    assert digest(bank/'manifest.json') == manifest['bank_manifest_sha256']
    for name, expected in bm['files'].items():
        assert digest(bank/name) == expected, ('Head bank file changed',name)
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines() if line.strip()]
    assert len(history) == 1000 and [r['epoch'] for r in history] == list(range(1,1001))
    assert [r['global_step'] for r in history] == [225*epoch for epoch in range(1,1001)]
    errors = np.asarray([r['validation'] for r in history])
    assert np.isfinite(errors).all()
    assert int(np.argmin(errors))+1 == manifest['best_epoch']
    assert float(errors.min()) == manifest['best_validation']
    saved = torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
    assert saved['epochs_completed'] == 1000 and saved['logical_steps'] == 225000
    assert saved['scheduler']['last_epoch'] == 225000
    assert all(int(state['step']) == 225000 for state in saved['optimizer']['state'].values())
    best_saved = saved['best_model']
    best_actual = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert set(best_saved) == set(best_actual)
    assert all(torch.equal(best_saved[k],best_actual[k]) for k in best_saved)

    ids = np.load(bank/'sample_ids.npy',allow_pickle=False)
    indices = np.load(bank/'coarse_indices.npy',mmap_mode='r',allow_pickle=False)
    features = np.load(bank/'features.npy',mmap_mode='r',allow_pickle=False)
    prediction = np.load(bank/'teacher_predictions.npy',mmap_mode='r',allow_pickle=False)
    raw = np.load(bank/'raw_scores.npy',mmap_mode='r',allow_pickle=False)
    targets = np.load(bank/'targets.npy',mmap_mode='r',allow_pickle=False)
    np.testing.assert_array_equal(ids,np.arange(1000))
    assert indices.shape == raw.shape == (1000,count)
    assert features.shape == (1000,count,128) and prediction.shape == targets.shape == (1000,count,1)
    assert indices.dtype == np.int64 and features.dtype == prediction.dtype == targets.dtype == np.float32 and raw.dtype == np.float64
    for sid in range(1000):
        expected = np.random.default_rng(np.random.SeedSequence([0,sid,20261114])).choice(n,count,replace=False)
        np.testing.assert_array_equal(indices[sid],expected)
    logs = np.log1p(np.asarray(raw[:900],dtype=np.float64))
    mean = float(logs.mean()); std = float(logs.std()+1e-8)
    assert mean == bm['target_mean'] == manifest['target_mean']
    assert std == bm['target_std'] == manifest['target_std']
    for lower in range(0,1000,32):
        assert np.isfinite(features[lower:lower+32]).all() and np.isfinite(prediction[lower:lower+32]).all()
        assert np.isfinite(raw[lower:lower+32]).all() and (raw[lower:lower+32]>=0).all()
        expected = ((np.log1p(np.asarray(raw[lower:lower+32]))-mean)/std).astype(np.float32)
        np.testing.assert_array_equal(targets[lower:lower+32,:,0],expected)
    data = OUT/f'data/r{resolution}'
    reference = np.load(data/'reference_coords.npy',allow_pickle=False)
    assert reference.shape == (n,2)
    teacher_checks=[]
    for sid in [0,449,899,900,999]:
        actual = coarse_grad_lap_score(reference,indices[sid],prediction[sid])
        np.testing.assert_array_equal(actual,raw[sid])
        teacher_checks.append(dict(case_id=sid,coarse_count=count,native_grid=[resolution,resolution],maximum_difference=0.))
    operator,_ = load_model(resolution,device=device)
    assert tensor_digest(operator) == bm['operator_tensor_sha256'] == manifest['operator_tensor_sha256']
    acquisition,_ = load_head(resolution,operator,device,require_verified=False)
    coeff = {'train':np.load(data/'train_coeff.npy',mmap_mode='r',allow_pickle=False),
             'validation':np.load(data/'validation_coeff.npy',mmap_mode='r',allow_pickle=False)}
    positions = np.load(data/'positions.npy',allow_pickle=False)
    with np.load(data/'normalizer.npz',allow_pickle=False) as normalizer:
        xm=torch.as_tensor(normalizer['x_mean'],dtype=torch.float32).reshape(1,1,1)
        xs=torch.as_tensor(normalizer['x_std'],dtype=torch.float32).reshape(1,1,1)
    reproduction=[]
    for sid in [0,900]:
        selected=indices[sid]
        p=torch.from_numpy(positions[selected].copy())[None].to(device)
        native_coeff=coeff['train'][sid] if sid<900 else coeff['validation'][sid-900]
        x=(torch.from_numpy(np.asarray(native_coeff).copy())[None]-xm)/xs
        x=x[:,selected].to(device)
        feat=frozen_features(operator,p,x).cpu().numpy()[0]
        pred=operator(p,x).cpu().numpy()[0]
        reproduction.append(compare(feat,features[sid],f'case{sid}_block0'))
        reproduction.append(compare(pred,prediction[sid],f'case{sid}_normalized_teacher_prediction'))
        head_pred=acquisition(p,x).cpu().numpy()
        assert head_pred.shape==(1,count,1) and np.isfinite(head_pred).all()
        decoded=acquisition.decode_score(head_pred)
        assert np.isfinite(decoded).all() and (decoded>=0).all()
    receipt=dict(passed=True,resolution=resolution,head_sha256=manifest['checkpoint_sha256'],
        operator_sha256=manifest['operator_sha256'],normalizer_sha256=manifest['normalizer_sha256'],
        epochs_completed=1000,logical_optimizer_steps=225000,history_rows=1000,
        model_parameter_count=manifest['parameter_count'],bank_records=1000,
        training_cases=900,validation_cases=100,test_cases_used_for_training_or_selection=0,
        all_coarse_indices_reproduced=True,all_standardized_targets_reproduced=True,
        teacher_spot_checks=teacher_checks,feature_and_prediction_reproduction=reproduction,
        verification_device=str(device),full_native_grid=[resolution,resolution],coarse_count=count,
        bank_manifest_sha256=digest(bank/'manifest.json'),verifier_sha256=digest(__file__),
        scope='Completed own-resolution head identity, full training steps/history, train-only target normalization, all masks and native teacher/features; final test accuracy separate')
    write(OUT/f'results/head_r{resolution}_validation.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print(json.dumps(verify(args.resolution,args.device),indent=2))


if __name__=='__main__':main()
