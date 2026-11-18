"""Verify complete head training, resolution identity, bank targets and selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model
from head_support import (OUT, CODE, coarse_count, coarse_grad_lap_score, digest,
                          frozen_features, head_paths, load_head, model_core, read, tensor_digest, write)


def difference_report(actual, expected, name):
    actual = np.asarray(actual); expected = np.asarray(expected)
    assert actual.shape == expected.shape
    difference = np.abs(actual - expected)
    violations = int(np.count_nonzero(difference > 1e-6 + 1e-5*np.abs(expected)))
    return dict(name=name, max_abs=float(difference.max()), violations=violations,
                relative_norm=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30)),
                absolute_difference_quantiles={str(q):float(np.quantile(difference,q)) for q in [0.,.5,.9,.99,1.]})


def compare(actual, expected, name):
    report = difference_report(actual, expected, name)
    if report['violations']:
        raise ValueError((name, 'feature/prediction reproduction failed', report['violations'], report['max_abs']))
    return report


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
    reproduction=[];batch_size_sensitivity=[];single_request_checks=[]
    capture={}
    def capture_block0(module, inputs, output):
        capture['features']=output.detach().cpu().numpy().copy()
    for sid in [0,900]:
        # Replay the original bank request, including batch shape and order.
        # FP32 GEMM reductions may differ between batch sizes; bank identity
        # must be checked under the bank's recorded execution context.
        lower=(sid//8)*8;upper=min(lower+8,1000);batch_ids=list(range(lower,upper))
        recorded=next(row for row in bm['actual_input_shapes'] if row['sample_ids']==batch_ids)
        assert recorded['coarse_coordinates']==[len(batch_ids),count,2]
        selected=np.array(indices[lower:upper],copy=True)
        coefficient=np.stack([np.asarray(coeff['train'][i]) if i<900 else np.asarray(coeff['validation'][i-900]) for i in batch_ids])
        normalized=(torch.from_numpy(coefficient.copy())-xm)/xs
        p=torch.from_numpy(positions[selected]).to(device)
        x=torch.stack([normalized[index,points] for index,points in enumerate(selected)]).to(device)
        handle=model_core(operator).blocks[0].register_forward_hook(capture_block0)
        try:pred=operator(p,x).cpu().numpy()
        finally:handle.remove()
        feat=capture['features']
        reproduction.append(compare(feat,features[lower:upper],f'bank_batch{lower}:{upper}_block0'))
        reproduction.append(compare(pred,prediction[lower:upper],f'bank_batch{lower}:{upper}_normalized_teacher_prediction'))
        extracted=frozen_features(operator,p,x).cpu().numpy()
        reproduction.append(compare(extracted,feat,f'bank_batch{lower}:{upper}_extractor_vs_full_forward'))

        # Independently check the actual batch-1 inference feature tap against
        # that same batch-1 full forward. The cross-batch differences are saved
        # as numerical diagnostics, not silently assigned a looser tolerance.
        offset=sid-lower;p_single=p[offset:offset+1].contiguous();x_single=x[offset:offset+1].contiguous()
        handle=model_core(operator).blocks[0].register_forward_hook(capture_block0)
        try:pred_single=operator(p_single,x_single).cpu().numpy()[0]
        finally:handle.remove()
        feat_single=capture['features'][0]
        extracted_single=frozen_features(operator,p_single,x_single).cpu().numpy()[0]
        reproduction.append(compare(extracted_single,feat_single,f'case{sid}_batch1_extractor_vs_full_forward'))
        batch_size_sensitivity.append(difference_report(feat_single,features[sid],f'case{sid}_batch1_vs_bank8_block0'))
        batch_size_sensitivity.append(difference_report(pred_single,prediction[sid],f'case{sid}_batch1_vs_bank8_prediction'))
        head_pred=acquisition(p_single,x_single).cpu().numpy()
        assert head_pred.shape==(1,count,1) and np.isfinite(head_pred).all()
        head_from_hook=acquisition.head(torch.from_numpy(feat_single.copy())[None].to(device)).cpu().numpy()
        reproduction.append(compare(head_pred,head_from_hook,f'case{sid}_batch1_head_vs_hook_features'))
        decoded=acquisition.decode_score(head_pred)
        assert np.isfinite(decoded).all() and (decoded>=0).all()
        single_request_checks.append(dict(case_id=sid,input_shape=[1,count,1],output_shape=list(head_pred.shape),
                                          native_resolution=resolution,finite_prediction_and_decoded_score=True))
    receipt=dict(passed=True,resolution=resolution,head_sha256=manifest['checkpoint_sha256'],
        operator_sha256=manifest['operator_sha256'],normalizer_sha256=manifest['normalizer_sha256'],
        epochs_completed=1000,logical_optimizer_steps=225000,history_rows=1000,
        model_parameter_count=manifest['parameter_count'],bank_records=1000,
        training_cases=900,validation_cases=100,test_cases_used_for_training_or_selection=0,
        all_coarse_indices_reproduced=True,all_standardized_targets_reproduced=True,
        teacher_spot_checks=teacher_checks,feature_and_prediction_reproduction=reproduction,
        bank_reproduction_batch_size=8,bank_reproduction_case_groups=[[0,1,2,3,4,5,6,7],[896,897,898,899,900,901,902,903]],
        reproduction_tolerance='abs(actual-expected) <= 1e-6 + 1e-5*abs(expected)',
        batch_size_sensitivity_diagnostics=batch_size_sensitivity,single_request_interface_checks=single_request_checks,
        batch_size_diagnostic_scope='Different FP32 batch shapes; bank identity and batch1 feature-tap identity are independently checked with the unchanged elementwise tolerance',
        verification_device=str(device),full_native_grid=[resolution,resolution],coarse_count=count,
        bank_manifest_sha256=digest(bank/'manifest.json'),verifier_sha256=digest(__file__),
        scope='Completed own-resolution head identity, full training steps/history, train-only target normalization, all masks and native teacher/features; final test accuracy separate')
    write(OUT/f'results/head_r{resolution}_validation.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    print(json.dumps(verify(args.resolution,args.device),indent=2))


if __name__=='__main__':main()
