"""CPU-only interface checks on real native grids and untrained debug models."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch

from common import build_model
from head_support import (OUT, CODE, ScoreHead, coarse_count, coarse_grad_lap_score,
                          digest, frozen_features, model_core, read, tensor_digest, write)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import adasolver.train.darcy_ca_head as original
    from adasolver.inference.cpg.darcy import DarcyCPG
    checks=[]
    for resolution in [85,141,211]:
        torch.manual_seed(0)
        model=build_model().requires_grad_(False).eval()
        identity=tensor_digest(model)
        data=OUT/f'data/r{resolution}'
        coefficient=np.load(data/'train_coeff.npy',mmap_mode='r',allow_pickle=False)
        positions=np.load(data/'positions.npy',allow_pickle=False)
        reference=np.load(data/'reference_coords.npy',allow_pickle=False)
        assert coefficient.shape==(900,resolution*resolution,1)
        assert positions.shape==reference.shape==(resolution*resolution,2)
        with np.load(data/'normalizer.npz',allow_pickle=False) as normalizer:
            mean=torch.as_tensor(normalizer['x_mean'],dtype=torch.float32).reshape(1,1,1)
            std=torch.as_tensor(normalizer['x_std'],dtype=torch.float32).reshape(1,1,1)
        count=coarse_count(resolution)
        indices=np.stack([np.random.default_rng(np.random.SeedSequence([0,sid,20261114])).choice(resolution*resolution,count,replace=False) for sid in range(4)])
        pos=torch.from_numpy(positions[indices].copy())
        full_coeff=(torch.from_numpy(np.asarray(coefficient[:4]).copy())-mean)/std
        fx=torch.stack([full_coeff[i,idx] for i,idx in enumerate(indices)])
        captured={}
        hook=model_core(model).blocks[0].register_forward_hook(lambda module,args,output:captured.update(features=output.detach().clone()))
        with torch.no_grad():
            prediction=model(pos,fx)
            feature=captured['features'].clone()
            helper=frozen_features(model,pos,fx)
        hook.remove()
        np.testing.assert_array_equal(feature.numpy(),helper.numpy())
        scores=[];teacher_differences=[];normalized_differences=[]
        # The native scoring method is pure in these fields. Avoid unrelated
        # patch-building work in this teacher-equivalence check.
        policy=object.__new__(DarcyCPG)
        policy.reference_coords=reference;policy.acquisition='grad_lap'
        policy.clip_quantile=None;policy.score_power=1.
        for idx,pred in zip(indices,prediction.numpy()):
            value=coarse_grad_lap_score(reference,idx,pred)
            old=original.coarse_grad_lap_score(reference,idx,pred)
            np.testing.assert_array_equal(value,old)
            normalized=value/max(float(value.mean()),1e-12)
            cp=policy._score(idx,pred)
            np.testing.assert_allclose(cp,normalized,rtol=1e-13,atol=1e-14)
            teacher_differences.append(float(np.max(np.abs(value-old))))
            normalized_differences.append(float(np.max(np.abs(cp-normalized))))
            scores.append(value)
        logs=np.log1p(np.asarray(scores));target=torch.from_numpy(((logs-logs.mean())/(logs.std()+1e-8))[...,None].astype(np.float32))
        head=ScoreHead()
        assert sum(p.numel() for p in head.parameters())==236049
        initial=tensor_digest(head)
        grad_check=None
        if resolution==85:
            first=copy.deepcopy(head);second=copy.deepcopy(head)
            torch.nn.functional.smooth_l1_loss(first(feature),target).backward()
            for lower in [0,2]:
                (torch.nn.functional.smooth_l1_loss(second(feature[lower:lower+2]),target[lower:lower+2])*.5).backward()
            a=torch.cat([p.grad.flatten() for p in first.parameters()]);b=torch.cat([p.grad.flatten() for p in second.parameters()])
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            grad_check=dict(global4_vs_average_two_local2=True,max_gradient_difference=float((a-b).abs().max()),
                relative_gradient_difference=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)))
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=1e-5)
        losses=[]
        for step in range(2):
            value=head(feature);loss=torch.nn.functional.smooth_l1_loss(value,target)
            assert value.shape==(4,count,1) and bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters())
            torch.nn.utils.clip_grad_norm_(head.parameters(),.1);optimizer.step();losses.append(float(loss.detach()))
        assert tensor_digest(head)!=initial
        assert tensor_digest(model)==identity
        checks.append(dict(resolution=resolution,source_train_shape=list(coefficient.shape),
            full_coordinate_shape=list(reference.shape),coarse_count=count,
            actual_coarse_input_shape=list(pos.shape),coarse_coefficient_shape=list(fx.shape),
            normalized_operator_prediction_shape=list(prediction.shape),feature_shape=list(feature.shape),
            teacher_grid_shape=[resolution,resolution],head_output_shape=list(value.shape),
            feature_hook_matches_inference_helper=True,teacher_scalar_reference_max_difference=max(teacher_differences),
            normalized_teacher_matches_native_CPG=max(normalized_differences),
            head_train_steps=2,head_losses=losses,head_parameters=236049,
            backbone_frozen=True,ddp_mean_loss_gradient_check=grad_check,
            source_cases=[0,1,2,3],test_data_read=False))
    receipt=dict(passed=True,scope='CPU engineering checks on real grid/coefficient input and full native initialized models; no trained accuracy or latency claim',
        production_weights_reused=False,debug_weights_discarded=True,checks=checks,
        head_source_sha256={name:digest(CODE/name) for name in ['head_support.py','prepare_head.py','train_head.py','verify_head.py','preflight_head.py']})
    write(OUT/'development/head_preflight.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
