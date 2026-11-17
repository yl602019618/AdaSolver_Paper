"""Verify the actual head bank, checkpoint selection and native feature path."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from common import ROOT,E3,activate_sources,sha256,write_json,verify_lock


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);activate_sources();verify_lock()
    import numpy as np
    import torch
    from evaluate_linearno import LinearNORuntime
    from linearno_adapter import install_linear_compensation
    from linearno_ca import LinearNOCA
    from adasolver.inference.ca.shapenet_car import CarPressureHead
    from adasolver.inference.shapenet_car_contract import rng_for,coarse_count
    from adasolver.train.shapenet_car_ca_head import load_split
    lock=open(f'/tmp/adasolver_profile_gpu_{args.gpu}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    manifest=E3/'checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json'
    runtime=LinearNORuntime(manifest);splits=runtime.splits(ROOT/'splits/seed0.json')
    bank_root=ROOT/'records/head_dataset_seed0';bank=json.loads((bank_root/'manifest.json').read_text())
    head_path=ROOT/'checkpoints/seed0/head/ca_pressure_linearno_trainingseed0_headseed0_manifest.json';head=json.loads(head_path.read_text())
    assert bank['status']=='complete' and head['status']=='trained_head'
    assert bank['operator_sha256']==head['operator_sha256']==runtime.metadata['checkpoint_sha256']
    assert bank['normalizer_sha256']==head['normalizer_sha256']==runtime.metadata['normalizer_sha256']
    assert bank['split_hash']==head['split_hash']==splits['split_hash']
    assert bank['train_ids']==head['train_ids']==splits['head_train'] and len(bank['train_ids'])==710
    assert bank['validation_ids']==head['validation_ids']==splits['head_validation'] and len(bank['validation_ids'])==39
    assert not set(bank['train_ids']+bank['validation_ids'])&set(splits['test'])
    assert head['training_seed']==head['head_seed']==head['seed']==0
    assert head['epochs_completed']==head['epochs_requested']==200 and head['input_dim']==256 and head['width']==64
    assert head['feature_backend']=='official_linearno_native_block0'
    assert sha256(head['checkpoint'])==head['head_sha256']
    history=head['training_history'];assert len(history)==200 and [r['epoch'] for r in history]==list(range(200))
    best=min(history,key=lambda r:r['validation_mse'])
    assert best['epoch']==head['best_epoch'] and best['validation_mse']==head['best_validation_mse']
    assert all(np.isfinite([r['train_mse'],r['validation_mse']]).all() for r in history)
    checked=0;rows=0;max_target_difference=0.;max_group_weight_difference=0.
    total=total_sq=0.;count=0
    for entry in bank['entries']:
        sid=entry['sample_id'];item=runtime.sample(sid);n0=coarse_count(len(item))
        if entry['split']=='train':
            raw=runtime.data_path/'preprocessed_data'/sid
            mask=np.load(raw/'surf.npy').astype(bool);values=np.asarray(np.load(raw/'y.npy')[mask,3],dtype=np.float64)
            total+=values.sum();total_sq+=np.square(values).sum();count+=len(values)
        with np.load(bank_root/entry['path']) as arrays:
            assert arrays['coarse_indices'].shape==(2,n0) and arrays['features'].shape==(len(arrays['target']),256)
            assert arrays['features'].dtype==np.float32 and np.isfinite(arrays['features']).all()
            wanted=[]
            for mask_seed in [0,1]:
                indices=rng_for(sid,mask_seed,'head_mask').choice(len(item),n0,replace=False)
                np.testing.assert_array_equal(arrays['coarse_indices'][mask_seed],indices)
                active=item.surface_mask[indices].numpy();target=item.target_phys[indices][active,3].numpy()
                wanted.append(((target-bank['target_mean'])/bank['target_std']).astype(np.float32))
                weights=arrays['weights'][arrays['group']==mask_seed]
                max_group_weight_difference=max(max_group_weight_difference,abs(float(weights.sum())-1.))
                checked+=1
            diff=float(np.max(np.abs(arrays['target']-np.concatenate(wanted))))
            max_target_difference=max(max_target_difference,diff);assert diff==0.
            rows+=len(arrays['target'])
    assert checked==1498 and max_group_weight_difference<2e-6
    mean=total/count;std=np.sqrt(max(total_sq/count-mean**2,1e-16))
    assert mean==bank['target_mean'] and std==bank['target_std']
    model=runtime.load_operator('cuda:0');context,_=install_linear_compensation(model)
    ca=LinearNOCA(model,head_path,runtime.binding(),'cuda:0')
    item=runtime.sample(splits['head_validation'][0]);indices=rng_for(item.sample_id,0).choice(len(item),coarse_count(len(item)),replace=False)
    pos=item.pos[indices][None].cuda();fx=item.x_norm[indices][None].cuda();captured=[]
    hook=model.model.blocks[0].register_forward_hook(lambda module,inputs,output:captured.append(output.detach().clone()))
    context.clear()
    with torch.no_grad():model(pos,fx)
    hook.remove();features,_=ca.feature_pass(pos,fx,len(item));feature_error=float((features-captured[0]).abs().max())
    assert feature_error==0.
    x,y,w=load_split(bank_root,bank,'validation','cuda:0')
    loaded=CarPressureHead(256,64).cuda();loaded.load_state_dict(torch.load(head['checkpoint'],map_location='cuda:0',weights_only=True));loaded.eval()
    with torch.no_grad():value=float(sum(((loaded(xx)[:,0]-yy).square()*ww).sum() for xx,yy,ww in zip(x.split(65536),y.split(65536),w.split(65536)))/len(x))
    assert abs(value-head['best_validation_mse'])<1e-6
    receipt={'passed':True,'checkpoint_sha256':runtime.metadata['checkpoint_sha256'],'head_sha256':head['head_sha256'],
             'head_seed':0,'epochs':200,'training_cases':710,'validation_cases':39,'mask_groups_checked':checked,'stored_surface_rows':rows,
             'bank_target_max_abs_difference':max_target_difference,'per_geometry_mask_weight_max_difference':max_group_weight_difference,
             'real_geometry_native_block0_max_difference':feature_error,'saved_checkpoint_validation_mse':value,
             'recorded_best_validation_mse':head['best_validation_mse'],'head_source_sha256':sha256(head_path),
             'validator_sha256':sha256(__file__),'feature_path':'official LinearNO own native first block, actual10% points'}
    write_json(ROOT/'head_acceptance.json',receipt);print(json.dumps(receipt,indent=2));lock.close()


if __name__=='__main__':main()
