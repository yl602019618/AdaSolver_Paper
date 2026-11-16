"""Fit only small pressure heads; operator features are immutable on-disk evidence."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from adasolver.inference.ca.superwing import SuperWingPressureHead
from adasolver.inference.superwing_contract import write_json
from adasolver.models.checkpoint import sha256_file


def load_bank(bank,metadata,split,device):
    rows=metadata['splits'][split]['rows']
    x=torch.empty((rows,128),device=device,dtype=torch.float32)
    y=torch.empty((rows,1),device=device,dtype=torch.float32)
    offset=0
    for shard in metadata['splits'][split]['shards']:
        path=bank/shard
        info=json.loads((path/'manifest.json').read_text())
        for key in ('operator_sha256','normalizer_sha256','split_sha256'):
            if info[key]!=metadata[key]:raise ValueError('incompatible feature bank shard')
        feature_array=np.load(path/'features.npy',mmap_mode='r')
        target_array=np.load(path/'target.npy',mmap_mode='r')
        if feature_array.ndim!=3 or feature_array.shape[1:]!=(3277,128) or target_array.shape!=feature_array.shape[:2]:
            raise ValueError('CA bank must contain actual 3277-point first-block contexts')
        features=feature_array.reshape(-1,128)
        target=target_array.reshape(-1,1)
        if len(features)!=info['rows'] or info['status']!='complete':
            raise ValueError('incomplete feature bank')
        for begin in range(0,len(features),262144):
            end=min(begin+262144,len(features))
            x[offset+begin:offset+end]=torch.from_numpy(np.array(features[begin:end])).to(device)
            y[offset+begin:offset+end]=torch.from_numpy(np.array(target[begin:end])).to(device)
        offset+=len(features)
        print(f'loaded {split} {offset}/{rows} feature rows',flush=True)
    if offset!=rows:raise ValueError('bank row count mismatch')
    return x,y


def fit_statistics(values,chunk=262144):
    total=torch.zeros(values.shape[1],device=values.device,dtype=torch.float64)
    square=torch.zeros_like(total)
    for block in values.split(chunk):
        d=block.double()
        total+=d.sum(0);square+=(d*d).sum(0)
    mean=total/len(values)
    std=(square/len(values)-mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(),std.float()


def train_heads(bank,output,*,device='cuda:4',widths=(0,4),epochs=100,patience=10,batch_size=262144,seed=0):
    bank,output=Path(bank),Path(output)
    metadata=json.loads((bank/'manifest.json').read_text())
    if metadata['status']!='complete':raise ValueError('finish feature bank before training')
    if metadata['coarse_count']!=3277 or metadata['feature_depth']!=1 or metadata['target'] not in ('Cp_GT','log1p_joint_lap_GT','log1p_cd_lap_GT','log1p_cdcl_lap_GT'):
        raise ValueError('CA training requires a supported first-block 10% scalar bank')
    if metadata['dataset_role']=='full' and (len(metadata['splits']['fit']['sample_ids'])!=23366 or
                                             len(metadata['splits']['head_val']['sample_ids'])!=519):
        raise ValueError('full CA training must cover the complete frozen fit/head_val splits')
    output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device)
    if device.type=='cuda':
        needed=sum(v['rows'] for v in metadata['splits'].values())*129*4+5*1024**3
        if torch.cuda.mem_get_info(device)[0]<needed:
            raise RuntimeError(f'feature-bank training needs about {needed/1024**3:.1f} GiB free on the selected GPU')
    started=time.perf_counter()
    x,y=load_bank(bank,metadata,'fit',device)
    vx,vy=load_bank(bank,metadata,'head_val',device)
    mean,std=fit_statistics(x)
    ym,ys=fit_statistics(y)
    for array in (x,vx):array.sub_(mean).div_(std)
    for array in (y,vy):array.sub_(ym).div_(ys)
    target_mean,target_std=float(ym[0]),float(ys[0])
    states={}
    for width in widths:
        directory=output/('linear' if width==0 else f'd{width}')
        directory.mkdir(exist_ok=True)
        if (directory/'manifest.json').exists():
            previous=json.loads((directory/'manifest.json').read_text())
            if previous.get('status')=='complete':
                expected=dict(bank_manifest_sha256=sha256_file(bank/'manifest.json'),epochs_requested=epochs,
                              batch_size=batch_size,training_seed=seed,bottleneck=width)
                if any(previous[k]!=v for k,v in expected.items()):
                    raise ValueError('completed head uses a different bank or recipe; choose a new output version')
                continue
        torch.manual_seed(seed)
        head=SuperWingPressureHead(width).to(device)
        head.feature_mean.copy_(mean);head.feature_std.copy_(std)
        states[width]=dict(head=head,optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-5),
                           best=float('inf'),stale=0,history=[],directory=directory,stopped=False,best_epoch=None)
    generator=torch.Generator(device=device).manual_seed(seed)
    for epoch in range(epochs):
        active={w:s for w,s in states.items() if not s['stopped']}
        if not active:break
        order=torch.randperm(len(x),device=device,generator=generator)
        losses={w:torch.zeros((),device=device) for w in active}
        for indices in order.split(batch_size):
            batch_x,batch_y=x[indices],y[indices]
            for width,state in active.items():
                head,opt=state['head'],state['optimizer']
                head.train()
                # Features have already been normalized with fit-only statistics.
                prediction=head.network(batch_x)
                loss=F.mse_loss(prediction,batch_y)
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                opt.step()
                losses[width]+=loss.detach()*len(indices)
        with torch.no_grad():
            for width,state in active.items():
                head=state['head'];head.eval()
                validation=torch.zeros((),device=device)
                for begin in range(0,len(vx),batch_size):
                    end=min(begin+batch_size,len(vx))
                    validation+=F.mse_loss(head.network(vx[begin:end]),vy[begin:end],reduction='sum')
                train_loss=float(losses[width]/len(x));val_loss=float(validation/len(vx))
                state['history'].append(dict(epoch=epoch,train_mse=train_loss,head_val_mse=val_loss))
                if val_loss<state['best']:
                    state.update(best=val_loss,stale=0,best_epoch=epoch)
                    torch.save(head.state_dict(),state['directory']/'best.pt')
                else:state['stale']+=1
                state['stopped']=state['stale']>=patience
                write_json(state['directory']/'history.json',state['history'])
                print(f'head d={width} epoch={epoch} train={train_loss:.6g} head_val={val_loss:.6g} best={state["best"]:.6g}',flush=True)
    result=[]
    for width,state in states.items():
        path=state['directory']
        torch.save(state['head'].state_dict(),path/'last.pt')
        manifest=dict(status='complete',dataset_role=metadata['dataset_role'],target=metadata['target'],feature_depth=1,
                      input_hidden_dim=128,physical_output_count=1,bottleneck=width,
                      architecture='linear' if width==0 else '128_d_GELU_1',
                      parameter_count=sum(p.numel() for p in state['head'].parameters()),
                      operator_sha256=metadata['operator_sha256'],normalizer_sha256=metadata['normalizer_sha256'],
                      split_sha256=metadata['split_sha256'],bank_manifest_sha256=sha256_file(bank/'manifest.json'),
                      head_sha256=sha256_file(path/'best.pt'),coarse_fraction=.1,coarse_count=3277,
                      target_mean=target_mean,target_std=target_std,feature_normalization='fit_only_per_channel_standard',
                      feature_bank_dtype='float32',training_seed=seed,train_view_seed=11,head_val_view_seed=29,
                      train_samples=len(metadata['splits']['fit']['sample_ids']),head_val_samples=len(metadata['splits']['head_val']['sample_ids']),
                      train_rows=len(x),head_val_rows=len(vx),epochs_requested=epochs,epochs_completed=len(state['history']),
                      best_epoch=state['best_epoch'],best_head_val_mse=state['best'],early_stopping_patience=patience,
                      stop_reason='head_val_patience' if state['stopped'] else 'max_epochs',
                      optimizer='AdamW',learning_rate=.001,weight_decay=1e-5,gradient_clip=1.,batch_size=batch_size,
                      operator_training=False,loss='MSE_standardized_'+metadata['target'],total_training_job_seconds=time.perf_counter()-started)
        write_json(path/'manifest.json',manifest)
        result.append(manifest)
    return result
