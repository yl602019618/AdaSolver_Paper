"""Train only a small pressure MLP on frozen, deployment-matched feature rows."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from adasolver.inference.ca.driverml import DriverMLPressureHead
from adasolver.models.driverml import file_sha256


def load_bank(root,metadata,split,device):
    ids=metadata['splits'][split]
    counts=[json.loads((root/split/f'run_{rid}.json').read_text())['supervision_rows'] for rid in ids]
    x=torch.empty(sum(counts),256,device=device);y=torch.empty(sum(counts),1,device=device)
    offset=0
    feature_root=Path(metadata.get('feature_source',root))
    if not feature_root.is_absolute():
        from adasolver.inference.driverml_contract import ROOT
        feature_root=ROOT/feature_root
    for rid,count in zip(ids,counts):
        with np.load(feature_root/split/f'run_{rid}.npz') as arrays:
            features=arrays['features']
            if feature_root==root:target=arrays['target']
            else:
                with np.load(root/split/f'run_{rid}.npz') as targets:
                    if not np.array_equal(arrays['point_indices'],targets['point_indices']):
                        raise ValueError('score supervision and feature rows differ')
                    target=targets['target']
            if features.shape!=(count,256) or target.shape!=(count,):
                raise ValueError('invalid head feature-bank shape')
            x[offset:offset+count].copy_(torch.from_numpy(features).to(device))
            y[offset:offset+count,0].copy_(torch.from_numpy(target).to(device))
        offset+=count
    return x,y


def train(dataset,output,*,feature_dim=8,device='cuda:0',epochs=200,seed=0,batch_size=65536,stop_after=None,
          input_normalization='none',learning_rate=1e-3,loss_name='smooth_l1'):
    dataset,output=Path(dataset).resolve(),Path(output)
    metadata=json.loads((dataset/'manifest.json').read_text())
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate<=0 or (stop_after is not None and stop_after < 1):
        raise ValueError('head epochs, batch size and stop_after must be positive')
    if metadata['status']!='complete' or metadata['coarse_fraction']!=.1 or metadata['target'] not in ('normalized_pressure','normalized_log1p_cpg_lap_proxy'):
        raise ValueError('CA training needs a complete bank built with 10% contexts')
    if loss_name not in ('smooth_l1','mse'):raise ValueError('unknown pressure head loss')
    objective=F.smooth_l1_loss if loss_name=='smooth_l1' else F.mse_loss
    torch.manual_seed(seed);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    x,y=load_bank(dataset,metadata,'train',device)
    vx,vy=load_bank(dataset,metadata,'calibration',device)
    head=DriverMLPressureHead(feature_dim,input_normalization).to(device)
    if input_normalization=='standard':
        with torch.no_grad():
            head.feature_mean.copy_(x.mean(dim=0))
            head.feature_std.copy_(x.std(dim=0,unbiased=False).clamp_min(1e-5))
    optimizer=torch.optim.AdamW(head.parameters(),lr=learning_rate,weight_decay=1e-5)
    steps=math.ceil(len(x)/batch_size)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,total_steps=epochs*steps,pct_start=.3)
    generator=torch.Generator(device=device).manual_seed(100+seed)
    output.mkdir(parents=True,exist_ok=True)
    last=output/'last.pt';best_path=output/'head.pt';history_path=output/'history.json'
    start_epoch=0;best=float('inf');history=[]
    recipe=dict(dataset_id=metadata['dataset_id'],feature_dim=feature_dim,epochs=epochs,seed=seed,batch_size=batch_size)
    if input_normalization!='none' or learning_rate!=1e-3:
        recipe.update(input_normalization=input_normalization,learning_rate=learning_rate)
    if loss_name!='smooth_l1':recipe['loss_name']=loss_name
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False)
        if state['recipe']!=recipe: raise ValueError('resume recipe differs from the saved CA head')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['shuffle_state'].cpu());start_epoch=state['epoch']+1;best=state['best']
        history=json.loads(history_path.read_text())
    limit=epochs if stop_after is None else min(epochs,start_epoch+stop_after)
    for epoch in range(start_epoch,limit):
        head.train();started=time.perf_counter();permutation=torch.randperm(len(x),device=device,generator=generator)
        total=0.
        for begin in range(0,len(x),batch_size):
            rows=permutation[begin:begin+batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction=head(x[rows]);loss=objective(prediction,y[rows])
            if not torch.isfinite(loss): raise ValueError('non-finite head training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);optimizer.step();scheduler.step()
            total+=float(loss.detach())*len(rows)
        head.eval();val=0.
        with torch.no_grad():
            for begin in range(0,len(vx),batch_size):
                end=min(begin+batch_size,len(vx));val+=float(objective(head(vx[begin:end]),vy[begin:end]))*(end-begin)
        val/=len(vx)
        if val<best:
            best=val;torch.save(head.state_dict(),best_path)
        row=dict(epoch=epoch+1,train_loss=total/len(x),val_loss=val,best_val=best,seconds=time.perf_counter()-started)
        history.append(row);history_path.write_text(json.dumps(history,indent=2)+'\n')
        torch.save(dict(recipe=recipe,head=head.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        shuffle_state=generator.get_state(),epoch=epoch,best=best),last)
        print(json.dumps(row),flush=True)
    manifest=dict(target=metadata['target'],target_mean=metadata['target_mean'],target_std=metadata['target_std'],
                  operator_sha256=metadata['operator_sha256'],normalizer_sha256=metadata['normalizer_sha256'],
                  head_sha256=file_sha256(best_path),feature_dim=feature_dim,input_hidden_dim=256,output_dim=1,
                  architecture='pointwise_mlp_256_d_32_1',parameter_count=sum(p.numel() for p in head.parameters()),
                  coarse_fraction=.1,dataset=str(dataset.resolve()),dataset_id=metadata['dataset_id'],dataset_role=metadata['role'],
                  feature_layer=f"block{metadata.get('feature_depth',1)-1}_output",feature_depth=metadata.get('feature_depth',1),backbone_implementation='global_raw_statistics_v1',
                  train_ids=metadata['splits']['train'],validation_ids=metadata['splits']['calibration'],
                  epochs_completed=len(history),epochs_requested=epochs,seed=seed,loss=f"{loss_name}_{metadata['target']}",
                  input_normalization=input_normalization,learning_rate=learning_rate,
                  best_val=best,operator_training=False,torch=torch.__version__,status='complete' if len(history)==epochs else 'partial')
    if metadata['target']=='normalized_log1p_cpg_lap_proxy':
        manifest.update(score_kind='lap_proxy',score_neighborhood=metadata['score_neighborhood'],teacher='frozen_pressure_operator')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
