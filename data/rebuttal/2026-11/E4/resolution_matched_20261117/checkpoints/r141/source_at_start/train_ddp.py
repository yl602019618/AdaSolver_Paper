"""Fresh matched-resolution Transolver3 training with a fixed global batch.

The launcher owns GPU admission. DDP averages gradients; multiplying the local
summed loss by world_size preserves the existing global-batch SUM objective.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common import (OUT,CODE,ResolutionData,array_digest,build_model,configure_precision,
                    digest,read,relative_error,tensor_state_digest,training_loss,write)


def initialize():
    rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    configure_precision(0,4);torch.cuda.set_device(local_rank)
    if world!=2:raise ValueError('This locked study uses two ranks and global batch4')
    dist.init_process_group('nccl',timeout=dt.timedelta(minutes=15))
    return rank,local_rank,world,torch.device('cuda',local_rank)


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_torch(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def validate_protocol():
    p=read(OUT/'protocol.json');t=p['training']
    expected={'epochs':1000,'effective_batch_size':4,'world_size':2,'local_batch_size':2,
              'seed':0,'lr':.001,'weight_decay':1e-5,'max_grad_norm':.1,
              'pct_start':.3,'updates_per_epoch':225,'total_updates':225000}
    for key,value in expected.items():
        if t.get(key)!=value:raise ValueError(('Training lock mismatch',key,t.get(key),value))
    if t['optimizer']!='AdamW' or t['scheduler']!='OneCycleLR':raise ValueError('Optimizer/scheduler changed')
    return p


def prepare_training(resolution,device):
    data=ResolutionData(resolution)
    train_x,train_y=data.load_split('train');val_x,val_y=data.load_split('validation')
    assert all('/test_' not in path for path in data.accessed_files)
    arrays=[value.to(device) for value in [train_x,train_y,val_x,val_y]]
    return data,*arrays,data.positions.to(device),data.y_mean.to(device),data.y_std.to(device)


def epoch_validation(model,val_x,val_y,pos,mean,std,rank,world,batch=4):
    model.eval();start=rank*50;stop=(rank+1)*50
    total=torch.zeros(2,dtype=torch.float64,device=pos.device)
    with torch.no_grad():
        for lo in range(start,stop,batch):
            fx=val_x[lo:min(stop,lo+batch)]
            pred=model(pos[None].expand(len(fx),-1,-1),fx)*std+mean
            total[0]+=relative_error(pred,val_y[lo:min(stop,lo+batch)]).to(torch.float64).sum()
            total[1]+=len(fx)
    dist.all_reduce(total,op=dist.ReduceOp.SUM)
    assert int(total[1].item())==100
    return float((total[0]/total[1]).item())


def optimizer_step(ddp,optimizer,scheduler,x,y,pos,mean,std,global_ids,
                   rank,world,microbatch,resolution):
    local_ids=global_ids[rank*2:(rank+1)*2]
    optimizer.zero_grad(set_to_none=True)
    totals=torch.zeros(3,device=pos.device)
    prediction_shape=None
    for offset in range(0,2,microbatch):
        ids=local_ids[offset:offset+microbatch]
        synchronize=offset+microbatch>=2
        with nullcontext() if synchronize else ddp.no_sync():
            pred=ddp(pos[None].expand(len(ids),-1,-1),x[ids])*std+mean
            prediction_shape=list(pred.shape)
            loss,field,derivative=training_loss(pred,y[ids],resolution)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite physical training loss')
            (loss*world).backward()
            totals+=torch.stack((loss.detach(),field.detach(),derivative.detach()))
    norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),.1)
    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite global gradient')
    optimizer.step();scheduler.step()
    return totals,prediction_shape,float(norm.item())


def recover_history(folder,first):
    """Preserve uncheckpointed attempts, then replay from the actual resume state."""
    later=[p for p in (folder/'history').glob('epoch_*.json') if int(p.stem.split('_')[1])>first]
    if later:
        recovery=folder/'recovery'/f'before_resume_epoch{first}_{time.time_ns()}';recovery.mkdir(parents=True)
        for p in later:shutil.move(str(p),recovery/p.name)
        for name in ['best.pt','manifest.json']:
            if (folder/name).exists():shutil.copyfile(folder/name,recovery/name)
    return len(later)


def write_curve(folder):
    import csv,json
    rows=[read(p) for p in sorted((folder/'history').glob('epoch_*.json'))]
    if not rows:return
    with (folder/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/'history.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')


def train(args):
    process_started=time.perf_counter();rank,local_rank,world,device=initialize()
    protocol=validate_protocol();resolution=args.resolution
    if args.microbatch_size not in (1,2):raise ValueError('Local microbatch must divide the locked local batch2')
    folder=OUT/'checkpoints'/f'r{resolution}'
    if rank==0:
        folder.mkdir(parents=True,exist_ok=True);(folder/'history').mkdir(exist_ok=True)
    dist.barrier()
    preflight=read(OUT/f'results/training_preflight_r{resolution}.json')
    assert preflight['passed'] and preflight['resolution']==resolution
    assert preflight['microbatch_size']==args.microbatch_size
    assert preflight['protocol_sha256']==digest(OUT/'protocol.json')
    data,train_x,train_y,val_x,val_y,pos,mean,std=prepare_training(resolution,device)
    model=build_model().to(device)
    ddp=DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,
        broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,epochs=1000,
        steps_per_epoch=225,pct_start=.3)
    generator=torch.Generator().manual_seed(0)
    first=0;best=float('inf');best_epoch=0;best_state=None;previous_wall=0.;replayed_epochs=0
    initial_tensor_sha=tensor_state_digest(model.state_dict())
    initial_hashes=[None for _ in range(world)];dist.all_gather_object(initial_hashes,initial_tensor_sha)
    assert len(set(initial_hashes))==1
    source_hashes={name:digest(CODE/name) for name in ['common.py','train_ddp.py']}
    resume_path=folder/'resume.pt'
    if args.resume:
        previous=read(folder/'manifest.json') if (folder/'manifest.json').exists() else {}
        if previous.get('execution_status')=='completed':
            dist.destroy_process_group();return
        previous_wall=previous.get('training_wall_seconds',0.)
        if resume_path.exists():
            saved=torch.load(resume_path,map_location=device,weights_only=False)
            assert saved['resolution']==resolution and saved['normalizer_sha256']==data.normalizer_sha256
            assert saved['protocol_sha256']==digest(OUT/'protocol.json')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            generator.set_state(saved['data_generator_state'].cpu())
            first=saved['epochs_completed'];best=saved['best_validation'];best_epoch=saved['best_epoch']
            best_state={k:v.cpu() for k,v in saved['best_model'].items()}
            if 'rng_by_rank' in saved:
                own=saved['rng_by_rank'][rank]
                torch.set_rng_state(own['cpu'].cpu());torch.cuda.set_rng_state(own['cuda'].cpu(),device)
        else:
            assert (folder/'initial.pt').exists(),'Cannot resume without initial or resume checkpoint'
            model.load_state_dict(torch.load(folder/'initial.pt',map_location=device,weights_only=True))
        if rank==0:
            replayed_epochs=recover_history(folder,first)
            if best_state is not None:atomic_torch(folder/'best.pt',best_state)
            write_curve(folder)
    else:
        if (folder/'initial.pt').exists() or resume_path.exists() or (folder/'best.pt').exists():
            raise FileExistsError('Existing training requires --resume; no checkpoint is overwritten implicitly')
        if rank==0:atomic_torch(folder/'initial.pt',cpu_state(model))
    dist.barrier()
    manifest=None
    if rank==0:
        source_dir=folder/'source_at_start';source_dir.mkdir(exist_ok=True)
        for name in source_hashes:
            target=source_dir/name
            if target.exists() and digest(target)!=source_hashes[name]:raise RuntimeError('Training source changed across resume')
            if not target.exists():shutil.copyfile(CODE/name,target)
        manifest=dict(execution_status='running',benchmark='darcy',resolution=resolution,
            model_class='adasolver.models.adapter.Transolver3Adapter',architecture='Transolver3',
            model_config=dict(hidden_dim=128,layers=8,heads=8,slices=64,mlp_ratio=2,space_dim=2,fun_dim=1,out_dim=1,dropout=0.),
            parameter_count=sum(p.numel() for p in model.parameters()),training_seed=0,seed=0,
            trained_from_scratch=True,initial_checkpoint=str(folder/'initial.pt'),
            initial_checkpoint_sha256=digest(folder/'initial.pt'),initial_state_tensor_sha256=initial_tensor_sha,
            checkpoint=str(folder/'best.pt'),epochs=1000,epochs_completed=first,
            total_optimization_steps=first*225,updates_per_epoch=225,effective_batch_size=4,batch_size=4,
            world_size=world,local_batch_size=2,microbatch_size=args.microbatch_size,gradient_accumulation_steps=2//args.microbatch_size,
            gradient_scaling='local summed loss times world_size; DDP gradient mean equals global4-case loss sum',
            training_ids=list(range(900)),train_ids=list(range(900)),validation_ids=list(range(900,1000)),
            test_used_for_selection=False,test_opened_by_training=False,
            input_data_access=data.accessed_files,nfull=resolution*resolution,nmin_count=resolution*resolution,nmin_fraction=1.,
            train_coefficient_shape=list(train_x.shape),validation_coefficient_shape=list(val_x.shape),positions_shape=list(pos.shape),
            native_training_grid=[resolution,resolution],training_point_order='unchanged full native C-order lattice',
            normalizer=str(data.normalizer_path),normalizer_sha256=data.normalizer_sha256,
            training_objective='physical relative L2 +.1*(x and y derivative relative L2); summed overglobal4 cases',
            derivative_recipe='same native central differences with scale resolution/2; zero predicted boundary collar',
            optimizer=dict(name='AdamW',lr=.001,weight_decay=1e-5,max_grad_norm=.1),
            scheduler=dict(name='OneCycleLR',pct_start=.3,steps_per_epoch=225,total_steps=225000),
            checkpoint_selection='strict minimum mean physical relative L2 on smooth1 rows900:1000; ties retain earlier epoch',
            validation_accumulation='FP32 per-case norms, FP64 sum acrosstwo validation shards',
            protocol_sha256=digest(OUT/'protocol.json'),split_manifest_sha256=digest(OUT/'split_manifest.json'),
            source_sha256=source_hashes,resumed_from_epoch=first,replayed_uncheckpointed_epochs=replayed_epochs,
            native_model_source_sha256={str(p.relative_to(CODE)):digest(p) for p in
                [CODE/'vendor/adasolver/models/transolver3.py',CODE/'vendor/adasolver/models/adapter.py']},
            previous_recorded_training_seconds=previous_wall,
            runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cpu_affinity=sorted(os.sched_getaffinity(0)),precision='eager FP32',tf32_matmul=False,tf32_cudnn=False,
                cudnn_deterministic=True,data_residency='train and validation only, uploaded once perrank'))
        write(folder/'manifest.json',manifest)
    dist.barrier();loop_start=time.perf_counter();capture={}
    def hook(module,inputs,output):
        if not capture:
            capture.update(preprocess_input_shape=list(inputs[0].shape),preprocess_output_shape=list(output.shape))
    handle=model.net.preprocess.register_forward_hook(hook)
    try:
        for epoch in range(first,1000):
            epoch_start=time.perf_counter();ddp.train()
            order=torch.randperm(900,generator=generator)
            order_hash=array_digest(order.numpy());first_batch=order[:4].tolist()
            total=torch.zeros(3,device=device);last_shape=None;last_grad=None
            for offset in range(0,900,4):
                ids=order[offset:offset+4].to(device)
                quantities,last_shape,last_grad=optimizer_step(ddp,optimizer,scheduler,train_x,train_y,pos,mean,std,
                    ids,rank,world,args.microbatch_size,resolution)
                total+=quantities
                if capture and handle is not None:
                    assert capture['preprocess_input_shape'][1:]==[resolution*resolution,3]
                    assert capture['preprocess_output_shape'][1:]==[resolution*resolution,128]
                    handle.remove();handle=None
            dist.all_reduce(total,op=dist.ReduceOp.SUM)
            validation=epoch_validation(model,val_x,val_y,pos,mean,std,rank,world)
            if not np.isfinite(validation):raise FloatingPointError('Nonfinite validation metric')
            improved=validation<best
            if improved:
                best=validation;best_epoch=epoch+1
                if rank==0:
                    best_state=cpu_state(model);atomic_torch(folder/'best.pt',best_state)
            if rank==0:
                components=(total/900).cpu().tolist()
                row=dict(epoch=epoch+1,global_step=(epoch+1)*225,train_loss=components[0],train_field_l2=components[1],
                    train_derivative_loss=components[2],validation_field_l2=validation,best_validation=best,best_epoch=best_epoch,
                    learning_rate=scheduler.get_last_lr()[0],epoch_seconds=time.perf_counter()-epoch_start,
                    training_wall_seconds=previous_wall+time.perf_counter()-loop_start,
                    global_data_order_sha256=order_hash,first_effective_batch_ids=first_batch,
                    actual_prediction_shape=last_shape,actual_native_grid=[resolution,resolution],last_unclipped_gradient_norm=last_grad)
                write(folder/'history'/f'epoch_{epoch+1:04d}.json',row)
                manifest.update(epochs_completed=epoch+1,total_optimization_steps=(epoch+1)*225,best_validation_relative_l2=best,
                    best_epoch=best_epoch,training_wall_seconds=row['training_wall_seconds'],actual_shape_evidence=capture)
                write(folder/'manifest.json',manifest)
                write(OUT/f'runtime/r{resolution}_backbone_progress.json',dict(execution_status='running',pid=os.getpid(),resolution=resolution,**row))
                print(json.dumps(row),flush=True)
            if (epoch+1)%10==0 or epoch+1==1000:
                rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
                rng_states=[None for _ in range(world)];dist.all_gather_object(rng_states,rng)
                if rank==0:
                    atomic_torch(resume_path,dict(resolution=resolution,model=cpu_state(model),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),epochs_completed=epoch+1,best_validation=best,best_epoch=best_epoch,
                        best_model=best_state,data_generator_state=generator.get_state(),rng_by_rank=rng_states,
                        normalizer_sha256=data.normalizer_sha256,protocol_sha256=digest(OUT/'protocol.json')))
                    write_curve(folder)
            dist.barrier()
        if rank==0:
            atomic_torch(folder/'last.pt',cpu_state(model));write_curve(folder)
            histories=[read(p) for p in sorted((folder/'history').glob('epoch_*.json'))]
            assert [h['epoch'] for h in histories]==list(range(1,1001))
            assert len(histories)==1000 and histories[-1]['global_step']==225000
            assert min(h['validation_field_l2'] for h in histories)==best
            manifest.update(execution_status='completed',checkpoint_sha256=digest(folder/'best.pt'),
                best_state_tensor_sha256=tensor_state_digest(best_state),last_checkpoint=str(folder/'last.pt'),
                last_checkpoint_sha256=digest(folder/'last.pt'),resume_checkpoint=str(resume_path),
                resume_checkpoint_sha256=digest(resume_path),training_wall_seconds=previous_wall+time.perf_counter()-loop_start,
                process_wall_seconds=time.perf_counter()-process_started)
            write(folder/'manifest.json',manifest)
            write(OUT/f'runtime/r{resolution}_backbone_progress.json',dict(execution_status='completed',resolution=resolution,
                epochs_completed=1000,total_optimization_steps=225000,manifest=str(folder/'manifest.json')))
            print(json.dumps(dict(event='completed',resolution=resolution,best_validation=best,best_epoch=best_epoch)),flush=True)
        dist.barrier()
    finally:
        if handle is not None:handle.remove()
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    parser.add_argument('--microbatch-size',type=int,choices=[1,2],default=2);parser.add_argument('--resume',action='store_true')
    train(parser.parse_args())
