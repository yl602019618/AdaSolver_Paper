"""Two real full-grid optimizer steps with the exact DDP training dataflow."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common import OUT,array_digest,build_model,digest,tensor_state_digest,write
from train_ddp import initialize,optimizer_step,prepare_training,validate_protocol


def main(args):
    rank,local_rank,world,device=initialize();validate_protocol();n=args.resolution
    started=time.perf_counter()
    data,x,y,xv,yv,pos,mean,std=prepare_training(n,device)
    assert x.shape==(900,n*n,1) and xv.shape==(100,n*n,1)
    assert not any('/test_' in p for p in data.accessed_files)
    # Independently fit only the explicit training arrays and compare every
    # declared normalizer value with the prepared, saved training-only values.
    original_x=x*data.x_std.to(device)+data.x_mean.to(device)
    normalized_checks={}
    for label,value,m,s in [('x',original_x,data.x_mean,data.x_std),('y',y,data.y_mean,data.y_std)]:
        empirical_mean=value.mean(dim=(0,1),keepdim=True).cpu()
        empirical_std=(value.std(dim=(0,1),keepdim=True)+1e-8).cpu()
        # CPU-vs-GPU reductions are not bitwise identical; check the saved scale
        # within FP32 reduction precision, not after examining the fit result.
        torch.testing.assert_close(empirical_mean,m,rtol=2e-6,atol=1e-8)
        torch.testing.assert_close(empirical_std,s,rtol=2e-6,atol=1e-8)
        normalized_checks[label]={'mean_absolute_difference':float((empirical_mean-m).abs().max()),
                                  'std_absolute_difference':float((empirical_std-s).abs().max()),
                                  'fit_cases':900}
    del original_x
    model=build_model().to(device);initial=tensor_state_digest(model.state_dict())
    ddp=DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,
        broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,epochs=1000,steps_per_epoch=225,pct_start=.3)
    captures=[]
    def hook(module,inputs,output):
        captures.append({'input_shape':list(inputs[0].shape),'output_shape':list(output.shape)})
    handle=model.net.preprocess.register_forward_hook(hook)
    torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    for step in range(2):
        global_ids=torch.arange(step*4,(step+1)*4,device=device)
        total,shape,norm=optimizer_step(ddp,optimizer,scheduler,x,y,pos,mean,std,global_ids,
            rank,world,args.microbatch_size,n)
        dist.all_reduce(total,op=dist.ReduceOp.SUM)
        rows.append(dict(global_step=step+1,global_case_ids=global_ids.cpu().tolist(),
                         rank_case_ids=global_ids[rank*2:(rank+1)*2].cpu().tolist(),
                         actual_prediction_shape=shape,global_loss_components=total.cpu().tolist(),
                         global_gradient_norm=norm))
    handle.remove()
    assert all(row['input_shape'][1:]==[n*n,3] and row['output_shape'][1:]==[n*n,128] for row in captures)
    assert all(row['actual_prediction_shape'][1:]==[n*n,1] for row in rows)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert tensor_state_digest(model.state_dict())!=initial
    steps={int(value['step'].item()) for value in optimizer.state.values()}
    assert steps=={2}
    model.eval()
    with torch.no_grad():
        start=rank*2;pred=model(pos[None].expand(2,-1,-1),xv[start:start+2])*std+mean
        assert pred.shape==(2,n*n,1) and bool(torch.isfinite(pred).all())
    current=dict(rank=rank,local_rank=local_rank,pid=os.getpid(),resolution=n,
        train_array_shape=list(x.shape),validation_array_shape=list(xv.shape),position_shape=list(pos.shape),
        normalization_fit_checks=normalized_checks,actual_training_capture=captures,
        validation_probe_case_ids=list(range(900+rank*2,902+rank*2)),validation_prediction_shape=list(pred.shape),
        steps=rows,optimizer_parameter_steps=2,initial_state_tensor_sha256=initial,
        updated_state_tensor_sha256=tensor_state_digest(model.state_dict()),
        parameter_count=sum(p.numel() for p in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        observed_seconds=time.perf_counter()-started,data_files_opened=data.accessed_files)
    records=[None for _ in range(world)];dist.all_gather_object(records,current)
    if rank==0:
        assert len({row['initial_state_tensor_sha256'] for row in records})==1
        assert len({row['updated_state_tensor_sha256'] for row in records})==1,'DDP replicas diverged after optimizer steps'
        for step in range(2):
            ids=[sid for row in records for sid in row['steps'][step]['rank_case_ids']]
            assert sorted(ids)==list(range(step*4,(step+1)*4))
        receipt=dict(passed=True,resolution=n,world_size=world,effective_batch_size=4,local_batch_size=2,
            microbatch_size=args.microbatch_size,gradient_accumulation_steps=2//args.microbatch_size,
            actual_full_grid_points=n*n,executed_optimizer_steps=2,
            no_test_arrays_opened=True,test_used_for_checkpoint_selection=False,
            full_training_grid_verified=True,ddp_scaling='local SUM objective multiplied by world_size before DDP average',
            protocol_sha256=digest(OUT/'protocol.json'),normalizer_sha256=data.normalizer_sha256,
            common_source_sha256=digest(__file__.replace('preflight_training.py','common.py')),
            trainer_source_sha256=digest(__file__.replace('preflight_training.py','train_ddp.py')),
            rank_records=records,scope='real native full-grid dataflow and finite two-step training check; not trained-model accuracy')
        write(OUT/f'results/training_preflight_r{n}.json',receipt)
        print(json.dumps({'preflight_passed':True,'resolution':n,'points':n*n,
            'microbatch_size':args.microbatch_size,'max_allocated_bytes':max(x['peak_allocated_bytes'] for x in records),
            'seconds':max(x['observed_seconds'] for x in records)}),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    p.add_argument('--microbatch-size',type=int,choices=[1,2],default=2);main(p.parse_args())
