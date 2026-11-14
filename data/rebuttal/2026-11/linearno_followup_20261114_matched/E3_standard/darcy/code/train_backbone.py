"""Matched900/100 Darcy cohort; one frozen architecture per independent process."""
from __future__ import annotations
import argparse,json,os,platform,time,shutil
from pathlib import Path
import numpy as np
import torch
from common import OUT,CODE,Dataset,build_model,seeds,write,digest,relative_error,training_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['Transolver3','LinearNO'],required=True)
    p.add_argument('--variant',choices=['point','conv'],default='point');p.add_argument('--resume',action='store_true');args=p.parse_args()
    key='transolver3' if args.backbone=='Transolver3' else 'linearno' if args.variant=='point' else 'linearno_conv'
    if args.backbone=='Transolver3' and args.variant!='point':raise ValueError('T3 has only the registered native point class')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;seeds(0)
    directory=OUT/'checkpoints'/key;directory.mkdir(parents=True,exist_ok=True)
    preflight=json.loads((OUT/'results/interface_preflight.json').read_text());assert preflight['status']=='passed'
    protocol=json.loads((OUT/'config/protocol_lock.json').read_text())
    spec=protocol['models'][key];epochs=protocol['training']['epochs'];batch=protocol['training']['batch_size']
    data=Dataset();fit_ids=data.split['training_ids'];val_ids=data.split['validation_ids']
    xt,yt=data.arrays(fit_ids);xv,yv=data.arrays(val_ids)
    device=torch.device('cuda:0');pos=data.pos.to(device);xt,yt,xv,yv=[t.to(device) for t in [xt,yt,xv,yv]]
    mean,std=data.y_mean.to(device),data.y_std.to(device)
    model=build_model(args.backbone,conv=args.variant=='conv').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=spec['weight_decay'])
    steps=len(fit_ids)//batch;assert steps==225
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,epochs=epochs,steps_per_epoch=steps,pct_start=.3)
    generator=torch.Generator().manual_seed(0)
    first=0;best=float('inf');best_epoch=0
    resume=directory/'resume.pt'
    if args.resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        first=saved['epochs_completed'];best=saved['best_validation'];best_epoch=saved['best_epoch'];generator.set_state(saved['data_generator_state'])
    elif resume.exists() or (directory/'best.pt').exists():raise FileExistsError('Existing run requires explicit --resume')
    source_manifest={name:digest(CODE/name) for name in ['train_backbone.py','common.py']}
    archive=directory/'source_at_start';archive.mkdir(exist_ok=True)
    for name in source_manifest:shutil.copy2(CODE/name,archive/name)
    info=dict(benchmark='darcy',architecture=args.backbone,variant=spec['variant'],model_class=spec['model_class'],
        model_parameters=spec,parameter_count=sum(p.numel() for p in model.parameters()),seed=0,
        epochs=epochs,epochs_completed=first,batch_size=batch,steps_per_epoch=steps,total_optimization_steps=0,
        nfull=7225,nmin_fraction=1.,input_point_order='fixed native85x85 lattice',
        train_samples=900,validation_samples=100,test_samples=200,train_ids=fit_ids,validation_ids=val_ids,
        test_used_for_selection=False,test_history='same historical smooth2 population was previously viewed',
        normalizer=str(OUT/'data/normalizer.npz'),normalizer_sha256=digest(OUT/'data/normalizer.npz'),
        training_objective='physical relative L2 +0.1*(central-difference x relativeL2 + y relativeL2), summed over batch',
        predicted_derivative_boundary='zero Dirichlet collar as official exp_darcy.py',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=spec['weight_decay'],max_grad_norm=spec['max_grad_norm']),
        scheduler=dict(name='OneCycleLR',pct_start=.3,total_steps=epochs*steps),
        official_source_commit=protocol['official_linearno_commit'] if args.backbone=='LinearNO' else None,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     affinity=sorted(os.sched_getaffinity(0)),tf32_matmul=False,tf32_cudnn=False,
                     cudnn_deterministic=True,execution='eager FP32',data_residency='train/validation tensors uploaded once'),
        source_sha256=source_manifest,execution_status='running',checkpoint=str(directory/'best.pt'))
    start=time.perf_counter();write(directory/'manifest.json',info)
    history=(OUT/'logs'/f'{key}_history.jsonl').open('a')
    print(json.dumps(dict(event='started',model=key,pid=os.getpid(),epochs=epochs,steps_per_epoch=steps,train=900,validation=100,parameters=info['parameter_count'])),flush=True)
    for epoch in range(first,epochs):
        epoch_start=time.perf_counter();model.train();total=torch.zeros(3,device=device)
        order=torch.randperm(900,generator=generator).to(device)
        for offset in range(0,900,batch):
            ids=order[offset:offset+batch]
            prediction=model(pos[None].expand(len(ids),-1,-1),xt[ids])*std+mean
            loss,field,derivative=training_loss(prediction,yt[ids])
            if not bool(torch.isfinite(loss)):raise RuntimeError(f'nonfinite training loss at epoch{epoch}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),spec['max_grad_norm'])
            optimizer.step();scheduler.step();total+=torch.stack((loss.detach(),field.detach(),derivative.detach()))
        model.eval();values=[]
        with torch.no_grad():
            for offset in range(0,100,8):
                pred=model(pos[None].expand(len(xv[offset:offset+8]),-1,-1),xv[offset:offset+8])*std+mean
                values.append(relative_error(pred,yv[offset:offset+8]))
        validation=float(torch.cat(values).mean());components=(total/900).cpu().tolist()
        if not np.isfinite(validation):raise RuntimeError('nonfinite validation')
        improved=validation<best
        if improved:
            best=validation;best_epoch=epoch+1
            temp=directory/'best.tmp.pt';torch.save(model.state_dict(),temp);temp.replace(directory/'best.pt')
        row=dict(epoch=epoch+1,global_step=(epoch+1)*steps,train_loss=components[0],train_field_l2=components[1],
                 train_derivative_loss=components[2],validation_field_l2=validation,best_validation=best,
                 best_epoch=best_epoch,learning_rate=scheduler.get_last_lr()[0],seconds=time.perf_counter()-epoch_start)
        history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True)
        info.update(epochs_completed=epoch+1,total_optimization_steps=(epoch+1)*steps,
                    best_validation_relative_l2=best,best_epoch=best_epoch,training_wall_seconds=time.perf_counter()-start)
        write(directory/'manifest.json',info)
        write(OUT/'runtime'/f'{key}_progress.json',dict(execution_status='running',pid=os.getpid(),**row))
        if (epoch+1)%10==0 or epoch+1==epochs:
            temp=directory/'resume.tmp.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                epochs_completed=epoch+1,best_validation=best,best_epoch=best_epoch,data_generator_state=generator.get_state()),temp)
            temp.replace(resume)
    history.close();torch.save(model.state_dict(),directory/'last.pt')
    info.update(execution_status='completed',checkpoint_sha256=digest(directory/'best.pt'),
                last_checkpoint=str(directory/'last.pt'),last_checkpoint_sha256=digest(directory/'last.pt'),
                resume_checkpoint=str(resume),training_wall_seconds=time.perf_counter()-start)
    write(directory/'manifest.json',info);write(OUT/'runtime'/f'{key}_progress.json',dict(execution_status='completed',pid=os.getpid(),epochs_completed=epochs,manifest=str(directory/'manifest.json')))
    print(json.dumps(dict(event='completed',model=key,best_validation=best,epochs=epochs)),flush=True)


if __name__=='__main__':main()
