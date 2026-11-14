"""Create training/validation acquisition targets from each frozen backbone."""
import argparse,time,json
import numpy as np
import torch
from common import OUT,Dataset,write,digest
from ca_support import model_core
from adasolver.train.darcy_ca_head import coarse_grad_lap_score


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    key=args.backbone.lower();output=OUT/'head_banks'/key;output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError(output)
    data=Dataset();model,metadata=data.model(args.backbone,'cuda:0');assert metadata['execution_status']=='completed'
    ids=data.split['training_ids']+data.split['validation_ids'];features=[];targets=[];indices=[]
    capture={}
    def hook(module,inputs,value):capture['features']=value.detach().cpu().numpy().copy()
    handle=model_core(model,args.backbone).blocks[0].register_forward_hook(hook)
    start=time.perf_counter()
    for offset in range(0,len(ids),8):
        batch_ids=ids[offset:offset+8];x,y=data.arrays(batch_ids)
        selected=np.stack([np.random.default_rng(np.random.SeedSequence([0,int(sid),20261114])).choice(7225,722,replace=False) for sid in batch_ids])
        pos=torch.stack([data.pos[s] for s in selected]).to('cuda:0')
        fx=torch.stack([x[i,s] for i,s in enumerate(selected)]).to('cuda:0')
        prediction=model(pos,fx).cpu().numpy()
        features.extend(capture['features'])
        for s,pred in zip(selected,prediction):targets.append(coarse_grad_lap_score(data.reference,s,pred))
        indices.extend(selected)
        print('head_bank',key,offset+len(batch_ids),'/1000',flush=True)
    handle.remove()
    log_targets=np.log1p(np.asarray(targets,dtype=np.float64));mean=float(log_targets[:900].mean());std=float(log_targets[:900].std()+1e-8)
    normalized=((log_targets-mean)/std)[...,None].astype(np.float32)
    np.save(output/'features.npy',np.asarray(features,dtype=np.float32));np.save(output/'targets.npy',normalized)
    np.save(output/'coarse_indices.npy',np.asarray(indices,dtype=np.int32));np.save(output/'sample_ids.npy',np.asarray(ids,dtype=np.int64))
    write(output/'manifest.json',dict(execution_status='completed',backbone=args.backbone,
        operator_sha256=metadata['checkpoint_sha256'],normalizer_sha256=metadata['normalizer_sha256'],
        training_ids=ids[:900],validation_ids=ids[900:],test_ids_used=[],
        feature_source='own frozen backbone block0 output on selected coarse points',
        teacher='own frozen full coarse forward normalized solution; original grad_lap acquisition',
        coarse_count=722,target_mean=mean,target_std=std,target_normalizer_fit='training900 only',
        source_metric='coarse_grad_lap_score from frozen task-local code',head_seed=0,
        bank_seconds=time.perf_counter()-start,
        files={name:digest(output/name) for name in ['features.npy','targets.npy','coarse_indices.npy','sample_ids.npy']}))

if __name__=='__main__':main()
