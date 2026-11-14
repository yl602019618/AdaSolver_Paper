#!/usr/bin/env python3
"""Optional future archived-CA GPU raw replay. Never waits or steals a busy GPU.

Prepared only; this task did not execute this script. Use a separately allocated
idle device and a new output directory. Existing CPU/GPU evidence is untouched.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    query=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid,memory,util=query[0].strip(),int(query[1]),int(query[2]);processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if memory>100 or util>0 or uuid in processes:raise SystemExit('GPU is busy; replay not started. Coordinate with its current owner.')
    if a.output.exists():raise SystemExit('Use a new output directory; prior evidence is immutable.')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);sys.dont_write_bytecode=True
    root=a.workspace.resolve();e5=root/'rebuttal_20261112/experiments/E5_training_seeds';sys.path.insert(0,str(e5/'code/vendor'))
    import torch
    from adasolver.data.shapenet_car_runtime import CarRuntime
    from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
    from adasolver.inference.cpg.shapenet_car import surface_scalar_score,normalize_cpg_score
    runtime=CarRuntime(e5/'checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json');item=runtime.sample('param0/100715345ee54d7ae38b52b4ee9d36a3');operator=runtime.load_operator('cuda:0')
    ca=ShapeNetCarCA(operator,e5/'checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json',runtime.binding(),'cuda:0',use_cuda_graph=True)
    with np.load(e5/'records/seed0/adaptive/ca_field_99d5b859e461/arrays/param0_100715345ee54d7ae38b52b4ee9d36a3_s0_b0.500.npz') as z:coarse=z['coarse_indices'];reference=z['score']
    captured={}
    def hook(module,inputs,output):captured.update(features=inputs[0].detach(),raw=output.detach())
    handle=ca.head.register_forward_hook(hook)
    with torch.no_grad():pressure,_=ca.acquire(item.pos[coarse][None].cuda(),item.x_norm[coarse][None].cuda(),len(item),capture=True)
    torch.cuda.synchronize();handle.remove()
    pressure=pressure.cpu().numpy();raw=surface_scalar_score(item.pos.numpy(),item.surface_mask.numpy(),coarse,pressure,feature='gradient',k=4,surface_band=.1);score=normalize_cpg_score(raw,method='q95',floor=.1,power=1.)
    difference=np.abs(score-reference);violations=int(np.sum(difference>1e-6+1e-5*np.abs(reference)))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'raw_trace.npz',coarse_indices=coarse,features=captured['features'][0].cpu().numpy(),head_raw=captured['raw'][0,:,0].cpu().numpy(),pressure=pressure,raw_score=raw,normalized_score=score)
    report={'status':'VERIFIED_COMPLETE' if violations==0 else 'PARTIAL','scope':'GPU acquisition raw replay only','gpu':a.gpu,'torch':torch.__version__,'atol':1e-6,'rtol':1e-5,'violations':violations,'max_abs':float(difference.max()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_sha256':runtime.metadata['checkpoint_sha256'],'head_sha256':ca.metadata['head_sha256'],'normalizer_sha256':runtime.metadata['normalizer_sha256']}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
