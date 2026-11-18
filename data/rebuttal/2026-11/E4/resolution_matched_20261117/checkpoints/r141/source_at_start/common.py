"""Resolution-generic model, data and training objective for matched Darcy."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1]
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'vendor'))
sys.dont_write_bytecode=True


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except (OSError,json.JSONDecodeError):
            if attempt==4:raise
            time.sleep(.15*(attempt+1))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    for attempt in range(5):
        try:temporary.write_text(payload);temporary.replace(path);return
        except OSError:
            if attempt==4:raise
            time.sleep(.15*(attempt+1))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_state_digest(state):
    """Stable parameter/buffer identity independent of torch-save zip metadata."""
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def configure_precision(seed=0,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def build_model():
    import adasolver
    from adasolver.models import Transolver3ModelSpec
    if not Path(adasolver.__file__).resolve().is_relative_to((CODE/'vendor').resolve()):
        raise RuntimeError('Model imported outside the frozen task-local source bundle')
    return Transolver3ModelSpec(space_dim=2,fun_dim=1,out_dim=1,hidden_dim=128,
        heads=8,layers=8,mlp_ratio=2,slices=64,dropout=0.,ref=8).build()


def relative_error(prediction,target):
    if prediction.shape!=target.shape:raise ValueError((prediction.shape,target.shape))
    p=prediction.flatten(1);t=target.flatten(1)
    return torch.linalg.vector_norm(p-t,dim=1)/torch.linalg.vector_norm(t,dim=1).clamp_min(1e-12)


def gradients(field,resolution):
    """The existing central-difference recipe, generalized with h=1/resolution."""
    n=int(resolution)
    if field.numel()%(n*n):raise ValueError('Field does not fit the declared native resolution')
    field=field.reshape(-1,n,n,1)
    padded=torch.nn.functional.pad(field,(0,0,1,1,1,1),mode='constant',value=0.)
    dx=(padded[:,1:-1,2:]-padded[:,1:-1,:-2])*(n/2)
    dy=(padded[:,2:,1:-1]-padded[:,:-2,1:-1])*(n/2)
    return dx,dy


def training_loss(prediction,target,resolution):
    n=int(resolution)
    if prediction.ndim!=3 or prediction.shape[1:]!=(n*n,1):
        raise ValueError(('Training prediction is not the complete native grid',prediction.shape,n))
    if target.shape!=prediction.shape:raise ValueError('Prediction/target mismatch')
    field=relative_error(prediction,target).sum()
    native=prediction.reshape(-1,n,n,1)
    interior=native[:,1:-1,1:-1].permute(0,3,1,2)
    boundary_zero=torch.nn.functional.pad(interior,(1,1,1,1)).permute(0,2,3,1)
    pg=gradients(boundary_zero,n);tg=gradients(target,n)
    derivative=relative_error(pg[0],tg[0]).sum()+relative_error(pg[1],tg[1]).sum()
    return field+.1*derivative,field,derivative


class ResolutionData:
    """Lazy split access: constructing training data does not open test arrays."""
    def __init__(self,resolution):
        self.resolution=int(resolution);self.npoints=self.resolution**2
        self.directory=OUT/'data'/f'r{self.resolution}'
        self.split=read(OUT/'split_manifest.json')
        if self.split['training_ids']!=list(range(900)) or self.split['validation_ids']!=list(range(900,1000)):
            raise ValueError('The matched contiguous training/validation split is not in effect')
        if self.split['test_ids']!=list(range(1000,1200)):
            raise ValueError('Unexpected common physical evaluation population')
        self.accessed_files=[]
        self.normalizer_path=self.directory/'normalizer.npz'
        with np.load(self.normalizer_path,allow_pickle=False) as norm:
            arrays={name:np.asarray(norm[name],dtype=np.float32).reshape(1,1,1).copy()
                    for name in ['x_mean','x_std','y_mean','y_std']}
        if not all(np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite normalizer')
        if arrays['x_std'].item()<=0 or arrays['y_std'].item()<=0:raise ValueError('Invalid normalizer scale')
        self.x_mean,self.x_std,self.y_mean,self.y_std=[torch.from_numpy(arrays[k]) for k in ['x_mean','x_std','y_mean','y_std']]
        self.normalizer_sha256=digest(self.normalizer_path)
        positions=np.load(self.directory/'positions.npy',allow_pickle=False)
        reference=np.load(self.directory/'reference_coords.npy',allow_pickle=False)
        assert positions.shape==reference.shape==(self.npoints,2)
        assert positions.dtype==np.float32 and reference.dtype==np.float64
        assert np.array_equal(reference,positions.astype(np.float64))
        assert len(np.unique(positions[:,0]))==len(np.unique(positions[:,1]))==self.resolution
        self.positions=torch.from_numpy(positions.copy());self.reference=reference
        self.pos=self.positions;self.ref=self.reference
        self.accessed_files.extend(map(str,[self.normalizer_path,self.directory/'positions.npy',self.directory/'reference_coords.npy']))

    def load_split(self,split):
        if split not in ['train','validation','test']:raise ValueError(split)
        count={'train':900,'validation':100,'test':200}[split]
        px=self.directory/f'{split}_coeff.npy';py=self.directory/f'{split}_solution.npy'
        coeff=np.load(px,mmap_mode='r',allow_pickle=False);truth=np.load(py,mmap_mode='r',allow_pickle=False)
        if coeff.shape!=truth.shape or coeff.shape!=(count,self.npoints,1):
            raise ValueError(('Source split does not have full native shape',px,coeff.shape,truth.shape))
        if coeff.dtype!=np.float32 or truth.dtype!=np.float32:raise ValueError('Expected FP32 physical fields')
        x=torch.from_numpy(np.array(coeff,copy=True));y=torch.from_numpy(np.array(truth,copy=True))
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):raise ValueError('Nonfinite source arrays')
        self.accessed_files.extend(map(str,[px,py]))
        return (x-self.x_mean)/self.x_std,y


def load_split(resolution,split):
    return ResolutionData(resolution).load_split(split)


def load_model(resolution,device='cpu'):
    folder=OUT/'checkpoints'/f'r{int(resolution)}'
    manifest=read(folder/'manifest.json')
    if manifest['execution_status']!='completed' or manifest['epochs_completed']!=1000:
        raise RuntimeError('Backbone is not fully trained')
    if manifest['total_optimization_steps']!=225000:raise RuntimeError('Training update count differs')
    if digest(folder/'best.pt')!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    norm=OUT/'data'/f'r{int(resolution)}'/'normalizer.npz'
    if digest(norm)!=manifest['normalizer_sha256']:raise ValueError('Normalizer changed')
    if manifest['training_ids']!=list(range(900)) or manifest['validation_ids']!=list(range(900,1000)):
        raise ValueError('Backbone belongs to another split')
    model=build_model().to(device)
    model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True),strict=True)
    return model.requires_grad_(False).eval(),manifest
