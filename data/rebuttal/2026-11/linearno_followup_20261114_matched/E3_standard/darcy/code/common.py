"""Locked Darcy runtime; fits and model identities are local to this cohort."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

OUT=Path(__file__).resolve().parents[1]
FOLLOWUP=OUT.parents[1]
CODE=Path(__file__).resolve().parent
sys.path[:0]=[str(CODE/'vendor'),str(CODE/'vendor/LinearNOStandard')]
from adasolver.models import Transolver3ModelSpec
from adasolver.layers import install_compensation

source=CODE/'vendor/LinearNOStandard/model/LinearAttnNeuralOperator.py'
spec=importlib.util.spec_from_file_location('darcy_official_linearno',source)
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def seeds(seed=0):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def build_model(backbone,*,conv=False):
    if backbone=='Transolver3':
        return Transolver3ModelSpec(space_dim=2,fun_dim=1,out_dim=1,hidden_dim=128,
            heads=8,layers=8,mlp_ratio=2,slices=64,dropout=0.,ref=8).build()
    if backbone!='LinearNO':raise ValueError(backbone)
    return official.LinearAttentionNeuralOperator(space_dim=2,fun_dim=1,out_dim=1,
        n_hidden=128,n_head=8,n_layers=8,mlp_ratio=1,key_ratio=64,dropout=0.,
        Time_Input=False,unified_pos=False,H=85,W=85,
        args=types.SimpleNamespace(model='conv_temp' if conv else 'temp'))


class LinearContext:
    def __init__(self):self.weights=None;self.log_weights=None
    def set(self,weights):
        self.weights=weights;self.log_weights=None if weights is None else weights.log()
    def clear(self):self.weights=None;self.log_weights=None


def compensated_linear_forward(self,x):
    context=self._measure_context
    if context.weights is None:return self._native_forward(x)
    mid=rearrange(self.in_project_x(x),'b n (h d) -> b h n d',h=self.heads,d=self.dim_head)
    q,k,v=self.to_q(mid),self.to_k(mid),self.to_v(mid)
    q=torch.softmax(q/torch.clamp(self.temperature_q,min=.01,max=1),dim=-1)
    logits=k/torch.clamp(self.temperature_k,min=.01,max=1)
    log=context.log_weights.to(x)
    if log.ndim==1:log=log[None,None,:,None]
    elif log.ndim==2:log=log[:,None,:,None]
    else:raise ValueError('weights must align to [N] or [B,N]')
    k=torch.softmax(logits+log,dim=-2)
    kv=torch.einsum('bhnd,bhnc->bhdc',k,v)
    output=torch.einsum('bhnd,bhdc->bhnc',q,kv)
    return self.to_out(rearrange(output,'b h n d -> b n (h d)'))


def install(backbone,model):
    if backbone=='Transolver3':return install_compensation(model)[0]
    context=LinearContext();count=0
    for layer in model.modules():
        if isinstance(layer,official.LinearNO_temp):
            layer._native_forward=layer.forward;layer._measure_context=context
            layer.forward=types.MethodType(compensated_linear_forward,layer);count+=1
    if count!=8:raise ValueError(f'expected8 point-temp blocks, got{count}')
    return context


def relative_error(pred,target):
    pred=pred.flatten(1);target=target.flatten(1)
    return torch.linalg.vector_norm(pred-target,dim=1)/torch.linalg.vector_norm(target,dim=1).clamp_min(1e-12)


def gradients(field):
    """Official Darcy central-difference convention, h=1/85 and zero padding."""
    field=field.reshape(-1,85,85,1)
    padded=torch.nn.functional.pad(field,(0,0,1,1,1,1),mode='constant',value=0.)
    dx=(padded[:,1:-1,2:]-padded[:,1:-1,:-2])*(85/2)
    dy=(padded[:,2:,1:-1]-padded[:,:-2,1:-1])*(85/2)
    return dx,dy


def training_loss(prediction,target):
    field=relative_error(prediction,target).sum()
    predicted_grid=prediction.reshape(-1,85,85,1)
    interior=predicted_grid[:,1:-1,1:-1].permute(0,3,1,2)
    boundary_zero=torch.nn.functional.pad(interior,(1,1,1,1)).permute(0,2,3,1)
    pg=gradients(boundary_zero);tg=gradients(target)
    derivative=relative_error(pg[0],tg[0]).sum()+relative_error(pg[1],tg[1]).sum()
    return field+.1*derivative,field,derivative


class Dataset:
    def __init__(self):
        self.split=read(FOLLOWUP/'standard_benchmarks/split_manifest.json')
        self.normalizer=np.load(OUT/'data/normalizer.npz')
        self.x_mean=torch.from_numpy(self.normalizer['x_mean'].copy())
        self.x_std=torch.from_numpy(self.normalizer['x_std'].copy())
        self.y_mean=torch.from_numpy(self.normalizer['y_mean'].copy())
        self.y_std=torch.from_numpy(self.normalizer['y_std'].copy())
        self.pos=torch.from_numpy(np.load(OUT/'data/positions.npy'))
        self.reference=np.load(OUT/'data/reference_coordinates.npy')
        self.pool_x=np.load(OUT/'data/pool_coeff.npy',mmap_mode='r')
        self.pool_y=np.load(OUT/'data/pool_solution.npy',mmap_mode='r')
        self.test_x=np.load(OUT/'data/test_coeff.npy',mmap_mode='r')
        self.test_y=np.load(OUT/'data/test_solution.npy',mmap_mode='r')

    def arrays(self,ids):
        x=[];y=[]
        for sid in ids:
            source_x,source_y,index=(self.pool_x,self.pool_y,sid) if sid<1000 else (self.test_x,self.test_y,sid-1000)
            x.append(np.asarray(source_x[index]));y.append(np.asarray(source_y[index]))
        xx=torch.from_numpy(np.stack(x).copy());yy=torch.from_numpy(np.stack(y).copy())
        return (xx-self.x_mean)/self.x_std,yy

    def sample(self,sid):
        x,y=self.arrays([int(sid)]);return self.pos,x[0],y[0]

    def model(self,backbone,device='cpu'):
        directory=OUT/'checkpoints'/backbone.lower()
        manifest=read(directory/'manifest.json')
        path=directory/'best.pt'
        if digest(path)!=manifest['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
        model=build_model(backbone).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        return model.requires_grad_(False).eval(),manifest
