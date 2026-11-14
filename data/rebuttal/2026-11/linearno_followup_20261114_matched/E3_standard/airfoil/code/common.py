"""Airfoil pressure data and the two explicitly identified native backbones."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import hashlib
import numpy as np
import torch
from torch import nn
from einops import rearrange

OUT=Path(__file__).resolve().parents[1]
CURRENT=OUT.parents[2]
DATA=Path('/mnt/ai-material-data/shuaibinqu/NACA/data/naca')
SPLITS=OUT.parent/'split_manifest.json'
sys.path[:0]=[str(OUT/'code/vendor'),str(OUT/'code/vendor/linearno_standard')]
from adasolver.models.adapter import Transolver3Adapter
from adasolver.layers import install_compensation as install_t3_compensation
from adasolver.metrics.field import relative_l2

OFFICIAL_PATH=OUT/'code/vendor/linearno_standard/model/LinearAttnNeuralOperator.py'
spec=importlib.util.spec_from_file_location('airfoil_official_linearno',OFFICIAL_PATH)
official=importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class LinearNOAdapter(nn.Module):
    def __init__(self,variant='temp'):
        super().__init__();self.variant=variant
        self.net=official.LinearAttentionNeuralOperator(
            space_dim=2,fun_dim=0,out_dim=1,n_hidden=128,n_layers=8,n_head=8,
            mlp_ratio=1,key_ratio=64,dropout=0.,Time_Input=False,unified_pos=False,
            H=221,W=51,args=types.SimpleNamespace(model=variant))

    def forward(self,pos,fx=None):
        return self.net(pos,None)


def build_model(architecture):
    if architecture=='t3':
        return Transolver3Adapter(space_dim=2,fun_dim=2,out_dim=1,hidden_dim=128,
                                 layers=8,heads=8,mlp_ratio=2,slices=64)
    if architecture=='linearno':return LinearNOAdapter('temp')
    if architecture=='linearno_conv':return LinearNOAdapter('conv_temp')
    raise ValueError(architecture)


def feature0(model,pos):
    inputs=pos if isinstance(model,LinearNOAdapter) else torch.cat((pos,pos),dim=-1)
    hidden=model.net.preprocess(inputs)+model.net.placeholder[None,None,:]
    return model.net.blocks[0](hidden)


class LinearContext:
    def __init__(self):self.weights=None;self.log_weights=None
    def clear(self):self.weights=None;self.log_weights=None
    def set(self,weights):
        self.weights=weights;self.log_weights=None if weights is None else weights.log()


def compensated_linear_forward(self,x):
    context=self._followup_context
    if context.weights is None:return self._native_forward(x)
    mid=rearrange(self.in_project_x(x),'b n (h d) -> b h n d',h=self.heads,d=self.dim_head)
    q,k,v=self.to_q(mid),self.to_k(mid),self.to_v(mid)
    if hasattr(self,'temperature_q'):
        q=q/torch.clamp(self.temperature_q,min=.01,max=1)
        k=k/torch.clamp(self.temperature_k,min=.01,max=1)
    lw=context.log_weights.to(x)
    if lw.ndim==1:lw=lw[None,None,:,None]
    elif lw.ndim==2:lw=lw[:,None,:,None]
    else:raise ValueError('correction must be [N] or [B,N]')
    q=torch.softmax(q,dim=-1);k=torch.softmax(k+lw,dim=-2)
    kv=torch.einsum('bhnd,bhnc->bhdc',k,v)
    out=torch.einsum('bhnd,bhdc->bhnc',q,kv)
    return self.to_out(rearrange(out,'b h n d -> b n (h d)'))


def install_compensation(model):
    if not isinstance(model,LinearNOAdapter):return install_t3_compensation(model)
    if model.variant!='temp':raise ValueError('variable-point correction is limited to the official point-temp model')
    context=LinearContext();count=0
    for m in model.modules():
        if isinstance(m,official.LinearNO_temp):
            m._followup_context=context;m._native_forward=m.forward
            m.forward=types.MethodType(compensated_linear_forward,m);count+=1
    assert count==8
    return context,count


class Data:
    def __init__(self):
        self.splits=read(SPLITS)
        self.pos=np.load(OUT/'cache/coords_trainpool.npy',mmap_mode='r')
        self.pressure=np.load(OUT/'cache/pressure_trainpool.npy',mmap_mode='r')
        normalizer=np.load(OUT/'cache/normalizer.npz');self.mean=np.float32(normalizer['mean'].item());self.std=np.float32(normalizer['std'].item())
        self.normalizer_path=OUT/'cache/normalizer.npz'
        self.x=self.y=self.q=None

    def sample(self,sid):
        if sid<1000:return np.array(self.pos[sid]),np.array(self.pressure[sid])
        if self.x is None:
            self.x=np.load(DATA/'NACA_Cylinder_X.npy',mmap_mode='r');self.y=np.load(DATA/'NACA_Cylinder_Y.npy',mmap_mode='r');self.q=np.load(DATA/'NACA_Cylinder_Q.npy',mmap_mode='r')
        pos=np.stack((self.x[sid],self.y[sid]),axis=-1).reshape(-1,2).astype(np.float32)
        pressure=np.asarray(self.q[sid,3],dtype=np.float32).reshape(-1,1)
        return pos,pressure

    def decode(self,pred):
        return pred*self.std+self.mean


def load_model(architecture,device='cuda:0'):
    m=read(OUT/f'checkpoints/{architecture}/manifest.json')
    assert m['execution_status']=='completed'
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    assert sha(m['normalizer'])==m['normalizer_sha256']
    model=build_model(architecture).to(device)
    model.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=True))
    return model.requires_grad_(False).eval(),m
