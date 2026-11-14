"""Two-block acquisition heads with each backbone's own frozen features."""
import numpy as np
import torch
from torch import nn
import types
from common import official
from adasolver.inference.ca.head import TransolverGradientHead


def model_core(model,backbone):
    return model.net if backbone=='Transolver3' else model


class ScoreHead(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        if backbone=='Transolver3':self.body=TransolverGradientHead(hidden_dim=128,heads=8,mlp_ratio=2,slices=64,blocks=2)
        else:
            self.body=nn.Sequential(*[official.LinearNO_block(num_heads=8,hidden_dim=128,dropout=0.,
                mlp_ratio=1,key_ratio=64,last_layer=i==1,out_dim=1,args=types.SimpleNamespace(model='temp')) for i in range(2)])
    def forward(self,hidden):return self.body(hidden)


@torch.no_grad()
def frozen_features(model,backbone,pos,fx):
    core=model_core(model,backbone)
    hidden=core.preprocess(torch.cat((pos,fx),dim=-1))
    if backbone=='Transolver3':hidden=hidden+core.placeholder[None,None,:]
    return core.blocks[0](hidden)


def decode_score(prediction,mean,std):
    value=np.asarray(prediction,dtype=np.float64)
    return np.expm1(np.clip(value*std+mean,0.,None))
