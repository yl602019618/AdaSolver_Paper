"""Small scalar heads on frozen SuperWing block-0 features with real cache reuse."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from adasolver.inference.superwing_contract import NCOARSE,validate_indices,resolve
from adasolver.models.checkpoint import sha256_file


class SuperWingPressureHead(nn.Module):
    def __init__(self,bottleneck=4):
        super().__init__()
        if bottleneck not in (0,4,8,16):
            raise ValueError('SuperWing pressure head must be linear or have bottleneck 4/8/16')
        self.bottleneck=bottleneck
        self.register_buffer('feature_mean',torch.zeros(128))
        self.register_buffer('feature_std',torch.ones(128))
        self.network=nn.Linear(128,1) if bottleneck==0 else nn.Sequential(
            nn.Linear(128,bottleneck),nn.GELU(),nn.Linear(bottleneck,1))

    def forward(self,features):
        return self.network((features-self.feature_mean)/self.feature_std)


def load_head(policy,runtime,allow_diagnostic=False):
    checkpoint=resolve(policy['head_checkpoint'])
    manifest=resolve(policy['head_manifest'])
    metadata=json.loads(manifest.read_text())
    if metadata['status']!='complete' or (metadata['dataset_role']!='full' and not allow_diagnostic):
        raise ValueError('CA requires a completed head trained on the declared complete fit bank')
    if metadata['dataset_role']=='full' and (metadata['train_samples']!=23366 or metadata['train_rows']!=23366*NCOARSE):
        raise ValueError('CA head does not cover the complete deployment-matched fit bank')
    if metadata['operator_sha256']!=runtime.metadata['checkpoint_sha256'] or metadata['normalizer_sha256']!=runtime.metadata['normalizer_sha256']:
        raise ValueError('CA head belongs to a different operator or normalizer')
    if metadata['coarse_count']!=NCOARSE or metadata['coarse_fraction']!=.1 or metadata['feature_depth']!=1:
        raise ValueError('CA head requires the frozen first block at exactly 10%')
    targets=('Cp_GT','log1p_joint_lap_GT','log1p_cd_lap_GT','log1p_cdcl_lap_GT')
    if metadata['target'] not in targets or metadata['input_hidden_dim']!=128:
        raise ValueError('unsupported scalar-head target or native feature dimension')
    if policy.get('head_target','Cp_GT')!=metadata['target']:
        raise ValueError('head target differs from the acquisition policy')
    if metadata['target']!='Cp_GT':
        expected={'log1p_joint_lap_GT':'field','log1p_cd_lap_GT':'cd','log1p_cdcl_lap_GT':'cdcl'}[metadata['target']]
        proxy=(metadata['target']=='log1p_joint_lap_GT' and policy.get('head_role')=='field_variation_proxy' and
               ((policy['target']=='cd' and policy.get('cd_revision')=='v2') or
                (policy['target']=='cdcl' and policy.get('density_revision')=='v2')))
        if (policy['target']!=expected and not proxy) or policy['feature']!='direct_score':
            raise ValueError('direct score head must use its trained objective without another variation transform')
    if sha256_file(checkpoint)!=metadata['head_sha256'] or metadata['head_sha256']!=policy['head_sha256']:
        raise ValueError('CA checkpoint identity differs from its manifest/policy')
    if not np.isfinite([metadata['target_mean'],metadata['target_std']]).all() or metadata['target_std']<=0:
        raise ValueError('invalid CA target normalization')
    head=SuperWingPressureHead(metadata['bottleneck']).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint,map_location=runtime.device,weights_only=True),strict=True)
    head.eval().requires_grad_(False)
    return head,metadata


def decode_values(values,metadata):
    decoded=values*metadata['target_std']+metadata['target_mean']
    return decoded if metadata['target']=='Cp_GT' else torch.expm1(decoded.clamp(0,20))


@torch.inference_mode()
def acquire_scalar(runtime,geometry,s0,policy,heads,acquisition):
    validate_indices(s0,NCOARSE)
    key=policy['head_sha256']
    if key not in heads:
        heads[key]=load_head(policy,runtime)
    head,metadata=heads[key]
    if '_first_block' not in acquisition:
        acquisition['_first_block']=runtime.acquire_features(geometry.features(s0),capture=True)
    features,cache,block_seconds=acquisition['_first_block']
    runtime.sync()
    started=time.perf_counter()
    pressure=decode_values(head(features).reshape(-1),metadata)
    pressure=pressure.cpu().numpy()
    runtime.sync()
    if not np.isfinite(pressure).all():
        raise ValueError('non-finite learned pressure acquisition')
    return pressure,cache,block_seconds,time.perf_counter()-started
