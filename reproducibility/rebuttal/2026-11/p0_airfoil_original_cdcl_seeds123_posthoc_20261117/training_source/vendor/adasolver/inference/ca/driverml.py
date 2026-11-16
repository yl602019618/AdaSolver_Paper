"""Small, single-pressure CA heads over frozen DriverML first-block features."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
from adasolver.layers.first_block import first_block_features
from adasolver.models.driverml import file_sha256


class DriverMLPressureHead(nn.Module):
    def __init__(self, feature_dim=8, input_normalization='none'):
        super().__init__()
        if feature_dim not in (4,8,16):
            raise ValueError('DriverML pressure head bottleneck must be 4, 8 or 16')
        self.feature_dim = feature_dim
        if input_normalization not in ('none','standard'):
            raise ValueError('unknown CA input normalization')
        self.input_normalization=input_normalization
        if input_normalization=='standard':
            self.register_buffer('feature_mean',torch.zeros(256))
            self.register_buffer('feature_std',torch.ones(256))
        self.layers = nn.Sequential(nn.Linear(256,feature_dim),nn.GELU(),nn.Linear(feature_dim,32),nn.GELU(),nn.Linear(32,1))

    def forward(self, features):
        if self.input_normalization=='standard':
            features=(features-self.feature_mean)/self.feature_std
        return self.layers(features)


@torch.no_grad()
def acquisition_features(runtime,coarse_features,*,depth=1,capture=False):
    """Frozen prefix representation; cache remains the first block's pointwise terms."""
    if depth not in (1,2,3):
        raise ValueError('CA feature depth must be 1, 2 or 3 frozen blocks')
    features,cache=first_block_features(runtime.model,coarse_features[None].to(runtime.device),capture=capture)
    for block in runtime.model.net.blocks[1:depth]:
        features=torch.cat(block.forward_chunks(list(features.split(runtime.chunk_size,dim=1)),use_checkpoint=False),dim=1)
    return features,cache


def decode_head_values(values,metadata):
    decoded=values*metadata['target_std']+metadata['target_mean']
    if metadata['target']=='normalized_pressure':return decoded
    if metadata['target']=='normalized_log1p_cpg_lap_proxy':return torch.expm1(decoded.clamp(0,20))
    raise ValueError('unsupported CA scalar target')


def load_head(checkpoint, manifest, operator_sha256, device, *, normalizer_sha256=None,allow_diagnostic=False):
    metadata = json.loads(Path(manifest).read_text())
    if metadata.get('status')!='complete' or (metadata.get('dataset_role')!='full' and not allow_diagnostic):
        raise ValueError('CA evaluation requires a completed head with the declared full training bank')
    if metadata['operator_sha256'] != operator_sha256 or metadata['coarse_fraction'] != .1:
        raise ValueError('CA head operator or coarse context does not match this run')
    if metadata['target'] not in ('normalized_pressure','normalized_log1p_cpg_lap_proxy') or metadata['input_hidden_dim'] != 256:
        raise ValueError('unsupported DriverML CA feature/target contract')
    if metadata.get('feature_depth',1) not in (1,2,3):
        raise ValueError('unsupported DriverML CA frozen feature depth')
    if normalizer_sha256 is not None and metadata['normalizer_sha256']!=normalizer_sha256:
        raise ValueError('CA head normalization does not match the operator inputs')
    if not np.isfinite([metadata['target_mean'],metadata['target_std']]).all() or metadata['target_std']<=0:
        raise ValueError('invalid pressure head normalization')
    if file_sha256(checkpoint) != metadata['head_sha256']:
        raise ValueError('CA head checksum does not match its manifest')
    head = DriverMLPressureHead(metadata['feature_dim'],metadata.get('input_normalization','none')).to(device)
    head.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    head.eval().requires_grad_(False)
    return head, metadata


@torch.no_grad()
def acquire_pressure(runtime, head, metadata, coarse_features):
    runtime.sync();start=time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        features,cache = acquisition_features(runtime,coarse_features,depth=metadata.get('feature_depth',1),capture=True)
        runtime.sync();block_seconds=time.perf_counter()-start
        started=time.perf_counter()
        pressure = head(features).reshape(-1).float()
        pressure = decode_head_values(pressure,metadata)
        result = pressure.cpu().numpy()
        runtime.sync();head_seconds=time.perf_counter()-started
    if not np.isfinite(result).all():
        raise ValueError('non-finite CA pressure prediction')
    return result,cache,{'coarse_pressure_seconds':block_seconds,'head_seconds':head_seconds,
                        'coarse_wss_seconds':0.0,'score_seconds':0.0}
