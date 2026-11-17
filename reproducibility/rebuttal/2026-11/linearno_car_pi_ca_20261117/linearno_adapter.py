"""ShapeNet-Car tensor interface and measure correction for official LinearNO.

The native graph interface receives seven normalized feature channels. This
adapter preserves that path and its placeholder; it only packs/unpacks batches.
Compensation is optional, external to the checkpoint, and acts on the key
normalization over points, not on the query softmax over feature channels.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

from einops import rearrange
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parent/'vendor/LinearNO/ShapeNetCar'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
spec = importlib.util.spec_from_file_location(
    'e3_official_linearno', VENDOR/'models/LinearAttnNeuralOperator.py')
official = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = official
spec.loader.exec_module(official)


@dataclass(frozen=True)
class LinearNOSpec:
    space_dim: int = 3
    fun_dim: int = 7  # Total preprocessed dataset channels, including xyz.
    out_dim: int = 4
    hidden_dim: int = 256
    heads: int = 8
    layers: int = 8
    mlp_ratio: int = 2
    key_ratio: int = 1
    dropout: float = 0.0

    def build(self):
        return LinearNOAdapter(self)


class LinearNOAdapter(nn.Module):
    def __init__(self, spec: LinearNOSpec):
        super().__init__()
        if (spec.space_dim, spec.fun_dim, spec.out_dim) != (3, 7, 4):
            raise ValueError('ShapeNet-Car native interface requires 3D, seven inputs and four outputs')
        self.model = official.LinearAttentionNeuralOperator(
            space_dim=spec.space_dim, fun_dim=spec.fun_dim-spec.space_dim,
            out_dim=spec.out_dim, n_hidden=spec.hidden_dim, n_head=spec.heads,
            n_layers=spec.layers, mlp_ratio=spec.mlp_ratio, key_ratio=spec.key_ratio,
            dropout=spec.dropout, Time_Input=False, unified_pos=False, isregular=False)

    def forward(self, pos, fx):
        if fx.ndim != 3 or fx.shape[-1] != 7 or pos.shape[:-1] != fx.shape[:-1]:
            raise ValueError('expected pos[B,N,3] and normalized fx[B,N,7]')
        # Official x already includes normalized xyz; raw pos is not appended
        # a second time to the seven-channel pretrained/native interface.
        return torch.stack([self.model((types.SimpleNamespace(x=x), None)) for x in fx])


class LinearCompensationContext:
    def __init__(self):
        self.weights = None
        self.log_weights = None

    def set(self, weights):
        self.weights = weights
        self.log_weights = None if weights is None else weights.log()

    def clear(self):
        self.weights = None
        self.log_weights = None


def _compensated_forward(self, x):
    weights = self._linear_compensation_context.weights
    if weights is None:
        return self._uncompensated_forward(x)
    if weights.ndim == 1:
        log_weights = self._linear_compensation_context.log_weights.to(x)[None, None, :, None]
    elif weights.ndim == 2:
        if weights.shape[0] != x.shape[0]:
            raise ValueError('batched layer compensation must match the layer batch size')
        log_weights = self._linear_compensation_context.log_weights.to(x)[:, None, :, None]
    else:
        raise ValueError('compensation must be [N] or [B,N]')
    if weights.shape[-1] != x.shape[1]:
        raise ValueError('compensation must match the point count')
    x_mid = rearrange(self.in_project_x(x), 'b n (h d) -> b h n d',
                      h=self.heads, d=self.dim_head)
    q = self.to_q(x_mid)
    k = self.to_k(x_mid)
    v = self.to_v(x_mid)
    q = torch.softmax(q/torch.clamp(self.tempreature_q, max=2, min=.1), dim=-1)
    k = torch.softmax(k/torch.clamp(self.tempreature_k, max=2, min=.1)+log_weights, dim=-2)
    kv = torch.einsum('bhnd,bhnc->bhdc', k, v)
    qkv = torch.einsum('bhnd,bhdc->bhnc', q, kv)
    return self.to_out(rearrange(qkv, 'b h n d -> b n (h d)'))


def install_linear_compensation(model):
    context = LinearCompensationContext()
    count = 0
    for module in model.modules():
        if isinstance(module, official.LinearNO):
            if hasattr(module, '_uncompensated_forward'):
                raise ValueError('LinearNO compensation is already installed')
            module._uncompensated_forward = module.forward
            module._linear_compensation_context = context
            module.forward = types.MethodType(_compensated_forward, module)
            count += 1
    if count == 0:
        raise ValueError('no official LinearNO point-softmax layer found')
    return context, count
