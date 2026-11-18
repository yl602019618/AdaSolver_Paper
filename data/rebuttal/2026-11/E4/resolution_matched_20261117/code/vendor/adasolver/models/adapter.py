from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .transolver3 import Model


@dataclass(frozen=True)
class Transolver3ModelSpec:
    space_dim: int = 2
    fun_dim: int = 2
    out_dim: int = 1
    hidden_dim: int = 128
    heads: int = 8
    layers: int = 8
    mlp_ratio: int = 2
    slices: int = 64
    dropout: float = 0.0
    activation: str = "gelu"
    ref: int = 8

    def build(self) -> "Transolver3Adapter":
        return Transolver3Adapter(**self.__dict__)


class Transolver3Adapter(nn.Module):
    def __init__(self, *, space_dim: int = 2, fun_dim: int = 2, out_dim: int = 1,
                 hidden_dim: int = 128, heads: int = 8, layers: int = 8,
                 mlp_ratio: int = 2, slices: int = 64, dropout: float = 0.0,
                 activation: str = "gelu", ref: int = 8):
        super().__init__()
        self.net = Model(space_dim=space_dim, fun_dim=fun_dim, out_dim=out_dim,
                         n_hidden=hidden_dim, n_head=heads, n_layers=layers,
                         mlp_ratio=mlp_ratio, slice_num=slices, dropout=dropout,
                         act=activation, ref=ref, unified_pos=False)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None) -> torch.Tensor:
        features = pos if fx is None else torch.cat((pos, fx), dim=-1)
        return self.net(features, use_checkpoint=False, input_list=False)
