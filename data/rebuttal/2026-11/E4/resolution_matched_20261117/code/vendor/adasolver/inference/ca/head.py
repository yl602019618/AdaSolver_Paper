from __future__ import annotations

import numpy as np
import torch
from torch import nn

from adasolver.models.transolver3 import Transolver_block


class TransolverGradientHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 8, mlp_ratio: int = 2,
                 slices: int = 64, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            Transolver_block(num_heads=heads, hidden_dim=hidden_dim, dropout=0.0,
                             act="gelu", mlp_ratio=mlp_ratio, out_dim=1,
                             slice_num=slices, last_layer=(index == blocks - 1))
            for index in range(blocks)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            features = block(features)
        return features


class PipeCAHead(nn.Module):
    target_mean = 0.40099185705184937
    target_std = 0.20525811612606049

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        for parameter in self.operator.parameters():
            parameter.requires_grad_(False)
        self.head = TransolverGradientHead()

    def features(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        net = self.operator.net
        values = torch.cat((pos, fx), dim=-1)
        hidden = net.preprocess(values) + net.placeholder[None, None, :]
        return net.blocks[0](hidden)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pos, fx))

    @classmethod
    def to_gradient_score(cls, standardized: np.ndarray) -> np.ndarray:
        log_gradient = np.clip(standardized * cls.target_std + cls.target_mean, 0.0, None)
        return np.expm1(log_gradient) ** 2
