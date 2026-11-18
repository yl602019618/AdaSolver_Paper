from __future__ import annotations

import torch
from torch import nn


class LegacyMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[act]
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers)]
        )
        self.res = res

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            y = layer(x)
            x = y + x if self.res else y
        return self.linear_post(x)


class LegacyPhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch, points, _ = x.shape
        fx_mid = self.in_project_fx(x).reshape(batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).reshape(batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3)
        weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        norm = weights.sum(2)
        token = torch.einsum("bhnc,bhng->bhgc", fx_mid, weights)
        token = token / (norm[..., None] + 1e-5)
        q = self.to_q(token)
        k = self.to_k(token)
        v = self.to_v(token)
        attention = self.dropout(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out = torch.matmul(attention, v)
        out = torch.einsum("bhgc,bhng->bhnc", out, weights)
        out = out.permute(0, 2, 1, 3).reshape(batch, points, -1)
        return self.to_out(out)


class LegacyTransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act, mlp_ratio, last_layer, out_dim, slice_num):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = LegacyPhysicsAttention(hidden_dim, num_heads, hidden_dim // num_heads, dropout, slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = LegacyMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class LegacyDarcyTransolver(nn.Module):
    """Exact unstructured historical Transolver namespace for migration only."""
    def __init__(self, space_dim=2, fun_dim=1, out_dim=1, n_hidden=128, n_heads=8,
                 n_layers=8, mlp_ratio=2, slice_num=64, dropout=0.0, act="gelu"):
        super().__init__()
        self.preprocess = LegacyMLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                    n_layers=0, res=False, act=act)
        self.blocks = nn.ModuleList([
            LegacyTransolverBlock(n_heads, n_hidden, dropout, act, mlp_ratio,
                                  i == n_layers - 1, out_dim, slice_num)
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def forward(self, pos, fx=None):
        features = pos if fx is None else torch.cat((pos, fx), -1)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden
