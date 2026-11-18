from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


class LegacyMLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, act: str = "gelu"):
        super().__init__()
        try:
            activation = _ACTIVATIONS[act]
        except KeyError as error:
            raise ValueError(f"unsupported activation: {act}") from error
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear_post(self.linear_pre(values))


class LegacyCompensationContext:
    def __init__(self):
        self.weights: torch.Tensor | None = None

    def set(self, weights: torch.Tensor | None) -> None:
        self.weights = weights

    def clear(self) -> None:
        self.weights = None


class LegacyPhysicsAttention(nn.Module):
    """Historical Transolver irregular-mesh attention used by Plasticity allT."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 16,
                 dropout: float = 0.0, slice_num: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, points, _ = values.shape
        features = self.in_project_fx(values).reshape(
            batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        positions = self.in_project_x(values).reshape(
            batch, points, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(positions) / self.temperature)
        aggregate_weights = slice_weights
        context = getattr(self, "_compensation_context", None)
        if context is not None and context.weights is not None:
            weights = context.weights.to(device=values.device, dtype=slice_weights.dtype)
            if weights.ndim == 1:
                weights = weights[None].expand(batch, -1)
            aggregate_weights = slice_weights * weights[:, None, :, None]
        slice_norm = aggregate_weights.sum(dim=2)
        tokens = torch.einsum("bhnc,bhng->bhgc", features, aggregate_weights)
        tokens = tokens / (slice_norm[..., None] + 1e-5)
        query = self.to_q(tokens)
        key = self.to_k(tokens)
        value = self.to_v(tokens)
        attention = self.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale)
        output_tokens = torch.matmul(self.dropout(attention), value)
        output = torch.einsum("bhgc,bhng->bhnc", output_tokens, slice_weights)
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


def install_legacy_compensation(model: nn.Module) -> LegacyCompensationContext:
    context = LegacyCompensationContext()
    for module in model.modules():
        if isinstance(module, LegacyPhysicsAttention):
            module._compensation_context = context
    return context


class LegacyTransolverBlock(nn.Module):
    def __init__(self, heads: int, hidden_dim: int, dropout: float,
                 mlp_ratio: int, out_dim: int, slices: int, last_layer: bool):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = LegacyPhysicsAttention(hidden_dim, heads, hidden_dim // heads,
                                           dropout, slices)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = LegacyMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.Attn(self.ln_1(values)) + values
        values = self.mlp(self.ln_2(values)) + values
        if self.last_layer:
            return self.mlp2(self.ln_3(values))
        return values


class LegacyPlasticityTransolver(nn.Module):
    """Legacy-only allT operator matching historical state_dict names exactly."""

    def __init__(self, *, space_dim: int = 2, fun_dim: int = 1, out_dim: int = 38,
                 hidden_dim: int = 128, heads: int = 8, layers: int = 8,
                 mlp_ratio: int = 1, slices: int = 64, dropout: float = 0.0):
        super().__init__()
        self.preprocess = LegacyMLP(fun_dim + space_dim, hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            LegacyTransolverBlock(heads, hidden_dim, dropout, mlp_ratio, out_dim,
                                  slices, index == layers - 1)
            for index in range(layers)
        ])
        self.placeholder = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        if pos.ndim != 3 or fx.ndim != 3 or pos.shape[:2] != fx.shape[:2]:
            raise ValueError(f"pos/fx must align as (batch, points, channels), got {pos.shape}, {fx.shape}")
        values = self.preprocess(torch.cat((pos, fx), dim=-1))
        values = values + self.placeholder[None, None, :]
        for block in self.blocks:
            values = block(values)
        return values
