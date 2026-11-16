from __future__ import annotations

import types

import torch
from torch.utils.checkpoint import checkpoint

from adasolver.models.transolver3 import Physics_Attention_Irregular_Mesh


class CompensationContext:
    def __init__(self):
        self.weights: torch.Tensor | None = None

    def set(self, weights: torch.Tensor | None) -> None:
        self.weights = weights

    def clear(self) -> None:
        self.weights = None


def _weights(context, batch, count, device, dtype):
    if context.weights is None:
        return None
    weights = context.weights.to(device=device, dtype=dtype)
    if weights.ndim == 1:
        weights = weights.unsqueeze(0).expand(batch, -1)
    if weights.shape != (batch, count):
        raise ValueError("compensation must align with the complete model input")
    return weights


def slice_statistics(attention, x, compensation=None):
    """Unprojected point sums; chunk boundaries do not affect their meaning."""
    weights = attention.chunk_weights(x)
    aggregate = weights if compensation is None else weights * compensation[:, None, :, None]
    return torch.einsum("bnc,bhng->bhgc", x, aggregate), aggregate.sum(dim=2)


def project_statistics(attention, numerator, denominator, eps=1e-5):
    """Apply the feature projection AFTER normalization, exactly as dense T3.

    Including the projection bias in the numerator would multiply that bias by
    den/(den+eps). This changes nearly empty slices in trained checkpoints.
    """
    heads, dim = attention.heads, attention.dim_head
    channels = numerator.shape[-1]
    weight = attention.in_project.weight[:heads * dim].view(heads, dim, channels)
    bias = attention.in_project.bias[:heads * dim].view(heads, dim)
    states = numerator / (denominator[..., None] + eps)
    return torch.einsum("bhgc,hdc->bhgd", states, weight) + bias[None, :, None, :]


def deslice_block(block, hidden, tokens):
    residual = block.Attn.chunk_deslice_to_out(block.ln_1(hidden), tokens) + hidden
    output = block.mlp(block.ln_2(residual)) + residual
    return block.mlp2(block.ln_3(output)) if block.last_layer else output


def compensated_forward_chunks(self, fx_list, eps=1e-5, use_checkpoint=True):
    """Global attention with explicit per-chunk weights, without mutable cursors."""
    if not fx_list:
        raise ValueError("chunked inference requires at least one chunk")
    count = sum(chunk.shape[1] for chunk in fx_list)
    first = fx_list[0]
    weights = _weights(self.Attn._compensation_context, first.shape[0], count,
                       first.device, first.dtype)
    numerator = denominator = None
    start = 0
    for hidden in fx_list:
        end = start + hidden.shape[1]
        c = None if weights is None else weights[:, start:end]
        normalized = self.ln_1(hidden)

        def statistics(x, compensation):
            return slice_statistics(self.Attn, x, compensation)

        if use_checkpoint and torch.is_grad_enabled():
            num, den = checkpoint(statistics, normalized, c, use_reentrant=False)
        else:
            num, den = statistics(normalized, c)
        numerator = num if numerator is None else numerator + num
        denominator = den if denominator is None else denominator + den
        start = end
    tokens = self.Attn.slice_attend(project_statistics(self.Attn, numerator, denominator, eps))
    if use_checkpoint and torch.is_grad_enabled():
        return [checkpoint(lambda h, t: deslice_block(self, h, t), hidden, tokens,
                           use_reentrant=False) for hidden in fx_list]
    return [deslice_block(self, hidden, tokens) for hidden in fx_list]


def _aggregate(self, x: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
    compensation = _weights(self._compensation_context, x.shape[0], x.shape[1],
                             x.device, slice_weights.dtype)
    if compensation is None:
        return slice_weights
    return slice_weights * compensation[:, None, :, None]


def compensated_forward(self, x: torch.Tensor) -> torch.Tensor:
    batch, _, channels = x.shape
    heads, dim_head = self.heads, self.dim_head
    fused_weight, fused_bias = self._get_fused_weight_slice()
    slice_weights = self.chunk_weights(x, fused_weight, fused_bias)
    aggregate_weights = _aggregate(self, x, slice_weights)
    normalizer = aggregate_weights.sum(dim=2, keepdim=True) + 1e-5
    raw_states = torch.einsum("bnc,bhng->bhgc", x, aggregate_weights)
    raw_states = raw_states / normalizer.transpose(-1, -2)
    feature_weight = self.in_project.weight[:heads * dim_head].view(heads, dim_head, channels)
    feature_bias = self.in_project.bias[:heads * dim_head].view(heads, dim_head)
    tokens = torch.einsum("bhgc,hdc->bhgd", raw_states, feature_weight)
    tokens = tokens + feature_bias.view(1, heads, 1, dim_head)
    output_tokens = self.slice_attend(tokens)
    return self.chunk_deslice_to_out(x, output_tokens, slice_weights)


def compensated_chunk_stats(self, x: torch.Tensor):
    """Accumulate compensation-weighted slice statistics for chunked inference."""
    batch, _, channels = x.shape
    heads, dim_head = self.heads, self.dim_head
    fused_weight, fused_bias = self._get_fused_weight_slice()
    slice_weights = self.chunk_weights(x, fused_weight, fused_bias)
    aggregate_weights = _aggregate(self, x, slice_weights)
    den = aggregate_weights.sum(dim=2)
    x_agg = torch.einsum("bnc,bhng->bhgc", x, aggregate_weights)
    feature_weight = self.in_project.weight[:heads * dim_head].view(heads, dim_head, channels)
    feature_bias = self.in_project.bias[:heads * dim_head].view(heads, dim_head)
    num = torch.einsum("bhgc,hdc->bhgd", x_agg, feature_weight)
    num = num + den.unsqueeze(-1) * feature_bias.unsqueeze(1)
    return num, den


def _is_irregular_attention(module: torch.nn.Module) -> bool:
    if isinstance(module, Physics_Attention_Irregular_Mesh):
        return True
    return (module.__class__.__name__ == "Physics_Attention_Irregular_Mesh"
            and all(hasattr(module, name) for name in (
                "heads", "dim_head", "in_project", "in_project_slice",
                "chunk_weights", "slice_attend", "chunk_deslice_to_out")))


def install_compensation(model: torch.nn.Module) -> tuple[CompensationContext, int]:
    context = CompensationContext()
    patched = 0
    for module in model.modules():
        if _is_irregular_attention(module):
            module._compensation_context = context
            module.forward = types.MethodType(compensated_forward, module)
            module.chunk_stats = types.MethodType(compensated_chunk_stats, module)
            patched += 1
    if not patched:
        raise RuntimeError("no Physics_Attention_Irregular_Mesh modules found")
    for module in model.modules():
        if hasattr(module, "forward_chunks") and _is_irregular_attention(getattr(module, "Attn", None)):
            module.forward_chunks = types.MethodType(compensated_forward_chunks, module)
    return context, patched
