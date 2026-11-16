"""Reuse pointwise first-layer slice weights with the FINAL compensation.

Token attention, deslicing, residual MLP and later blocks always recompute.
No global coarse block output is reused as a final block output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class Block0Cache:
    hidden: torch.Tensor
    normalized: torch.Tensor
    slice_weights: torch.Tensor
    pos: torch.Tensor
    fx: torch.Tensor
    operator_id: int


@torch.no_grad()
def coarse_features(operator, pos, fx=None, *, capture=False, nfull=11271):
    # Keep the existing Airfoil default for compatibility; other adapters pass
    # their per-sample N. No benchmark-specific module belongs in this layer.
    if pos.ndim != 3 or not 2 <= pos.shape[1] <= math.floor(.1 * nfull):
        raise ValueError("coarse input must contain two points and stay within the 10% cap")
    fx = pos if fx is None else fx
    if pos.ndim != 3 or fx.shape[:2] != pos.shape[:2]:
        raise ValueError("coarse feature inputs must align on batch and points")
    net = operator.net
    hidden = net.preprocess(torch.cat((pos, fx), dim=-1)) + net.placeholder[None, None, :]
    if not capture:
        return net.blocks[0](hidden), None
    attn = net.blocks[0].Attn
    original = attn.chunk_weights
    captured = {}

    def record(x, *args, **kwargs):
        weights = original(x, *args, **kwargs)
        captured.update(normalized=x.detach(), weights=weights.detach())
        return weights

    # Capture the weights produced by the actual acquisition pass, once.
    attn.chunk_weights = record
    try:
        output = net.blocks[0](hidden)
    finally:
        attn.chunk_weights = original
    cache = Block0Cache(hidden.detach(), captured["normalized"], captured["weights"],
                        pos.detach(), fx.detach(), id(operator))
    return output, cache


@torch.no_grad()
def cached_operator_forward(operator, pos, fx, coarse_indices, final_indices,
                            cache: Block0Cache, compensation=None):
    if cache.operator_id != id(operator):
        raise ValueError("cache belongs to another operator")
    import numpy as np
    coarse = np.asarray(coarse_indices)
    final = np.asarray(final_indices)
    if len(np.unique(final)) != len(final) or len(np.unique(coarse)) != len(coarse):
        raise ValueError("cache indices must be unique")
    net, block = operator.net, operator.net.blocks[0]
    attn = block.Attn
    ncoarse = len(coarse)
    prefix = len(final) >= ncoarse and np.array_equal(final[:ncoarse], coarse)
    if prefix:
        # Refinement naturally returns S0 followed by its new points. Slices
        # avoid an O(Nfinal) Python map and repeated CUDA gather/scatter kernels.
        if not torch.equal(pos[:, :ncoarse], cache.pos) or not torch.equal(fx[:, :ncoarse], cache.fx):
            raise ValueError("cache coordinates/features do not match final coarse rows")
        if len(final) == ncoarse:
            h, x, sw = cache.hidden, cache.normalized, cache.slice_weights
        else:
            new_h = net.preprocess(torch.cat((pos[:, ncoarse:], fx[:, ncoarse:]), -1)) + net.placeholder[None, None, :]
            new_x = block.ln_1(new_h)
            h = torch.cat((cache.hidden, new_h), dim=1)
            x = torch.cat((cache.normalized, new_x), dim=1)
            sw = torch.cat((cache.slice_weights, attn.chunk_weights(new_x)), dim=2)
    else:
        lookup = {int(value): index for index, value in enumerate(final)}
        if not set(coarse).issubset(lookup):
            raise ValueError("operator cache requires S0 to be a subset of Sfinal")
        s0 = torch.tensor([lookup[int(i)] for i in coarse], device=pos.device)
        coarse_set = set(coarse)
        new = torch.tensor([i for i, value in enumerate(final) if value not in coarse_set],
                           device=pos.device, dtype=torch.long)
        if not torch.equal(pos[:, s0], cache.pos) or not torch.equal(fx[:, s0], cache.fx):
            raise ValueError("cache coordinates/features do not match final coarse rows")
        h = cache.hidden.new_empty(pos.shape[0], len(final), cache.hidden.shape[-1])
        h[:, s0] = cache.hidden
        if len(new):
            h[:, new] = net.preprocess(torch.cat((pos[:, new], fx[:, new]), -1)) + net.placeholder[None, None, :]
        x = torch.empty_like(h)
        x[:, s0] = cache.normalized
        if len(new):
            x[:, new] = block.ln_1(h[:, new])
        sw = x.new_empty(x.shape[0], attn.heads, len(final), attn.slice_num)
        sw[:, :, s0] = cache.slice_weights
        if len(new):
            sw[:, :, new] = attn.chunk_weights(x[:, new])
    aggregate = sw
    if compensation is not None:
        c = compensation.to(x.device, x.dtype)
        if c.ndim == 1:
            c = c[None].expand(x.shape[0], -1)
        if c.shape != x.shape[:2]:
            raise ValueError("final compensation must align with final points")
        aggregate = sw * c[:, None, :, None]
    # Reaggregate every point using final weights; coarse aggregated statistics
    # would be stale because the final Voronoi cells changed.
    den = aggregate.sum(dim=2, keepdim=True) + 1e-5
    states = torch.einsum("bnc,bhng->bhgc", x, aggregate) / den.transpose(-1, -2)
    heads, dim, channels = attn.heads, attn.dim_head, x.shape[-1]
    weight = attn.in_project.weight[:heads * dim].view(heads, dim, channels)
    bias = attn.in_project.bias[:heads * dim].view(heads, dim)
    tokens = torch.einsum("bhgc,hdc->bhgd", states, weight) + bias.view(1, heads, 1, dim)
    h = h + attn.chunk_deslice_to_out(x, attn.slice_attend(tokens), sw)
    h = block.mlp(block.ln_2(h)) + h
    context = getattr(attn, "_compensation_context", None)
    if compensation is not None and context is None:
        raise ValueError("cached compensated forward requires install_compensation")
    previous = context.weights if context is not None else None
    if context is not None:
        context.set(compensation)
    try:
        for later in net.blocks[1:]:
            h = later(h)
    finally:
        if context is not None:
            context.set(previous)
    return h
