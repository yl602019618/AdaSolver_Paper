"""Input-dimension independent T3 first-block features and pointwise cache reuse.

Only pointwise terms are cached. Global tokens are always rebuilt on the final
set, using its final compensation. Works with the shared chunked attention path.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from adasolver.layers.compensation import project_statistics


@dataclass(frozen=True)
class FirstBlockCache:
    inputs: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    slice_weights: torch.Tensor
    operator_id: int
    parameter_versions: tuple[int, ...]


def _finish(block, hidden, normalized, weights, tokens):
    h = hidden + block.Attn.chunk_deslice_to_out(normalized, tokens, weights)
    h = h + block.mlp(block.ln_2(h))
    return block.mlp2(block.ln_3(h)) if block.last_layer else h


def _later_blocks(net, chunks):
    """Use the dense compensated path when the final field fits in one chunk."""
    if len(chunks) == 1:
        output = chunks[0]
        for block in net.blocks[1:]:
            output = block(output)
        return output
    for block in net.blocks[1:]:
        chunks = block.forward_chunks(chunks, use_checkpoint=False)
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def first_block_features(operator, features, *, capture=False):
    net = operator.net
    block = net.blocks[0]
    hidden = net.preprocess(features) + net.placeholder[None, None, :]
    normalized = block.ln_1(hidden)
    weights = block.Attn.chunk_weights(normalized)
    numerator = torch.einsum('bnc,bhng->bhgc', normalized, weights)
    denominator = weights.sum(dim=2)
    tokens = block.Attn.slice_attend(project_statistics(block.Attn, numerator, denominator))
    output = _finish(block, hidden, normalized, weights, tokens)
    cache = FirstBlockCache(features.detach(), hidden.detach(), normalized.detach(), weights.detach(), id(operator),
                            tuple(p._version for p in operator.parameters())) if capture else None
    return output, cache


def _prefix_forward(operator, features, cache, compensation, chunk_size):
    """Contiguous coarse rows avoid gathering full hidden and slice-weight arrays."""
    net, block = operator.net, operator.net.blocks[0]
    n, n0 = features.shape[1], cache.inputs.shape[1]
    context = block.Attn._compensation_context
    previous = context.weights
    context.set(compensation)
    h_chunks = []
    numerator = denominator = None

    def terms(hidden, start, end):
        if end <= n0:
            return cache.normalized[:,start:end], cache.slice_weights[:,:,start:end]
        if start >= n0:
            normalized = block.ln_1(hidden)
            return normalized, block.Attn.chunk_weights(normalized)
        normalized = block.ln_1(hidden[:,n0-start:])
        return (torch.cat((cache.normalized[:,start:n0], normalized),dim=1),
                torch.cat((cache.slice_weights[:,:,start:n0], block.Attn.chunk_weights(normalized)),dim=2))

    try:
        for start in range(0,n,chunk_size):
            end = min(start+chunk_size,n)
            if end <= n0:
                h = cache.hidden[:,start:end]
            else:
                new = net.preprocess(features[:,max(start,n0):end])+net.placeholder[None,None,:]
                h = new if start >= n0 else torch.cat((cache.hidden[:,start:n0],new),dim=1)
            normalized, weights = terms(h,start,end)
            aggregate = weights if compensation is None else weights*compensation[:,None,start:end,None]
            num = torch.einsum('bnc,bhng->bhgc',normalized,aggregate)
            den = aggregate.sum(dim=2)
            numerator = num if numerator is None else numerator+num
            denominator = den if denominator is None else denominator+den
            h_chunks.append(h)
        tokens = block.Attn.slice_attend(project_statistics(block.Attn,numerator,denominator))
        output = []
        for start,h in zip(range(0,n,chunk_size),h_chunks):
            normalized,weights = terms(h,start,start+h.shape[1])
            output.append(_finish(block,h,normalized,weights,tokens))
        del h_chunks
        return _later_blocks(net, output)
    finally:
        context.set(previous)


@torch.no_grad()
def cached_first_block_forward(operator, features, coarse_positions, cache, compensation=None, *, chunk_size=131072):
    """Final operator forward; coarse_positions map cached row order into final order."""
    if cache.operator_id != id(operator):
        raise ValueError('first-block cache belongs to another operator')
    if cache.parameter_versions != tuple(p._version for p in operator.parameters()):
        raise ValueError('operator parameters changed after cache acquisition')
    if cache.inputs.dtype != features.dtype or cache.inputs.device != features.device:
        raise ValueError('cached dtype/device differs from final inputs')
    positions = torch.as_tensor(coarse_positions, device=features.device, dtype=torch.long)
    n = features.shape[1]
    n0 = cache.inputs.shape[1]
    if positions.shape != (n0,) or n0 > n:
        raise ValueError('cached coarse positions must map uniquely into the final point set')
    prefix = torch.equal(positions,torch.arange(n0,device=features.device))
    if not prefix and (positions.unique().numel() != n0 or torch.any(positions < 0) or torch.any(positions >= n)):
        raise ValueError('cached coarse positions must map uniquely into the final point set')
    coarse_inputs = features[:,:n0] if prefix else features[:,positions]
    if not torch.equal(coarse_inputs, cache.inputs):
        raise ValueError('cached inputs differ from final coarse rows')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    net, block = operator.net, operator.net.blocks[0]
    context = getattr(block.Attn, '_compensation_context', None)
    if context is None:
        raise ValueError('cached forward requires installed compensation/chunk support')
    c = None if compensation is None else torch.as_tensor(compensation,device=features.device,dtype=features.dtype)
    if c is not None:
        if c.ndim == 1: c = c[None].expand(features.shape[0],-1)
        if c.shape != features.shape[:2]: raise ValueError('final compensation does not align with final points')
    if prefix:
        return _prefix_forward(operator,features,cache,c,chunk_size)
    cache_row = torch.full((n,), -1, device=features.device, dtype=torch.long)
    cache_row[positions] = torch.arange(n0,device=features.device)
    h_chunks = []
    global_num = global_den = None

    def terms(hidden, row_ids):
        old = row_ids >= 0
        new = ~old
        norm = torch.empty_like(hidden)
        sw = hidden.new_empty(hidden.shape[0],block.Attn.heads,hidden.shape[1],block.Attn.slice_num)
        if old.any():
            norm[:, old] = cache.normalized[:,row_ids[old]]
            sw[:,:,old] = cache.slice_weights[:,:,row_ids[old]]
        if new.any():
            norm[:,new] = block.ln_1(hidden[:,new])
            sw[:,:,new] = block.Attn.chunk_weights(norm[:,new])
        return norm, sw

    previous = context.weights
    context.set(c)
    try:
        for start in range(0,n,chunk_size):
            end = min(start+chunk_size,n)
            rows = cache_row[start:end]
            old = rows >= 0
            new = ~old
            h = cache.hidden.new_empty(features.shape[0],end-start,cache.hidden.shape[-1])
            if old.any(): h[:,old] = cache.hidden[:,rows[old]]
            if new.any(): h[:,new] = net.preprocess(features[:,start:end][:,new])+net.placeholder[None,None,:]
            normalized, sw = terms(h, rows)
            aggregate = sw if c is None else sw*c[:,None,start:end,None]
            num = torch.einsum('bnc,bhng->bhgc',normalized,aggregate)
            den = aggregate.sum(dim=2)
            global_num = num if global_num is None else global_num+num
            global_den = den if global_den is None else global_den+den
            h_chunks.append(h)
        tokens = block.Attn.slice_attend(project_statistics(block.Attn,global_num,global_den))
        output = []
        for start,h in zip(range(0,n,chunk_size),h_chunks):
            normalized, sw = terms(h,cache_row[start:start+h.shape[1]])
            output.append(_finish(block,h,normalized,sw,tokens))
        del h_chunks
        return _later_blocks(net, output)
    finally:
        context.set(previous)
