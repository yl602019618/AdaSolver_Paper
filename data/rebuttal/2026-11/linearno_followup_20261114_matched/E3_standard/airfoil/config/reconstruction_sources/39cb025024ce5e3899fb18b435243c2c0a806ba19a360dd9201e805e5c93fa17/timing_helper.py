"""Common direct-wall timing contract for reconstruction and standard tasks.

Callables own all reconstruction work and transfers required by their scope.
No method stages or CUDA-event totals are substituted for this wall clock.
"""
from __future__ import annotations
import gc
import time
import numpy as np
import torch


def timed_request(call,*,synchronize=None):
    sync=synchronize or torch.cuda.synchronize
    sync();started=time.perf_counter();output=call();sync()
    return output,(time.perf_counter()-started)*1000


def measure_paired(callables,*,scope,warmups=20,repeats=100,order_seed=20261114,
                   synchronize=None,check_output=None,guard=None,progress=None):
    """Balanced cyclic old/new/backbone ordering with all raw observations.

    Full aliases must be removed by the caller before invoking this function.
    Scope C's first observation includes cache construction; subsequent requests
    may reuse only the explicitly keyed geometry/selected-set map.
    """
    if not callables:raise ValueError('no requests supplied')
    if repeats<100 or warmups<20:raise ValueError('formal protocol requires20warmups/100repeats')
    names=list(callables);rng=np.random.default_rng(order_seed);base=list(rng.permutation(names))
    raw=[];first=[]
    if guard:guard()
    for name in names:
        output,elapsed=timed_request(callables[name],synchronize=synchronize)
        checked=check_output(name,output) if check_output else None
        first.append(dict(condition=name,scope=scope,first_request_ms=elapsed,output_check=checked))
        del output
    for kind,count in [('warmup',warmups),('formal',repeats)]:
        for repeat in range(count):
            offset=repeat%len(base);order=base[offset:]+base[:offset]
            if guard:guard()
            for position,name in enumerate(order):
                output,elapsed=timed_request(callables[name],synchronize=synchronize)
                raw.append(dict(condition=name,scope=scope,kind=kind,repeat_id=repeat,
                                order_position=position,total_ms=elapsed))
                del output
            if progress:progress(kind,repeat+1,count)
    if guard:guard()
    summaries=[]
    for name in names:
        values=np.array([r['total_ms'] for r in raw if r['condition']==name and r['kind']=='formal'])
        assert len(values)==repeats
        summaries.append(dict(condition=name,scope=scope,repeats=repeats,
                              median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                              mean_ms=float(values.mean()),sample_sd_ms=float(values.std(ddof=1))))
    return dict(first_requests=first,raw=raw,summary=summaries,
                order=dict(base=base,rule='cyclic rotation; no parity reversal',seed=order_seed),
                timing_scope='wall time with GPU synchronization at both boundaries')


def profile_components(components,*,synchronize=None,repeats=5):
    """Separate instrumented passes; values must not be summed as direct total."""
    rows=[]
    for repeat in range(repeats):
        for name,call in components.items():
            output,elapsed=timed_request(call,synchronize=synchronize)
            rows.append(dict(profile_repeat=repeat,component=name,ms=elapsed,
                             timing_scope='separate instrumented pass'))
            del output
    return rows


def isolated_memory(call,*,warmups=2,repeats=2,synchronize=None,nvml_post_sample=None):
    """Caller must load only one backbone/method in this process.

    PyTorch peaks are exact for its allocator. Optional NVML values are samples
    after synchronized requests, not a claim to capture transient external peaks.
    """
    sync=synchronize or torch.cuda.synchronize
    for _ in range(warmups):output=call();sync();del output
    gc.collect();sync();torch.cuda.reset_peak_memory_stats()
    observations=[]
    for _ in range(repeats):
        output=call();sync()
        observations.append(nvml_post_sample() if nvml_post_sample else None)
        del output
    result=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                nvml_post_samples=observations,
                allocator_scope='PyTorch CUDA allocations, including returned GPU output and resident cached maps',
                external_scope='SciPy uses CPU memory; CUDA context/library allocations are outside PyTorch. NVML samples do not certify all transient external peaks.')
    return result
