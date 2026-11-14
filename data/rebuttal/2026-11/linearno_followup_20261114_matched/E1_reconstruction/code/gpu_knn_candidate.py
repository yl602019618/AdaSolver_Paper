"""Exact exhaustive GPU kNN in bounded query blocks, with native tie resolution.

This is a separately evaluated candidate, not the accepted primary mapper.
Every source point is inspected for each query, so candidate coverage does not
depend on a heuristic radius. No full Nquery-by-Nsource distance matrix is kept.
The pair budget bounds each search tile, not top-k workspace or retained maps.
Optional CPU copies are checked against the tensors before any search; those
checks and their device transfers belong to this candidate's execution cost.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial import cKDTree

CANDIDATE_VERSION='0.2-input-guards'


def _positive_integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise TypeError(f'{name} must be a positive integer')
    if value<1:raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _checked_cpu_copy(tensor,provided,name):
    if provided is None:return None
    if isinstance(provided,torch.Tensor):
        if provided.device.type!='cpu':raise ValueError(f'{name} must be a CPU coordinate copy')
        array=provided.detach().numpy()
    else:array=np.asarray(provided)
    if array.dtype!=np.float64:raise TypeError(f'{name} must retain float64 coordinate precision')
    if array.shape!=tuple(tensor.shape):raise ValueError(f'{name} shape differs from its coordinate tensor')
    if not np.isfinite(array).all():raise ValueError(f'{name} contains nonfinite coordinates')
    actual=tensor.detach().cpu().numpy()
    if not np.array_equal(array,actual):
        raise ValueError(f'{name} values or point order differ from its coordinate tensor')
    # Keep a checked snapshot for fallback; caller-owned side arrays must not
    # be able to substitute a later geometry during the search.
    return np.array(array,dtype=np.float64,copy=True,order='C')


@dataclass
class GPUKNNMap:
    indices: torch.Tensor
    distances: torch.Tensor
    effective_k: int
    metadata: dict
    epsilon: float=1e-12

    def apply(self,values):
        if values.ndim==1:values=values[:,None]
        gathered=values[self.indices].to(torch.float64)
        if self.effective_k==1:return gathered[:,0]
        weight=1/(self.distances+self.epsilon)
        result=(gathered*weight[:,:,None]).sum(1)/weight.sum(1)[:,None]
        exact=self.distances<=self.epsilon
        slot=exact.to(torch.int32).argmax(1)
        nearest=gathered[torch.arange(len(slot),device=slot.device),slot]
        return torch.where(exact.any(1)[:,None],nearest,result)


@torch.no_grad()
def build_exact_gpu_knn(source,query,*,source_cpu=None,query_cpu=None,k=4,workers=4,max_pairs=8*1024*1024):
    k=_positive_integer(k,'k');max_pairs=_positive_integer(max_pairs,'max_pairs')
    if not isinstance(source,torch.Tensor) or not isinstance(query,torch.Tensor):
        raise TypeError('source/query must be coordinate tensors')
    if source.dtype!=torch.float64 or query.dtype!=torch.float64:
        raise TypeError('GPU exact-search candidate requires explicit float64 coordinates')
    if source.ndim!=2 or query.ndim!=2 or source.shape[1]!=query.shape[1]:raise ValueError('coordinate dimensions differ')
    if source.shape[1]<1:raise ValueError('coordinate dimension must be positive')
    if source.device!=query.device:raise ValueError('source/query devices differ')
    m=len(source);n=len(query);effective=min(k,m);search_k=min(m,effective+1)
    if m<1 or n<1:raise ValueError('empty point set')
    if m>max_pairs:
        raise ValueError('max_pairs must cover at least one query over all source points; source-axis blocking is not implemented')
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(query).all()):
        raise ValueError('source/query coordinates must be finite')
    source_cpu=_checked_cpu_copy(source,source_cpu,'source_cpu')
    query_cpu=_checked_cpu_copy(query,query_cpu,'query_cpu')
    chunk=min(512,max_pairs//m);all_distance=[];all_indices=[]
    for lower in range(0,n,chunk):
        # Explicit direct Euclidean accumulation avoids x²+y²-2xy cancellation.
        distance=torch.cdist(query[lower:lower+chunk],source,p=2,compute_mode='donot_use_mm_for_euclid_dist')
        value,index=torch.topk(distance,search_k,dim=1,largest=False,sorted=True)
        all_distance.append(value);all_indices.append(index)
    distance=torch.cat(all_distance);index=torch.cat(all_indices)
    if search_k>1:
        gap=distance[:,1:]-distance[:,:-1]
        # Resolve all nearly equal selected/boundary neighbors through the
        # original SciPy selected-tree query, including duplicated coordinates.
        tolerance=1e-12+1e-12*distance[:,1:].abs()
        ambiguous=(gap<=tolerance).any(1)
    else:ambiguous=torch.zeros(n,dtype=torch.bool,device=query.device)
    bad=torch.nonzero(ambiguous).flatten().cpu().numpy()
    index=index[:,:effective].contiguous();distance=distance[:,:effective].contiguous()
    if len(bad):
        if source_cpu is None:source_cpu=source.detach().cpu().numpy()
        if query_cpu is None:query_cpu=query.detach().cpu().numpy()
        d,i=cKDTree(source_cpu).query(query_cpu[bad],k=effective,workers=workers)
        if effective==1:d=d[:,None];i=i[:,None]
        rows=torch.as_tensor(bad,device=query.device)
        distance[rows]=torch.as_tensor(d,device=query.device);index[rows]=torch.as_tensor(i,device=query.device)
    return GPUKNNMap(index,distance,effective,dict(search='exhaustive blocked direct Euclidean float64',
        candidate_version=CANDIDATE_VERSION,max_pairs_budget=max_pairs,
        query_block=chunk,source_points=m,query_points=n,max_distance_pairs=min(chunk,n)*m,
        max_distance_tile_bytes=min(chunk,n)*m*8,legacy_tie_rows=len(bad),
        tie_rule='ambiguous top-k or boundary distances delegated to original cKDTree source order',
        coordinate_dtype='float64',distance_kind='Euclidean distance, not squared distance',candidate_only=True,
        input_contract='positive integer k/pair budget; finite coordinate tensors; optional CPU copies have matching float64 dtype, shape, values and order'))
