"""Exact exhaustive GPU kNN in bounded query blocks, with native tie resolution.

This is a separately evaluated candidate, not the accepted primary mapper.
Every source point is inspected for each query, so candidate coverage does not
depend on a heuristic radius. No full Nquery-by-Nsource distance matrix is kept.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial import cKDTree


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
    if source.dtype!=torch.float64 or query.dtype!=torch.float64:
        raise TypeError('GPU exact-search candidate requires explicit float64 coordinates')
    if source.ndim!=2 or query.ndim!=2 or source.shape[1]!=query.shape[1]:raise ValueError('coordinate dimensions differ')
    if source.device!=query.device:raise ValueError('source/query devices differ')
    m=len(source);n=len(query);effective=min(k,m);search_k=min(m,effective+1)
    if m<1 or n<1:raise ValueError('empty point set')
    chunk=max(1,min(512,max_pairs//m));all_distance=[];all_indices=[]
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
        if source_cpu is None:source_cpu=source.cpu().numpy()
        if query_cpu is None:query_cpu=query.cpu().numpy()
        d,i=cKDTree(np.asarray(source_cpu,dtype=np.float64)).query(np.asarray(query_cpu,dtype=np.float64)[bad],k=effective,workers=workers)
        if effective==1:d=d[:,None];i=i[:,None]
        rows=torch.as_tensor(bad,device=query.device)
        distance[rows]=torch.as_tensor(d,device=query.device);index[rows]=torch.as_tensor(i,device=query.device)
    return GPUKNNMap(index,distance,effective,dict(search='exhaustive blocked direct Euclidean float64',
        query_block=chunk,source_points=m,query_points=n,max_distance_pairs=min(chunk,n)*m,
        max_distance_tile_bytes=min(chunk,n)*m*8,legacy_tie_rows=len(bad),
        tie_rule='ambiguous top-k or boundary distances delegated to original cKDTree source order',
        coordinate_dtype='float64',distance_kind='Euclidean distance, not squared distance',candidate_only=True))
