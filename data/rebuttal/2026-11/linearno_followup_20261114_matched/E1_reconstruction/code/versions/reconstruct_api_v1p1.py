"""Shared rule-preserving full-field reconstruction, with explicit plan lifetime.

Physical predictions enter this API after the backbone's fixed normalizer is
decoded. Scope A constructs a plan for every request. Only scope C may reuse a
plan, keyed by complete geometry, selected order, surface mask and method ID.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay,cKDTree
import torch
from native_car_reference import reconstruct_car_fields as reference_car_fields
from native_grid_reference import reconstruct_field as reference_grid_field

CAR_NATIVE='car_scipy_idw4_native_v1'
CAR_PLAN='car_ckdtree4_sparse_apply_v1p1'
GRID_NATIVE='grid_scipy_linear_nearest_native_v1'
GRID_PLAN='grid_qhull_sparse_apply_v1p1'
IMPLEMENTATION_VERSION='1.1'
IMPLEMENTATION_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _values(values):
    values=np.asarray(values)
    return values[:,None] if values.ndim==1 else values


def native_idw(source,values,query,k=4,eps=1e-12):
    source=np.asarray(source);query=np.asarray(query);values=_values(values)
    k=min(k,len(source))
    distances,indices=cKDTree(source).query(query,k=k)
    if k==1:return values[indices]
    weights=1/(distances+eps)
    gathered=values[indices]
    output=(gathered*weights[...,None]).sum(axis=1)/weights.sum(axis=1)[:,None]
    exact=distances<=eps;rows=np.flatnonzero(exact.any(axis=1))
    output[rows]=gathered[rows,np.argmax(exact[rows],axis=1)]
    return output


def native_reconstruct(dataset,selected,values,reference_coords,surface_mask=None):
    selected=np.asarray(selected,dtype=np.int64);values=_values(values)
    coords=np.asarray(reference_coords)
    if dataset=='shapenet_car':
        mask=np.asarray(surface_mask,dtype=bool)
        velocity,pressure,surface_rows=reference_car_fields(selected,values,coords,mask)
        output=np.zeros((len(coords),4),dtype=np.float64)
        output[:,:3]=velocity;output[surface_rows,3]=pressure
        return output
    return reference_grid_field(selected,values,coords)[0]


@dataclass
class SparseMap:
    neighbors: np.ndarray
    weights: np.ndarray
    denominator: np.ndarray
    exact_slot: np.ndarray
    target_rows: np.ndarray
    value_columns: tuple[int,...]
    timings: dict

    def apply_numpy(self,values):
        value=_values(values)[:,self.value_columns]
        gathered=value[self.neighbors]
        result=(gathered*self.weights[...,None]).sum(axis=1)/self.denominator[:,None]
        exact=np.flatnonzero(self.exact_slot>=0)
        if len(exact):result[exact]=gathered[exact,self.exact_slot[exact]]
        return result

    def to(self,device):
        return DeviceSparseMap(
            torch.as_tensor(self.neighbors,dtype=torch.long,device=device),
            torch.as_tensor(self.weights,dtype=torch.float64,device=device),
            torch.as_tensor(self.denominator,dtype=torch.float64,device=device),
            torch.as_tensor(self.exact_slot,dtype=torch.long,device=device),
            torch.as_tensor(self.target_rows,dtype=torch.long,device=device),self.value_columns)


@dataclass
class DeviceSparseMap:
    neighbors: torch.Tensor
    weights: torch.Tensor
    denominator: torch.Tensor
    exact_slot: torch.Tensor
    target_rows: torch.Tensor
    value_columns: tuple[int,...]

    def apply(self,values):
        if values.ndim==1:values=values[:,None]
        value=values[:,self.value_columns]
        gathered=value[self.neighbors].to(torch.float64)
        result=(gathered*self.weights[:,:,None]).sum(dim=1)/self.denominator[:,None]
        # All slots are valid; exact_slot=-1 selects an unused value masked out.
        slot=self.exact_slot.clamp_min(0)
        exact=gathered[torch.arange(len(slot),device=slot.device),slot]
        return torch.where((self.exact_slot>=0)[:,None],exact,result)


def idw_plan(source,query,source_rows,target_rows,columns,workers=4,k=4,eps=1e-12,profile=False):
    times={};start=time.perf_counter() if profile else 0.;tree=cKDTree(source)
    times['build_index_ms']=(time.perf_counter()-start)*1000 if profile else None
    start=time.perf_counter() if profile else 0.;distance,neighbors=tree.query(query,k=min(k,len(source)),workers=workers)
    times['neighbor_query_ms']=(time.perf_counter()-start)*1000 if profile else None
    start=time.perf_counter() if profile else 0.
    if np.ndim(distance)==1:distance=distance[:,None];neighbors=neighbors[:,None]
    weight=1/(distance+eps);den=weight.sum(axis=1)
    exact=distance<=eps;slots=np.where(exact.any(axis=1),np.argmax(exact,axis=1),-1)
    if neighbors.shape[1]==1:slots[:]=0
    neighbors=np.asarray(source_rows,dtype=np.int64)[neighbors]
    times['weight_build_ms']=(time.perf_counter()-start)*1000 if profile else None
    return SparseMap(neighbors,weight,den,slots,np.asarray(target_rows),tuple(columns),times)


def linear_plan(source,query,source_rows,columns,workers=4,profile=False):
    times={};start=time.perf_counter() if profile else 0.;tri=Delaunay(source)
    times['triangulation_ms']=(time.perf_counter()-start)*1000 if profile else None
    start=time.perf_counter() if profile else 0.;simplex=tri.find_simplex(query)
    times['simplex_query_ms']=(time.perf_counter()-start)*1000 if profile else None
    start=time.perf_counter() if profile else 0.;inside=np.flatnonzero(simplex>=0);missing=np.flatnonzero(simplex<0)
    neighbors=np.zeros((len(query),3),dtype=np.int64);weights=np.zeros((len(query),3),dtype=np.float64)
    transform=tri.transform[simplex[inside]]
    bary=np.einsum('nij,nj->ni',transform[:,:2],query[inside]-transform[:,2])
    weights[inside]=np.column_stack((bary,1-bary.sum(axis=1)))
    neighbors[inside]=np.asarray(source_rows)[tri.simplices[simplex[inside]]]
    times['barycentric_weights_ms']=(time.perf_counter()-start)*1000 if profile else None
    start=time.perf_counter() if profile else 0.
    if len(missing):
        # Same cKDTree defaults and source order as NearestNDInterpolator.
        nearest=cKDTree(source).query(query[missing],k=1,workers=workers)[1]
        neighbors[missing,0]=np.asarray(source_rows)[nearest];weights[missing,0]=1.
    times['nearest_fallback_ms']=(time.perf_counter()-start)*1000 if profile else None
    times['fallback_count']=int(len(missing))
    return SparseMap(neighbors,weights,np.ones(len(query)),np.full(len(query),-1),np.arange(len(query)),tuple(columns),times)


class ReconstructionPlan:
    def __init__(self,dataset,coords,selected,maps,channels,reconstruction_id,timings):
        self.dataset=dataset;self.reference_coords=coords;self.selected=selected
        self.maps=maps;self.channels=channels;self.reconstruction_id=reconstruction_id;self.timings=timings
        self.device_maps={}

    def apply_numpy(self,values):
        values=_values(values)
        result=np.zeros((len(self.reference_coords),self.channels),dtype=np.float64)
        for mapping in self.maps:
            result[np.ix_(mapping.target_rows,mapping.value_columns)]=mapping.apply_numpy(values)
        return result

    def prepare_device(self,device):
        key=str(torch.device(device))
        if key not in self.device_maps:self.device_maps[key]=[mapping.to(device) for mapping in self.maps]
        return self

    def apply_tensor(self,values):
        self.prepare_device(values.device)
        result=torch.zeros((len(self.reference_coords),self.channels),dtype=torch.float64,device=values.device)
        for mapping in self.device_maps[str(values.device)]:
            value=mapping.apply(values)
            result[mapping.target_rows[:,None],torch.tensor(mapping.value_columns,device=values.device)[None,:]]=value
        return result

    @property
    def cpu_bytes(self):
        return sum(a.nbytes for m in self.maps for a in [m.neighbors,m.weights,m.denominator,m.exact_slot,m.target_rows])

    def device_bytes(self):
        return sum(a.numel()*a.element_size() for maps in self.device_maps.values() for m in maps
                   for a in [m.neighbors,m.weights,m.denominator,m.exact_slot,m.target_rows])


def build_plan(dataset,reference_coords,selected_indices,*,surface_mask=None,
               reconstruction_id=None,channels=1,workers=4,device=None,profile=False):
    start=time.perf_counter() if profile else 0.;coords=np.asarray(reference_coords);selected=np.asarray(selected_indices,dtype=np.int64)
    if coords.ndim!=2 or not np.isfinite(coords).all():raise ValueError('invalid reference coordinates')
    if selected.ndim!=1 or len(selected)==0 or len(np.unique(selected))!=len(selected):raise ValueError('invalid selected indices')
    if selected.min()<0 or selected.max()>=len(coords):raise ValueError('selected index out of bounds')
    prep=(time.perf_counter()-start)*1000 if profile else None;maps=[]
    if dataset=='shapenet_car':
        if reconstruction_id not in [None,CAR_PLAN]:raise ValueError('unsupported Car plan ID')
        reconstruction_id=CAR_PLAN;channels=4;mask=np.asarray(surface_mask,dtype=bool)
        if mask.shape!=(len(coords),):raise ValueError('invalid surface mask')
        source_rows=np.flatnonzero(mask[selected]);surface_rows=np.flatnonzero(mask)
        if not len(source_rows):raise ValueError('pressure reconstruction has no surface source')
        maps.append(idw_plan(coords[selected],coords,np.arange(len(selected)),np.arange(len(coords)),(0,1,2),workers,profile=profile))
        maps.append(idw_plan(coords[selected[source_rows]],coords[surface_rows],source_rows,surface_rows,(3,),workers,profile=profile))
    else:
        if reconstruction_id not in [None,GRID_PLAN]:raise ValueError('unsupported grid plan ID')
        reconstruction_id=GRID_PLAN
        if coords.shape[1]!=2:raise ValueError('grid linear reconstruction requires two coordinates')
        maps.append(linear_plan(coords[selected],coords,np.arange(len(selected)),range(channels),workers,profile=profile))
    timings={'coordinate_index_prepare_ms':prep,'maps':[m.timings for m in maps],
             'cpu_plan_total_ms':(time.perf_counter()-start)*1000 if profile else None,'profiling_enabled':profile}
    plan=ReconstructionPlan(dataset,coords,selected,maps,channels,reconstruction_id,timings)
    if device is not None:
        if profile:torch.cuda.synchronize(device);t=time.perf_counter()
        plan.prepare_device(device)
        if profile:torch.cuda.synchronize(device);timings['map_h2d_ms']=(time.perf_counter()-t)*1000
    return plan


def plan_key(dataset,coords,selected,surface_mask,reconstruction_id,geometry_identity):
    if geometry_identity is None:raise ValueError('scope C requires an explicit physical geometry identity')
    h=hashlib.sha256((dataset+'|'+reconstruction_id+'|'+str(geometry_identity)).encode())
    for value in [coords,selected,surface_mask]:
        if value is None:h.update(b'None');continue
        value=np.ascontiguousarray(value);h.update(str((value.shape,value.dtype)).encode());h.update(value.tobytes())
    return h.hexdigest()


class PlanCache:
    """Explicit scope-C cache; never caches predictions or dynamic scores."""
    def __init__(self,max_entries=8):self.entries=OrderedDict();self.max_entries=max_entries;self.hits=0;self.misses=0
    def get(self,dataset,coords,selected,*,geometry_identity,surface_mask=None,reconstruction_id=None,**kwargs):
        rid=reconstruction_id or (CAR_PLAN if dataset=='shapenet_car' else GRID_PLAN)
        key=plan_key(dataset,coords,selected,surface_mask,rid,geometry_identity)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key],True
        self.misses+=1;plan=build_plan(dataset,coords,selected,surface_mask=surface_mask,reconstruction_id=rid,**kwargs)
        self.entries[key]=plan
        while len(self.entries)>self.max_entries:self.entries.popitem(last=False)
        return plan,False
