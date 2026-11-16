"""Goal-oriented density influence from one 10% forward/backward pass."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import Delaunay,cKDTree
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from adasolver.inference.superwing_contract import NCOARSE,validate_indices
from adasolver.utils.interpolation import index_grid

REFERENCE=index_grid((128,256))


def interpolation_adjoint(indices,full_weights):
    """Transpose of the exact linear+nearest reconstruction, applied to Cd weights."""
    points=REFERENCE[indices]
    tri=Delaunay(points)
    simplex=tri.find_simplex(REFERENCE)
    valid=simplex>=0
    transform=tri.transform[simplex[valid]]
    bary=np.einsum('nij,nj->ni',transform[:,:2],REFERENCE[valid]-transform[:,2])
    bary=np.column_stack([bary,1-bary.sum(1)])
    vertices=tri.simplices[simplex[valid]]
    result=np.zeros((len(indices),3),dtype=np.float64)
    for j in range(3):np.add.at(result,vertices[:,j],bary[:,j,None]*full_weights[valid])
    if (~valid).any():
        _,nearest=cKDTree(points).query(REFERENCE[~valid])
        np.add.at(result,nearest,full_weights[~valid])
    return result


def density_influence(runtime,geometry,indices):
    validate_indices(indices,NCOARSE)
    started=time.perf_counter()
    weights=interpolation_adjoint(indices,geometry.weights[1])
    runtime.sync()
    with torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
        x=torch.as_tensor(geometry.features(indices),device=runtime.device)[None]
        adjoint=torch.as_tensor(weights,dtype=x.dtype,device=runtime.device)
        log_density=torch.zeros(len(indices),device=runtime.device,requires_grad=True)
        try:
            runtime.context.set(log_density.exp())
            prediction=runtime.model.net(x,input_list=False,use_checkpoint=False)[0]
            cd=torch.sum(prediction*adjoint)
            derivative=torch.autograd.grad(cd,log_density)[0]
            score=derivative.detach().abs().cpu().numpy()
        finally:runtime.context.clear()
    runtime.sync()
    if not np.isfinite(score).all():raise ValueError('non-finite Cd density influence')
    return score,time.perf_counter()-started
