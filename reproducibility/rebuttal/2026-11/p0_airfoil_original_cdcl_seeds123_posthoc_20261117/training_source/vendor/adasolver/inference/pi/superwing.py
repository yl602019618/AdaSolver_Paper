"""Inference-free SuperWing geometry/condition priors."""
from __future__ import annotations

import numpy as np


def physics_score(geometry, prior):
    surface=geometry.surface
    normal=surface.normals
    dn=np.gradient(normal,axis=(0,1))
    curvature=np.sqrt(sum(np.sum(d*d,axis=-1) for d in dn))
    if prior=='normal_variation':
        score=curvature
    elif prior=='curvature_area':
        score=curvature*surface.area
    elif prior=='inverse_area':
        score=1/np.maximum(surface.area,1e-20)
    elif prior=='anisotropy':
        points=geometry.pos.reshape(*surface.area.shape,3)
        di,dj=np.gradient(points,axis=(0,1))
        hi,hj=np.linalg.norm(di,axis=-1),np.linalg.norm(dj,axis=-1)
        score=np.log1p(np.maximum(hi,hj)/np.maximum(np.minimum(hi,hj),1e-12))
    elif prior=='cd_pressure_sensitivity':
        score=np.abs(geometry.weights[1,:,0])
    elif prior=='cd_curvature':
        score=np.abs(geometry.weights[1,:,0])*curvature.reshape(-1)
    elif prior=='cd_joint_sensitivity':
        score=np.linalg.norm(geometry.weights[1],axis=-1)
    elif prior=='cd_pressure_weight_grad':
        # Pure-geometry control for grad(w_Cd,Cp * Cp_hat), with Cp_hat=1.
        weight=geometry.weights[1,:,0].reshape(surface.area.shape)
        di,dj=np.gradient(weight,axis=(0,1))
        score=np.sqrt(di*di+dj*dj)
    else:
        raise ValueError(f'unknown SuperWing PI prior: {prior}')
    if not np.isfinite(score).all() or np.any(score<0):
        raise ValueError('PI score must be finite and nonnegative')
    return score.reshape(-1)
