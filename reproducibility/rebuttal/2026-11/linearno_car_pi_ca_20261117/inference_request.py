"""Complete LinearNO CPU-input to CPU-field inference for frozen Car policies.

The model and mesh pressure-influence vector are prepared outside the call.
Method-specific scorer/sampler geometry, dynamic acquisition, selection,
compensation, final native forward, decoding and k4 reconstruction are inside.
Ground-truth values and force-error evaluation are not part of this callable.
"""
from common import activate_sources
activate_sources()
import hashlib
import numpy as np
import torch
from adasolver.inference.shapenet_car_contract import coarse_count, rng_for, method_config
from adasolver.inference.uniform import sample_uniform
from adasolver.inference.pi.shapenet_car import sample_car_pi_two_stage
from adasolver.inference.cpg.shapenet_car import ShapeNetCarCPG
from adasolver.inference.cpg.shapenet_car_stratified import ShapeNetCarStratified
from adasolver.metrics.shapenet_car import reconstruct_car_fields
from adasolver.utils import budget_count


class LinearNOCarRequest:
    def __init__(self,model,context,item,normalizer,config,budget,seed,*,device='cuda:0',ca=None,pressure_influence=None):
        self.model=model.eval();self.context=context;self.ca=ca
        self.pos=item.pos.cpu();self.fx=item.x_norm.cpu()
        self.coords=self.pos.numpy();self.mask=item.surface_mask.cpu().numpy()
        self.case_id=item.sample_id;self.N=len(item);self.budget=float(budget);self.seed=seed
        self.config=method_config(config['method'],**{k:v for k,v in config.items() if k!='method'})
        self.method=self.config['method'];self.keep=self.N if self.method=='fullmesh' else budget_count(self.N,self.budget)
        self.mean=normalizer.mean.cpu().numpy();self.std=normalizer.std.cpu().numpy()
        self.influence=pressure_influence;self.device=torch.device(device)
        if self.method=='ca' and ca is None:raise ValueError('matching trained LinearNO CA head required')
        if self.config.get('cd_weight') and pressure_influence is None:raise ValueError('Cd policy requires mesh-derived influence')
        if self.method=='pi' and self.config['strategy']!='two_stage':raise ValueError('expected frozen two-stage PI')
        if self.method in ['cpg','ca'] and self.config.get('strategy')!='surface_stratified':raise ValueError('expected frozen surface-stratified policy')
        self.last_indices=None;self.last_coarse_indices=None;self.last_compensation=None
        self.last_coarse_pressure=None

    def inputs(self,indices):
        return self.pos[indices][None].to(self.device),self.fx[indices][None].to(self.device)

    def predict(self,indices):
        pos,fx=self.inputs(indices)
        return self.model(pos,fx)[0].cpu().numpy()

    @torch.no_grad()
    def __call__(self):
        self.context.clear();self.last_coarse_indices=None;self.last_coarse_pressure=None
        if self.method=='fullmesh':
            indices=np.arange(self.N);physical=self.predict(indices)*self.std+self.mean
            result=physical.astype(np.float64);result[~self.mask,3]=0.
            self.last_indices=indices;self.last_compensation=None
            return result
        cfg=self.config;rng=rng_for(self.case_id,self.seed,'refinement',self.budget)
        coarse=None;cache=None
        if self.method=='uniform':
            selected=sample_uniform(self.N,self.keep,rng)
        elif self.method=='pi':
            selected=sample_car_pi_two_stage(self.coords,self.mask,self.keep,rng,
                                             r=cfg['r'],frac_extra=cfg['frac_extra'],beta=cfg['beta'])
        else:
            options={k:cfg[k] for k in ['coarse_fraction','patch_count','neighborhood','feature','normalization',
                                       'score_floor','score_power','kappa','beta','signal','surface_band']}
            scorer=ShapeNetCarCPG(self.coords,self.mask,patches=(np.zeros(self.N,dtype=int),np.ones(1)),
                                 fps_distance_backend='native' if self.method=='ca' else 'numpy',placement='fps',**options)
            sampler=ShapeNetCarStratified(self.coords,self.mask,**{k:cfg[k] for k in
                                         ['wall_scale','wall_amplitude','physics_strength','beta','placement','score_bins']})
            coarse=rng_for(self.case_id,self.seed).choice(self.N,coarse_count(self.N),replace=False)
            if self.method=='cpg':
                pressure=(self.predict(coarse)*self.std+self.mean)[:,3]
            else:
                cp,cx=self.inputs(coarse);pressure,cache=self.ca.acquire(cp,cx,self.N,capture=True)
                pressure=pressure.cpu().numpy()
            score=scorer._score(coarse,pressure,influence=self.influence if cfg['cd_weight'] else None)
            selected=sampler.refine_score(coarse,score,self.keep,rng)
            self.last_coarse_indices=coarse;self.last_coarse_pressure=pressure
        selected.validate(self.N,self.keep)
        self.last_indices=selected.indices;self.last_compensation=selected.compensation
        weights=None if self.method=='uniform' else torch.as_tensor(selected.compensation,device=self.device)
        self.context.set(weights)
        try:
            if self.method=='ca':
                pos,fx=self.inputs(selected.indices)
                prediction=self.ca.final(pos,fx,coarse,selected.indices,cache,weights)[0].cpu().numpy()
            else:prediction=self.predict(selected.indices)
        finally:self.context.clear()
        velocity,pressure,surface=reconstruct_car_fields(selected.indices,prediction,self.coords,self.mask,
                         prediction_is_normalized=True,output_mean=self.mean,output_std=self.std,k=4)
        output=np.zeros((self.N,4),dtype=np.float64);output[:,:3]=velocity;output[surface,3]=pressure
        return output

    def point_identity(self):
        digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        return {'selected_indices_sha256':digest(self.last_indices.astype(np.int64)),
                'coarse_indices_sha256':None if self.last_coarse_indices is None else digest(self.last_coarse_indices.astype(np.int64))}
