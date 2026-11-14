"""Native E3/E5 Car policies with only the reconstruction execution exchanged."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1];CURRENT=OUT.parents[1]
E3=CURRENT/'rebuttal_20261112/experiments/E3_backbones'
E5=CURRENT/'rebuttal_20261112/experiments/E5_training_seeds'
sys.path[:0]=[str(E3/'code'),str(E3/'code/vendor'),str(CURRENT/'AdaSolver_official')]
from adasolver.data.shapenet_car_runtime import CarRuntime
from evaluate_linearno import LinearNORuntime
from linearno_adapter import install_linear_compensation
from adasolver.layers import install_compensation
from adasolver.inference.cpg.shapenet_car import ShapeNetCarCPG
from adasolver.inference.cpg.shapenet_car_stratified import ShapeNetCarStratified
from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
from adasolver.inference.pi.shapenet_car import sample_car_pi_two_stage
from adasolver.inference.shapenet_car_contract import rng_for,coarse_count
from adasolver.inference.uniform import sample_uniform
from reconstruct_api import build_plan,native_reconstruct,PlanCache,CAR_NATIVE,CAR_PLAN


def configuration_bank():
    p=json.loads((E3/'config/latency_protocol.json').read_text())
    cpg=next(c for c in p['configurations'] if c['method']=='cpg' and c['selection_target']=='field' and c['beta']==1.)
    cpg0={**cpg,'beta':0.}
    pi=next(json.loads(f.read_text())['config'] for f in (E5/'records/seed0/adaptive').glob('pi_field_*/run_manifest.json'))
    ca=next(json.loads(f.read_text())['config'] for f in (E5/'records/seed0/adaptive').glob('ca_field_*/run_manifest.json'))
    return {'full':{'method':'fullmesh','selection_target':'field'},'uniform':{'method':'uniform','selection_target':'field'},
            'pi':pi,'cpg_beta0':cpg0,'cpg_beta1':cpg,'ca':ca}


def load_backbone(architecture,device='cuda:0',with_ca=False):
    p=json.loads((E3/'config/latency_protocol.json').read_text())
    if architecture=='Transolver3':
        runtime=CarRuntime(p['t3_manifest']);model=runtime.load_operator(device);context,_=install_compensation(model)
    elif architecture=='LinearNO':
        runtime=LinearNORuntime(p['linearno_manifest']);model=runtime.load_operator(device);context,_=install_linear_compensation(model)
    else:raise ValueError(architecture)
    ca=None
    if with_ca:
        if architecture!='Transolver3':raise ValueError('No completed LinearNO Car CA head is available')
        path=E5/'checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json'
        ca=ShapeNetCarCA(model,path,runtime.binding(),device,use_cuda_graph=False)
    return runtime,model,context,ca


class CarRequest:
    def __init__(self,model,context,item,normalizer,config,budget,seed,*,device='cuda:0',ca=None):
        self.model=model;self.context=context;self.ca=ca;self.device=torch.device(device)
        self.pos=item.pos.cpu();self.fx=item.x_norm.cpu();self.coords=self.pos.numpy();self.mask=item.surface_mask.numpy()
        self.case_id=item.sample_id;self.N=len(item);self.config=dict(config);self.method=config['method']
        self.budget=float(budget);self.seed=int(seed) if seed is not None else None
        self.count=self.N if self.method=='fullmesh' else round(self.N*self.budget)
        self.mean=normalizer.mean.numpy();self.std=normalizer.std.numpy()
        self.gpu_pos=self.gpu_fx=None
        self.scorer=self.sampler=None
        # Geometry-dependent scorer/sampler setup belongs to each request.
        # In particular ShapeNetCarStratified computes surface distances here.
        self.preparation_ms=0.
        self.caches={'old':PlanCache(max_entries=1),'new':PlanCache(max_entries=1)}
        self.last_indices=None;self.last_compensation=None;self.last_plan_info=None;self.last_cache_hit=None
        self.capture_values=False;self.last_physical_values=None;self.profile_enabled=False;self.profile_rows=[]
        self.last_gpu_physical_values=None;self.last_coarse_hash=None;self.last_coarse_values=None

    def _prepare_sampling(self):
        if self.method in ['cpg','ca']:
            opts={k:self.config[k] for k in ['coarse_fraction','patch_count','neighborhood','feature','normalization',
                 'score_floor','score_power','kappa','beta','signal','surface_band']}
            self.scorer=ShapeNetCarCPG(self.coords,self.mask,patches=(np.zeros(self.N,dtype=int),np.ones(1)),placement='fps',**opts)
            self.sampler=ShapeNetCarStratified(self.coords,self.mask,**{k:self.config[k] for k in
                ['wall_scale','wall_amplitude','physics_strength','beta','placement','score_bins']})

    def prepare_scope(self,scope):
        if scope=='B' and self.gpu_pos is None:
            torch.cuda.synchronize(self.device);t=time.perf_counter()
            self.gpu_pos=self.pos[None].to(self.device);self.gpu_fx=self.fx[None].to(self.device)
            torch.cuda.synchronize(self.device)
            return (time.perf_counter()-t)*1000
        return 0.

    def _stage(self,label,fn):
        if not self.profile_enabled:return fn()
        torch.cuda.synchronize(self.device);t=time.perf_counter();value=fn();torch.cuda.synchronize(self.device)
        self.profile_rows.append(dict(component=label,ms=(time.perf_counter()-t)*1000,scope='separate instrumented request'))
        return value

    def _inputs(self,ids,scope):
        if scope=='B':
            if self.gpu_pos is None:raise RuntimeError('prepare_scope(B) before measurement')
            index=torch.as_tensor(ids,device=self.device)
            return self.gpu_pos[:,index],self.gpu_fx[:,index]
        return self.pos[ids][None].to(self.device),self.fx[ids][None].to(self.device)

    def _predict(self,ids,scope,coarse=None,cache=None,weights=None):
        pos,fx=self._inputs(ids,scope)
        if cache is not None:return self.ca.final(pos,fx,coarse,ids,cache,weights)[0]
        return self.model(pos,fx)[0]

    def _decode_cpu(self,value):return value.cpu().numpy()*self.std+self.mean
    def _decode_gpu(self,value):return value*torch.as_tensor(self.std,device=value.device)+torch.as_tensor(self.mean,device=value.device)

    def _sample(self,scope):
        self._stage('sampling_geometry_prepare_ms',self._prepare_sampling)
        rng=rng_for(self.case_id,self.seed,'refinement',self.budget)
        if self.method=='uniform':return self._stage('selection_compensation_ms',lambda:sample_uniform(self.N,self.count,rng)),None,None
        if self.method=='pi':
            cfg=self.config
            value=self._stage('selection_compensation_ms',lambda:sample_car_pi_two_stage(self.coords,self.mask,self.count,rng,
                r=cfg['r'],frac_extra=cfg['frac_extra'],beta=cfg['beta']))
            return value,None,None
        coarse=self._stage('coarse_index_ms',lambda:rng_for(self.case_id,self.seed).choice(self.N,coarse_count(self.N),replace=False))
        cache=None
        if self.method=='cpg':
            pressure=self._stage('coarse_forward_io_decode_ms',lambda:self._decode_cpu(self._predict(coarse,scope))[:,3])
        else:
            if self.ca is None:raise ValueError('CA needs the matching completed backbone head')
            def head_forward():
                cp,cx=self._inputs(coarse,scope);values,cache=self.ca.acquire(cp,cx,self.N,capture=True)
                return values.cpu().numpy(),cache
            pressure,cache=self._stage('coarse_head_io_decode_ms',head_forward)
        score=self._stage('score_ms',lambda:self.scorer._score(coarse,pressure))
        # Retain an existing CPU array for checks outside formal timing; no
        # hash, copy, or prediction cache is added to the measured request.
        self.last_coarse_values=pressure
        if self.capture_values:self.last_coarse_hash=hashlib.sha256(np.ascontiguousarray(pressure).tobytes()).hexdigest()
        sampling=self._stage('selection_compensation_ms',lambda:self.sampler.refine_score(coarse,score,self.count,rng))
        return sampling,coarse,cache

    @torch.no_grad()
    def __call__(self,implementation='new',scope='A'):
        if implementation not in ['old','new'] or scope not in ['A','B','C']:raise ValueError('unknown implementation/scope')
        self.context.clear();self.last_cache_hit=None;self.last_plan_info=None;self.last_coarse_hash=None;self.last_coarse_values=None
        if self.method=='fullmesh':
            ids=np.arange(self.N);pred=self._stage('final_forward_input_io_ms',lambda:self._predict(ids,scope))
            self.last_indices=ids;self.last_compensation=None
            return self._stage('reconstruction_pipeline_ms',lambda:self.reconstruct_prediction(pred,ids,implementation,scope))
        sampling,coarse,cache=self._sample(scope);ids=sampling.indices
        self.last_indices=ids.copy();self.last_compensation=sampling.compensation.copy()
        weights=None if self.method=='uniform' else torch.as_tensor(sampling.compensation,device=self.device)
        self.context.set(weights)
        try:pred=self._stage('final_forward_input_io_ms',lambda:self._predict(ids,scope,coarse,cache,weights))
        finally:self.context.clear()
        return self._stage('reconstruction_pipeline_ms',lambda:self.reconstruct_prediction(pred,ids,implementation,scope))

    def reconstruct_prediction(self,pred,ids,implementation,scope):
        """Complete output processing, including decode/map/apply/transfers.

        The outer profiling interval is measured directly, independently of
        the sum of its nested stages. The formal request calls this with all
        profiling clocks and barriers disabled.
        """
        if self.method=='fullmesh':
            if scope=='B':
                physical=self._stage('gpu_decode_ms',lambda:self._decode_gpu(pred))
                output=physical.to(torch.float64);output[torch.as_tensor(~self.mask,device=self.device),3]=0.
            else:
                output=self._stage('retained_d2h_decode_ms',lambda:self._decode_cpu(pred)).astype(np.float64,copy=False)
                output[~self.mask,3]=0.
            return output
        geometry_identity=self.case_id+':'+hashlib.sha256(self.coords.tobytes()).hexdigest() if scope=='C' else None
        if implementation=='old':
            physical=self._stage('retained_d2h_decode_ms',lambda:self._decode_cpu(pred))
            if self.capture_values:
                self.last_physical_values=physical.copy()
                self.last_gpu_physical_values=self._decode_gpu(pred).cpu().numpy().copy()
            if scope=='C':
                def get_plan():return self.caches['old'].get('shapenet_car',self.coords,ids,surface_mask=self.mask,
                    geometry_identity=geometry_identity,workers=1,profile=self.profile_enabled)
                plan,hit=self._stage('map_lookup_or_build_ms',get_plan);self.last_cache_hit=hit
                self.last_plan_info=dict(cpu_bytes=plan.cpu_bytes,device_bytes=plan.device_bytes(),timings=plan.timings)
                output=self._stage('interpolation_apply_ms',lambda:plan.apply_numpy(physical))
            else:output=self._stage('native_reconstruction_ms',lambda:native_reconstruct('shapenet_car',ids,physical,self.coords,self.mask))
            return self._stage('complete_field_h2d_ms',lambda:torch.as_tensor(output,device=self.device)) if scope=='B' else output
        physical=self._stage('gpu_decode_ms',lambda:self._decode_gpu(pred))
        if self.capture_values:self.last_physical_values=physical.cpu().numpy().copy()
        if scope=='C':
            plan,hit=self._stage('map_lookup_or_build_ms',lambda:self.caches['new'].get('shapenet_car',self.coords,ids,
                 surface_mask=self.mask,geometry_identity=geometry_identity,workers=4,device=self.device,profile=self.profile_enabled))
            self.last_cache_hit=hit
        else:
            plan=self._stage('plan_build_and_h2d_ms',lambda:build_plan('shapenet_car',self.coords,ids,
                 surface_mask=self.mask,workers=4,device=self.device,profile=self.profile_enabled))
        self.last_plan_info=dict(cpu_bytes=plan.cpu_bytes,device_bytes=plan.device_bytes(),timings=plan.timings)
        output=self._stage('interpolation_apply_ms',lambda:plan.apply_tensor(physical))
        return output if scope=='B' else self._stage('complete_field_d2h_ms',lambda:output.cpu().numpy())

    def profile(self,implementation,scope):
        self.profile_rows=[];self.profile_enabled=True
        try:output=self(implementation,scope);torch.cuda.synchronize(self.device)
        finally:self.profile_enabled=False
        return output,self.profile_rows,self.last_plan_info

    def clear_caches(self):
        self.caches={'old':PlanCache(max_entries=1),'new':PlanCache(max_entries=1)}
