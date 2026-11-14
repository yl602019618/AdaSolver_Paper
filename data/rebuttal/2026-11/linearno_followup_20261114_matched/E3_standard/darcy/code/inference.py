"""Darcy sampling and complete-output requests with explicitly keyed reconstruction."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import OUT,FOLLOWUP,Dataset,build_model,install,read,digest
from ca_support import ScoreHead,frozen_features,decode_score
from adasolver.inference.pi.darcy import sample_darcy_pi
from adasolver.inference.cpg.darcy import DarcyCPG
from adasolver.inference.ca.darcy import DarcyCA
from adasolver.inference.uniform import sample_uniform
from adasolver.utils import budget_count
from adasolver.utils.seed import make_numpy_rng
from adasolver.inference import adaptive

RECON=FOLLOWUP/'reconstruction'
sys.path.insert(0,str(RECON/'code'))
import reconstruct_api as reconstruction_module
from reconstruct_api import build_plan,native_reconstruct,PlanCache,GRID_NATIVE,GRID_PLAN


def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def validate_normalizer_identity(backbone_metadata,current_normalizer_sha256,head_metadata=None):
    """Reject stale preprocessing identities before model/head inference."""
    if backbone_metadata.get('normalizer_sha256')!=current_normalizer_sha256:
        raise ValueError('backbone/current data normalizer mismatch')
    if head_metadata is not None and head_metadata.get('normalizer_sha256')!=current_normalizer_sha256:
        raise ValueError('head/backbone/current data normalizer mismatch')


class HeadAcquisition(nn.Module):
    def __init__(self,operator,backbone,head,metadata):
        super().__init__();self.operator=operator;self.backbone=backbone;self.head=head
        self.metadata=metadata
    def forward(self,pos,fx):return self.head(frozen_features(self.operator,self.backbone,pos,fx))
    def decode_score(self,value):return decode_score(value,self.metadata['target_mean'],self.metadata['target_std'])


class Runtime:
    def __init__(self,backbone,device='cuda:0',conv=False,require_head=True,load_head=True):
        self.backbone=backbone;self.variant='conv_temp' if conv else 'point_temp' if backbone=='LinearNO' else 'native_point'
        self.label='LinearNO_conv_temp' if conv else 'LinearNO_point_temp' if backbone=='LinearNO' else 'Transolver3'
        self.device=torch.device(device);self.data=Dataset();self.conv=conv
        key='linearno_conv' if conv else backbone.lower();folder=OUT/'checkpoints'/key
        self.metadata=read(folder/'manifest.json');assert self.metadata['execution_status']=='completed'
        if digest(folder/'best.pt')!=self.metadata['checkpoint_sha256']:raise ValueError('backbone hash mismatch')
        self.normalizer_sha256=digest(OUT/'data/normalizer.npz')
        validate_normalizer_identity(self.metadata,self.normalizer_sha256)
        self.model=build_model(backbone,conv=conv).to(self.device)
        self.model.load_state_dict(torch.load(folder/'best.pt',map_location=self.device,weights_only=True));self.model.requires_grad_(False).eval()
        self.context=None if conv else install(backbone,self.model)
        self.mean=self.data.y_mean.to(self.device);self.std=self.data.y_std.to(self.device)
        self.reference=self.data.reference;self.cpg=self.ca=self.head_metadata=None
        self.cases={}
        if not conv:
            self.cpg=DarcyCPG(self.reference,patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.,acquisition='grad_lap')
            hm=folder/'head/manifest.json'
            if load_head and hm.exists() and read(hm).get('execution_status')=='completed':
                self.head_metadata=read(hm)
                if self.head_metadata['operator_sha256']!=self.metadata['checkpoint_sha256']:raise ValueError('head/backbone mismatch')
                validate_normalizer_identity(self.metadata,self.normalizer_sha256,self.head_metadata)
                head=ScoreHead(backbone).to(self.device)
                hp=folder/'head/best.pt'
                if digest(hp)!=self.head_metadata['checkpoint_sha256']:raise ValueError('head hash mismatch')
                head.load_state_dict(torch.load(hp,map_location=self.device,weights_only=True));head.requires_grad_(False).eval()
                self.ca=DarcyCA(self.reference,HeadAcquisition(self.model,backbone,head,self.head_metadata),
                                patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.)
            elif require_head:raise RuntimeError('Corresponding CA head incomplete')

    def case(self,sid):
        sid=int(sid)
        if sid not in self.cases:
            pos,fx,truth=self.data.sample(sid)
            self.cases[sid]={'pos':pos,'fx':fx,'truth':truth.numpy(),
                            'coeff':(fx*self.data.x_std[0]+self.data.x_mean[0]).numpy(),
                            'gpu_pos':None,'gpu_fx':None}
        return self.cases[sid]

    @torch.no_grad()
    def forward(self,case,ids=None,scope='A'):
        if scope=='B':
            if case['gpu_pos'] is None:
                case['gpu_pos']=case['pos'].to(self.device);case['gpu_fx']=case['fx'].to(self.device)
            if ids is None:pos=case['gpu_pos'][None];fx=case['gpu_fx'][None]
            else:
                idx=torch.as_tensor(ids,dtype=torch.long,device=self.device)
                pos=case['gpu_pos'][idx][None];fx=case['gpu_fx'][idx][None]
        elif ids is None:pos=case['pos'][None].to(self.device);fx=case['fx'][None].to(self.device)
        else:pos=case['pos'][ids][None].to(self.device);fx=case['fx'][ids][None].to(self.device)
        return self.model(pos,fx)[0]

    @torch.no_grad()
    def sample(self,sid,method,budget,seed,scope='A'):
        if self.conv and method!='full':raise ValueError('official conv-temp requires full85x85 grid')
        case=self.case(sid)
        if self.context:self.context.clear()
        count=7225 if method=='full' else budget_count(7225,budget)
        if method=='full':return np.arange(7225),None,None,None,{}
        rng=make_numpy_rng(int(seed),int(sid));coarse=None;coarse_prediction=None
        if method=='uniform':sampling=sample_uniform(7225,count,rng)
        elif method=='pi':
            coefficients=case['coeff']
            if scope=='B':
                coefficients=(case['gpu_fx']*self.data.x_std[0].to(self.device)+self.data.x_mean[0].to(self.device)).cpu().numpy()
            sampling=sample_darcy_pi(coefficients,self.reference,count,rng,
                      amp=.25,beta=.5,prior='interface_distance',ell=2/84,shape=(85,85))
        else:
            coarse=rng.choice(7225,722,replace=False)
            sampler=self.cpg if method.startswith('cpg') else self.ca
            if sampler is None:raise RuntimeError('corresponding acquisition unavailable')
            # Count native geometric patch construction in each request, including C.
            # C's only reuse is the explicitly keyed reconstruction map.
            sampler.patch_id,sampler.patch_mass=adaptive.build_patches(self.reference,32,seed=0)
            if method.startswith('cpg'):
                coarse_prediction=self.forward(case,coarse,scope).cpu().numpy()
                sampling=self.cpg.refine(coarse,coarse_prediction,count,rng)
            elif method=='ca':
                if self.ca is None:raise RuntimeError('head unavailable')
                if scope=='B':
                    idx=torch.as_tensor(coarse,dtype=torch.long,device=self.device)
                    cp=case['gpu_pos'][idx][None];cx=case['gpu_fx'][idx][None]
                else:cp=case['pos'][coarse][None].to(self.device);cx=case['fx'][coarse][None].to(self.device)
                sampling=self.ca.refine(coarse,cp,cx,count,rng)
            else:raise ValueError(method)
        weights=None if method=='uniform' else np.ones(count,dtype=np.float32) if method=='cpg_beta0' else sampling.compensation
        metadata=dict(sampling.metadata)
        if method.startswith('cpg'):metadata['coarse_prediction_space']='normalized solution output, before physical decode; native Darcy grad_lap semantics'
        metadata['compensation_beta']=None if method=='uniform' else 0. if method=='cpg_beta0' else .5 if method=='pi' else 1.
        if method.startswith('cpg') or method=='ca':metadata['geometry_preparation']='native full-grid patches rebuilt in every request'
        if method=='ca':
            metadata['head_architecture']=self.head_metadata['head_architecture']
            metadata['architecture']='official_point_temp_two_block_head' if self.backbone=='LinearNO' else 'transolver_gradient_head'
        return sampling.indices,weights,coarse,coarse_prediction,metadata

    @torch.no_grad()
    def predict(self,sid,method,budget,seed,scope='A'):
        case=self.case(sid)
        if scope=='B' and case['gpu_pos'] is None:
            case['gpu_pos']=case['pos'].to(self.device);case['gpu_fx']=case['fx'].to(self.device)
        ids,weights,coarse,cp,metadata=self.sample(sid,method,budget,seed,scope)
        if self.context:self.context.set(None if weights is None else torch.as_tensor(weights,dtype=torch.float32,device=self.device))
        try:pred=self.forward(self.case(sid),None if method=='full' else ids,scope)*self.std[0]+self.mean[0]
        finally:
            if self.context:self.context.clear()
        return pred,dict(indices=ids,weights=weights,coarse_indices=coarse,coarse_prediction=cp,sampling_metadata=metadata)


class CompleteRequest:
    def __init__(self,runtime,sid,method,budget,seed,reconstruction='new',scope='A'):
        if scope not in ['A','B','C']:raise ValueError(scope)
        self.runtime=runtime;self.sid=int(sid);self.method=method;self.budget=budget;self.seed=seed
        self.reconstruction=reconstruction;self.scope=scope;self.cache=PlanCache(max_entries=2);self.last=None
        self.runtime.case(sid)
        if scope=='B':
            case=self.runtime.case(sid)
            case['gpu_pos']=case['pos'].to(self.runtime.device);case['gpu_fx']=case['fx'].to(self.runtime.device)
        self.reconstruction_id=GRID_NATIVE if reconstruction=='old' else GRID_PLAN
        if scope=='C':self.reconstruction_id='grid_qhull_cached_cpu_apply_v1p1' if reconstruction=='old' else 'grid_qhull_cached_gpu_apply_v1p1'

    @torch.no_grad()
    def __call__(self):
        pred,meta=self.runtime.predict(self.sid,self.method,self.budget,self.seed,self.scope)
        if self.method=='full':
            self.last={**meta,'cache_hit':False,'plan_cpu_bytes':0,'plan_device_bytes':0}
            return pred.to(torch.float64) if self.scope=='B' else pred.cpu().numpy().astype(np.float64,copy=False)
        if self.scope=='C':
            plan,hit=self.cache.get('darcy',self.runtime.reference,meta['indices'],
                geometry_identity=f'darcy_case_{self.sid}',channels=1,reconstruction_id=GRID_PLAN,
                device=self.runtime.device if self.reconstruction=='new' else None)
            result=plan.apply_numpy(pred.cpu().numpy()) if self.reconstruction=='old' else plan.apply_tensor(pred).cpu().numpy()
        elif self.reconstruction=='old':
            result=native_reconstruct('darcy',meta['indices'],pred.cpu().numpy(),self.runtime.reference)
            if self.scope=='B':result=torch.as_tensor(result,dtype=torch.float64,device=self.runtime.device)
            plan=None;hit=False
        else:
            plan=build_plan('darcy',self.runtime.reference,meta['indices'],channels=1,
                            reconstruction_id=GRID_PLAN,device=self.runtime.device)
            result=plan.apply_tensor(pred)
            if self.scope=='A':result=result.cpu().numpy()
            hit=False
        self.last={**meta,'cache_hit':hit,'plan_cpu_bytes':plan.cpu_bytes if plan else 0,
                   'plan_device_bytes':plan.device_bytes() if plan else 0}
        return result


def metric(field,truth):
    return float(np.linalg.norm(np.asarray(field,dtype=np.float64)-np.asarray(truth,dtype=np.float64))/np.linalg.norm(np.asarray(truth,dtype=np.float64)))


def comparison(old,new,truth):
    old=np.asarray(old);new=np.asarray(new);difference=np.abs(new-old);tolerance=1e-6+1e-5*np.abs(old)
    a,b=metric(old,truth),metric(new,truth)
    return dict(max_absolute_difference=float(difference.max()),relative_difference=float(np.linalg.norm(new-old)/max(np.linalg.norm(old),1e-300)),
                violating_elements=int(np.count_nonzero(difference>tolerance)),metric_old=a,metric_new=b,
                metric_difference=abs(a-b),metric_tolerance=1e-7+1e-5*abs(a),
                difference_quantiles={str(q):float(np.quantile(difference,q)) for q in [.5,.9,.99,.999,1.]})
