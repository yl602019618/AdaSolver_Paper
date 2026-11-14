"""CPU-only integration checks; untrained initialized models are not experiment results."""
import numpy as np
import torch
from common import OUT,Dataset,build_model,install,seeds,write,digest
from ca_support import ScoreHead,frozen_features
from inference import Runtime,CompleteRequest,HeadAcquisition,comparison,array_hash,DarcyCPG,DarcyCA


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;seeds(0)
    data=Dataset();rows=[]
    for backbone in ['Transolver3','LinearNO']:
        rt=Runtime.__new__(Runtime);rt.backbone=backbone;rt.device=torch.device('cpu');rt.conv=False
        rt.data=data;rt.reference=data.reference;rt.model=build_model(backbone).requires_grad_(False).eval()
        rt.context=install(backbone,rt.model);rt.mean=data.y_mean;rt.std=data.y_std;rt.cases={}
        rt.cpg=DarcyCPG(rt.reference,patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.,acquisition='grad_lap')
        head=ScoreHead(backbone).eval();rt.head_metadata=dict(head_architecture='native_two_block',target_mean=0.,target_std=1.)
        rt.ca=DarcyCA(rt.reference,HeadAcquisition(rt.model,backbone,head,rt.head_metadata),patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.)
        sid=4;truth=rt.case(sid)['truth']
        for method in ['full','uniform','pi','cpg_beta0','cpg','ca']:
            budget=1. if method=='full' else .375;seed=None if method=='full' else 0
            old=CompleteRequest(rt,sid,method,budget,seed,'old','A');new=CompleteRequest(rt,sid,method,budget,seed,'new','A')
            a,b=old(),new();difference=comparison(a,b,truth)
            assert a.shape==b.shape==(7225,1) and a.dtype==b.dtype==np.float64
            assert difference['violating_elements']==0 and difference['metric_difference']<=difference['metric_tolerance']
            assert array_hash(old.last['indices'])==array_hash(new.last['indices'])
            c=CompleteRequest(rt,sid,method,budget,seed,'new','C');first=c();second=c()
            assert np.array_equal(first,second) and (method=='full' or c.last['cache_hit'])
            rows.append(dict(backbone=backbone,method=method,case_id=sid,budget=budget,
                retained_points=len(new.last['indices']),old_new_max_abs=difference['max_absolute_difference'],
                relative_difference=difference['relative_difference'],metric_difference=difference['metric_difference'],
                status='passed',scope='CPU integration only; untrained model, not accuracy/latency evidence'))
        selected=np.arange(722);pos,fx,_=data.sample(sid)
        features=frozen_features(rt.model,backbone,pos[selected][None],fx[selected][None])
        head.train();head.zero_grad();head(features).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        print('passed',backbone,flush=True)
    write(OUT/'results/request_integration_preflight.json',dict(status='passed',device='CPU',
        evidence_scope='implementation integration of initialized full classes only; no trained result generated',
        checks=rows,head_backward='both own-native two-block heads finite and participating',
        source_sha256={name:digest(__file__.replace('preflight_request.py',name)) for name in ['preflight_request.py','inference.py','common.py','ca_support.py']}))


if __name__=='__main__':main()
