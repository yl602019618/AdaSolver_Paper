"""Minimal normalizer gate and native Full-input checks; no trained metrics."""
import hashlib
import numpy as np
import torch
from common import OUT,Dataset,build_model,digest,seeds,write
from inference import Runtime,validate_normalizer_identity


def main():
    current=digest(OUT/'data/normalizer.npz');different=hashlib.sha256(b'explicit mismatched metadata fixture').hexdigest()
    identity=[]
    cases=[('matching',current,current,True),('head_mismatch',current,different,False),
           ('backbone_current_mismatch',different,different,False)]
    for name,backbone_hash,head_hash,expected in cases:
        error=None
        try:
            validate_normalizer_identity({'normalizer_sha256':backbone_hash},current,{'normalizer_sha256':head_hash})
            accepted=True
        except ValueError as exc:accepted=False;error=str(exc)
        assert accepted==expected
        identity.append(dict(case=name,accepted=accepted,expected_acceptance=expected,error=error))
    write(OUT/'results/head_normalizer_identity_validation.json',dict(status='passed',
        scope='identity-gate metadata fixtures using actual current normalizer hash; no completed head checkpoint claimed',
        current_normalizer_sha256=current,checks=identity,source_sha256=digest(OUT/'code/inference.py')))
    torch.set_num_threads(4);seeds(0);data=Dataset();pos,fx,truth=data.sample(4);rows=[]
    for backbone,conv in [('Transolver3',False),('LinearNO',False),('LinearNO',True)]:
        rt=Runtime.__new__(Runtime);rt.device=torch.device('cpu');rt.model=build_model(backbone,conv=conv).eval()
        inputs={'pos':pos,'fx':fx,'gpu_pos':None,'gpu_fx':None};pointers=[]
        hook=rt.model.register_forward_pre_hook(lambda module,args:pointers.append((args[0].data_ptr(),args[1].data_ptr())))
        direct=rt.forward(inputs,None,'A');indexed=rt.forward(inputs,np.arange(7225),'A');hook.remove()
        delta=float(torch.max(torch.abs(direct-indexed)))
        assert torch.equal(direct,indexed)
        assert pointers[0]==(pos.data_ptr(),fx.data_ptr())
        assert pointers[1]!=(pos.data_ptr(),fx.data_ptr())
        rows.append(dict(backbone=backbone,conv=conv,full_points=7225,max_absolute_difference=delta,
            full_inputs_share_original_tensor_storage=True,indexed_reference_allocates_copy=True))
    write(OUT/'results/full_input_path_validation.json',dict(status='passed',device='CPU',
        scope='initialized actual native classes; same full output without full-range fancy indexing; not a GPU speed measurement',
        checks=rows,source_sha256=digest(OUT/'code/inference.py')))
    print('head normalizer identity and native Full paths passed',flush=True)


if __name__=='__main__':main()
