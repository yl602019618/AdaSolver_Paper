"""CPU full-shape interface check; random initialized weights are not results."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
import json,time
import numpy as np
import torch
from common import OUT,ResolutionData,build_model,configure_precision,write
import evaluate
import head_support


def main():
    configure_precision(0,4);evaluate.policy_protocol_check();records=[];started=time.perf_counter()
    load_model=evaluate.load_model;load_head=head_support.load_head
    def fixture_model(n,device):
        model=build_model().to(device).eval().requires_grad_(False)
        return model,{'checkpoint_sha256':'engineering_random_initialization_not_a_checkpoint'}
    def fixture_head(n,operator,device):
        module=head_support.HeadAcquisition(operator,head_support.ScoreHead(),{'target_mean':1.,'target_std':1.})
        return module.to(device).eval(),{'checkpoint_sha256':'engineering_random_initialization_not_a_head'}
    evaluate.load_model=fixture_model;head_support.load_head=fixture_head
    try:
        for n in [85,141,211]:
            runtime=evaluate.Runtime(n,'cpu',True)
            try:
                x,_=runtime.data.load_split('validation')
                coeff=np.load(runtime.data.directory/'validation_coeff.npy',mmap_mode='r')
                for sid in [900,949]:
                    for method in evaluate.METHODS:
                        selected,weights,info=runtime.choose(sid,x[sid-900],coeff[sid-900],method,1. if method=='full' else .375,
                            None if method=='full' else 0,validate_tap=True)
                        prediction=runtime.forward(x[sid-900],selected,weights)
                        common,native,_=evaluate.reconstruct_pair(runtime.reference,selected,prediction,runtime.query,full=method=='full')
                        assert native.shape==(n*n,) and common.shape==(211**2,)
                        assert len(selected)==(n*n if method=='full' else round(.375*n*n))
                        records.append(dict(resolution=n,case_id=sid,method=method,available_input_shape=[1,n*n,3],
                            model_input_shape=[1,len(selected),3],retained_prediction_shape=list(prediction.shape),
                            native_output_shape=list(native.shape),common_output_shape=list(common.shape),
                            coarse_points=len(info['coarse_indices']),all_finite=True,validation_only=True))
                    print(json.dumps(dict(resolution=n,case_id=sid,completed_interface_requests=len(records))),flush=True)
            finally:runtime.close()
    finally:evaluate.load_model=load_model;head_support.load_head=load_head
    result=dict(passed=True,scope='CPU engineering interface validation with random initialized actual architecture; no accuracy claims',
        trained_weights=False,test_opened=False,physical_validation_case_ids=[900,949],records=records,
        elapsed_seconds=time.perf_counter()-started)
    write(OUT/'evaluation/inference_shape_preflight.json',result)


if __name__=='__main__':main()
