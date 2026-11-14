"""One architecture/method per process; allocator peaks with output scope bound."""
from __future__ import annotations
import argparse,json,os,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
from car_request import OUT,configuration_bank,load_backbone,CarRequest
from timing_helper import isolated_memory
from inference_speed.gpu_monitor import ExclusiveGPU


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',required=True);parser.add_argument('--variant',required=True)
    parser.add_argument('--scope',required=True,choices=['A','B','C']);parser.add_argument('--gpu',type=int,default=6);a=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((OUT/'config/car_protocol.json').read_text());bank=configuration_bank()
    root=OUT/'records/car_memory';root.mkdir(parents=True,exist_ok=True)
    prefix=f'{a.backbone}_{a.variant}_{a.scope}';out=root/(prefix+'.json')
    hashes={name:hashlib.sha256((OUT/'code'/name).read_bytes()).hexdigest()
            for name in ['car_memory.py','car_request.py','reconstruct_api.py','timing_helper.py']}
    if out.exists() and json.loads(out.read_text()).get('execution_status')=='completed':
        assert json.loads(out.read_text())['source_hashes']==hashes,'Memory source changed; do not mix runs'
        return
    def nvml():
        r=subprocess.run(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        return int(r.stdout.strip())*1024**2 if r.returncode==0 else None
    rows=[]
    with ExclusiveGPU(a.gpu,root/(prefix+'_gpu.json'),interval=1.) as monitor:
        runtime,model,context,head=load_backbone(a.backbone,'cuda:0',with_ca=a.variant=='ca')
        for case in protocol['timing_cases']:
            item=runtime.sample(case)
            for budget in ([1.] if a.variant=='full' else protocol['budgets']):
                for seed in ([None] if a.variant=='full' else protocol['sampling_seeds']):
                    for implementation in (['shared'] if a.variant=='full' else ['old','new']):
                        req=CarRequest(model,context,item,runtime.output_normalizer,bank[a.variant],budget,seed,ca=head)
                        req.prepare_scope(a.scope)
                        call=lambda:req('old' if implementation=='shared' else implementation,a.scope)
                        memory=isolated_memory(call,nvml_post_sample=nvml)
                        rows.append(dict(backbone=a.backbone,variant=a.variant,scope=a.scope,implementation=implementation,
                                         case_id=case,budget_fraction=budget,sampling_seed=seed,
                                         cached_map_info=req.last_plan_info,**memory))
                        del req;torch.cuda.empty_cache();monitor.assert_exclusive()
    out.write_text(json.dumps({'execution_status':'completed','evidence_status':'pending','rows':rows,
                             'one_architecture_and_one_method_loaded':True,'source_hashes':hashes},indent=2)+'\n')
    print(prefix,'completed',len(rows),'memory conditions',flush=True)


if __name__=='__main__':main()
