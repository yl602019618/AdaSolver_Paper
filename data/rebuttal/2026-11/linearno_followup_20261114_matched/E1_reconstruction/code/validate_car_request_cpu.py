"""Regression for request-boundary refactor, using the saved real CPU pilot."""
from __future__ import annotations
import hashlib,json
import numpy as np
import torch
from car_request import OUT,configuration_bank,load_backbone,CarRequest
from validate_contract import difference
from adasolver.metrics.shapenet_car import dual_field_metrics


@torch.no_grad()
def main():
    torch.set_num_threads(4);bank=configuration_bank()
    reference=[json.loads(line) for line in (OUT/'records/car_pilot/pairs.jsonl').open()];checks=[]
    for arch in ['Transolver3','LinearNO']:
        runtime,model,context,head=load_backbone(arch,'cpu',with_ca=arch=='Transolver3')
        for row in [r for r in reference if r['backbone']==arch]:
            item=runtime.sample(row['case_id']);request=CarRequest(model,context,item,runtime.output_normalizer,
                bank[row['variant']],row['budget_fraction'],row['sampling_seed'],device='cpu',ca=head)
            request.capture_values=True;old=request('old','A');ids=request.last_indices.copy()
            assert hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()==row['selected_indices_sha256']
            metrics=dual_field_metrics(old,item.target_phys.numpy(),item.surface_mask.numpy())
            assert abs(metrics.field_error-row['old_metrics']['field_error'])<1e-12
            for scope in ['A','C']:
                new=request('new',scope);np.testing.assert_array_equal(ids,request.last_indices)
                diff=difference(old,new);assert diff['violations']==0
                if scope=='C' and row['variant']!='full':
                    assert request.last_cache_hit is False
                    again=request('new','C');assert request.last_cache_hit is True
                    assert difference(new,again)['violations']==0
                checks.append(dict(backbone=arch,variant=row['variant'],scope=scope,**diff))
        del model,runtime,head
    (OUT/'results/car_request_cpu_regression.json').write_text(json.dumps(dict(passed=True,checks=checks,
        source_sha256=hashlib.sha256((OUT/'code/car_request.py').read_bytes()).hexdigest(),
        scope='CPU regression only; no GPU latency or trained new-cohort result',
        boundary='geometry-dependent sampler creation moved inside every request; previous saved pilot metrics unchanged'),indent=2)+'\n')
    print('CPU request regression passed',len(checks),'A/C comparisons; all saved pilot metrics and selected hashes unchanged')


if __name__=='__main__':main()
