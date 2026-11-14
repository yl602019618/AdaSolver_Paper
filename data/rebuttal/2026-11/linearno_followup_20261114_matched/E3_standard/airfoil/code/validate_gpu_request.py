"""Trained GPU request/accuracy identity preflight; no latency measurements."""
from benchmark import *


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    references={};sid=1001
    for arch in ['t3','linearno']:
        complete=read(OUT/f'results/{arch}_complete_completion.json')
        assert read(OUT/f'results/{arch}_accuracy_acceptance.json')['passed']
        for source in complete['source_files']:
            path=Path(source)
            for line in path.open():
                row=json.loads(line)
                if row['case_id']==sid:references[(arch,sid,row['method'],row['sampling_seed'],row['budget_fraction'])]=(row,path.parent/row['arrays'])
    checks=[];requests=0
    with ExclusiveGPU(2,OUT/'results/gpu_request_validation_occupancy.json',interval=1.) as monitor:
        for arch in ['t3','linearno']:
            rt=Runtime(arch)
            cfgs=[None]+[c for c in configurations(arch) if c['name'] in ['uniform','pi','cpg','ca']]
            for cfg in cfgs:
                method='full' if cfg is None else cfg['name']
                for scope in SCOPES.values():
                    for mode in (['new'] if cfg is None else ['old','new']):
                        request=Request(rt,api,gate,sid,None if cfg is None else 0,1. if cfg is None else .375,cfg,mode,scope)
                        output=request();requests+=1;torch.cuda.synchronize()
                        check=check_output(request,output,references);del output
                        reused=False
                        if cfg is not None and scope.startswith('C_'):
                            assert request.last_cache_hit is False
                            output=request();requests+=1;torch.cuda.synchronize();check_output(request,output,references);del output
                            assert request.last_cache_hit is True;reused=True
                        checks.append(dict(architecture=arch,method=method,scope=scope,implementation=mode,cache_reuse_checked=reused,**check))
                        print(arch,method,scope,mode,'passed',flush=True);monitor.assert_exclusive()
                        del request
            del rt;gc.collect();torch.cuda.empty_cache()
    write(OUT/'results/gpu_request_development_validation.json',dict(passed=True,physical_gpu=2,request_calls=requests,conditions=len(checks),checks=checks,
        source_snapshot_id=SOURCE_SNAPSHOT_ID,code_identities=CODE_IDENTITIES,reconstruction_sha256=api.IMPLEMENTATION_SOURCE_SHA256,
        scope='trained main checkpoints and accepted own heads, actual GPU2 A/B/C complete-field/index/coarse/weight identity against frozen test artifacts; no latency statistics'))


if __name__=='__main__':main()
