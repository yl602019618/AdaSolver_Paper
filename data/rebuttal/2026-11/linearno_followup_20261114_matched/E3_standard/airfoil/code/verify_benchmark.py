"""CPU development check of artifact-to-request matching, including own CA feature tap."""
from benchmark import *
from runtime import OwnPressureHead


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(314);sid=gate['validation_case_ids'][0]
    destination=OUT/'results/benchmark_development';destination.mkdir(exist_ok=True);checks=[]
    for arch in ['t3','linearno']:
        meta=read(OUT/f'checkpoints/{arch}/manifest.json');meta=dict(meta,checkpoint_sha256='random_development_only')
        rt=Runtime(arch,'cpu',model_override=(build_model(arch).eval(),meta))
        head=OwnPressureHead(rt.model,float(rt.data.mean),float(rt.data.std),feature_dim=8).eval()
        rt.head=lambda cfg:(head,'random_development_head_only',{})
        full=rt.predict_full(sid).numpy().astype(np.float64);full_path=destination/f'{arch}_full.npz';np.savez_compressed(full_path,field=full)
        full_metrics=metrics(full,rt.data.coords(sid),rt.data.target(sid).astype(np.float64))
        reference={(arch,sid,'full',None,1.):(dict(old=full_metrics,new=full_metrics,
            checkpoint_sha256=rt.operator_hash,normalizer_sha256=rt.data.normalizer_sha256),full_path)}
        for scope in ['A_cpu_cpu','C_cpu_cpu_cached_map']:
            request=Request(rt,api,gate,sid,None,1.,None,'new',scope)
            checked=check_output(request,request(),reference)
            checks.append(dict(architecture=arch,method='full',scope=scope,implementation='shared',**checked))
        for cfg in configurations(arch):
            rt.clear_dynamic();prediction,sampling,coarse,extra=rt.predict_subset(sid,0,.375,cfg)
            raw=api.native_reconstruct('airfoil',sampling.indices,prediction.numpy(),rt.reference)
            result=metrics(raw,rt.data.coords(sid),rt.data.target(sid).astype(np.float64))
            path=destination/f'{arch}_{cfg["name"]}.npz'
            np.savez_compressed(path,indices=sampling.indices,compensation=sampling.compensation,
                coarse_indices=np.asarray([] if coarse is None else coarse),prediction=prediction.numpy())
            reference={(arch,sid,cfg['name'],0,.375):(dict(old=result,new=result,coarse_prediction_hash=extra.get('coarse_prediction_hash'),
                checkpoint_sha256=rt.operator_hash,normalizer_sha256=rt.data.normalizer_sha256,
                head_sha256='random_development_head_only' if cfg['name']=='ca' else None),path)}
            for scope in ['A_cpu_cpu','C_cpu_cpu_cached_map']:
                for mode in ['old','new']:
                    request=Request(rt,api,gate,sid,0,.375,cfg,mode,scope)
                    verified=check_output(request,request(),reference)
                    if scope.startswith('C_'):
                        assert request.last_cache_hit is False
                        verified=check_output(request,request(),reference);assert request.last_cache_hit is True
                    checks.append(dict(architecture=arch,method=cfg['name'],scope=scope,implementation=mode,**verified))
                    print(arch,cfg['name'],scope,mode,'passed',flush=True)
    write(OUT/'results/benchmark_development_validation.json',dict(passed=True,checks=checks,
        scope='CPU random backbone/head initialization; complete output/index/weight/coarse identity checking, not accuracy or timing evidence',
        benchmark_source_sha256=sha(OUT/'code/benchmark.py'),request_source_sha256=sha(OUT/'code/request.py')))


if __name__=='__main__':main()
