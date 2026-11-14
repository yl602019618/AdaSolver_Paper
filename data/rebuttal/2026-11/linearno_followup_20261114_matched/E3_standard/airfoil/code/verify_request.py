"""CPU development check for the timing adapter; not trained accuracy evidence."""
from request import *
from evaluate import attach_reconstruction


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(123)
    api,gate=attach_reconstruction();checks=[]
    sid=gate['validation_case_ids'][0]
    for arch in ['t3','linearno']:
        meta=read(OUT/f'checkpoints/{arch}/manifest.json');meta=dict(meta,checkpoint_sha256='random_initialization_development_only')
        runtime=Runtime(arch,'cpu',model_override=(build_model(arch).eval(),meta))
        for cfg in configurations(arch,include_ca=False):
            runtime.clear_dynamic();normal=runtime.sampling(sid,0,.375,cfg)
            runtime.clear_dynamic();fast=untimed_sampling(runtime)(sid,0,.375,cfg)
            np.testing.assert_array_equal(normal[0].indices,fast[0].indices)
            np.testing.assert_array_equal(normal[0].compensation,fast[0].compensation)
            if normal[1] is not None:np.testing.assert_array_equal(normal[1],fast[1])
            old=Request(runtime,api,gate,sid,0,.375,cfg,'old','A_cpu_cpu')
            new=Request(runtime,api,gate,sid,0,.375,cfg,'new','A_cpu_cpu')
            a=old();b=new();np.testing.assert_array_equal(old.last_indices,new.last_indices)
            assert a.dtype==np.float64 and b.dtype==np.float64
            delta=np.abs(a-b);assert np.all(delta<=1e-6+1e-5*np.abs(a))
            checks.append(dict(architecture=arch,method=cfg['name'],indices_equal=True,weights_equal=True,
                               full_shape=list(a.shape),max_field_abs_difference=float(delta.max())))
        cfg=configurations(arch,include_ca=False)[0]
        cached=Request(runtime,api,gate,sid,0,.375,cfg,'new','C_cpu_cpu_cached_map')
        a=cached();assert cached.last_cache_hit is False
        b=cached();assert cached.last_cache_hit is True
        np.testing.assert_array_equal(a,b)
    write(OUT/'results/request_development_validation.json',dict(status='passed',scope='CPU random-initialization API/untimed-sampler correctness only; not training or performance evidence',checks=checks,
          reconstruction_source_sha256=gate['module_sha256'],request_code_sha256=sha(__file__.replace('verify_request.py','request.py'))))
    print('request development checks',len(checks),'passed')

if __name__=='__main__':main()
