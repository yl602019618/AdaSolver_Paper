"""Complete field requests with explicit A/B/C input/output/cache contracts."""
from runtime import *
import inspect
from types import FunctionType,SimpleNamespace


def untimed_sampling(runtime):
    original=inspect.unwrap(AirfoilEvaluator.sampling)
    namespace=dict(original.__globals__)
    namespace['time']=SimpleNamespace(perf_counter=lambda:0.)
    namespace['synchronized_time']=lambda device:0.
    function=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return types.MethodType(function,runtime)


class Request:
    def __init__(self,runtime,api,gate,sid,seed,budget,config,implementation,scope):
        self.runtime=runtime;self.api=api;self.gate=gate;self.sid=sid;self.seed=seed;self.budget=budget
        self.config=config;self.method='full' if config is None else config['name']
        self.implementation=implementation;self.scope=scope;self.sampling=untimed_sampling(runtime)
        self.coords=np.ascontiguousarray(runtime.data.coords(sid));self.reference=runtime.reference
        self.geometry_id=None
        self.gpu_coords=torch.as_tensor(self.coords,device=runtime.device) if scope=='B_gpu_gpu' else None
        self.cache=api.PlanCache(max_entries=1) if scope=='C_cpu_cpu_cached_map' else None
        self.last_indices=None;self.last_compensation=None;self.last_coarse=None;self.last_cache_hit=None
        self.last_plan=None
        self.last_map_cpu_bytes=0;self.last_map_gpu_bytes=0;self.collect_memory=False
        self.reconstruction_id='full_native_identity_float64' if self.method=='full' else gate['old_id'] if implementation=='old' and scope!='C_cpu_cpu_cached_map' else gate['new_id']
        if scope=='C_cpu_cpu_cached_map' and self.method!='full':
            self.reconstruction_id='grid_qhull_cached_cpu_apply_v1p1' if implementation=='old' else 'grid_qhull_cached_gpu_apply_v1p1'

    def _prepare_sampling(self,coords,instrumented=False):
        runtime=self.runtime;original=runtime.data.coords
        runtime.data.coords=lambda sid:coords
        try:
            function=runtime.sampling if instrumented else self.sampling
            return function(self.sid,self.seed,self.budget,self.config)
        finally:runtime.data.coords=original

    def _prediction(self,indices,compensation,coords):
        if self.method=='full':
            pos=self.gpu_coords[None] if self.scope=='B_gpu_gpu' else torch.as_tensor(coords,device=self.runtime.device)[None]
        elif self.scope=='B_gpu_gpu':
            selected=torch.as_tensor(indices,device=self.runtime.device)
            pos=self.gpu_coords[selected][None]
        else:pos=torch.as_tensor(coords[indices],device=self.runtime.device)[None]
        self.runtime.context.set(None if compensation is None else torch.as_tensor(compensation,dtype=torch.float32,device=self.runtime.device))
        try:return self.runtime.data.decode(self.runtime.model(pos,pos))[0]
        finally:self.runtime.context.clear()

    def _reconstruct(self,indices,prediction):
        if self.cache is not None:
            # The first cached request pays physical-geometry identity construction.
            if self.geometry_id is None:self.geometry_id=hashlib.sha256(self.coords.tobytes()).hexdigest()
            plan,hit=self.cache.get('airfoil',self.reference,indices,geometry_identity=self.geometry_id,
                                    reconstruction_id=self.gate['new_id'],channels=1)
            self.last_cache_hit=hit
            self.last_plan=plan
            if self.collect_memory:self.last_map_cpu_bytes=plan.cpu_bytes
            if self.implementation=='old':return plan.apply_numpy(prediction.cpu().numpy())
            output=plan.apply_tensor(prediction).cpu().numpy()
            if self.collect_memory:self.last_map_gpu_bytes=plan.device_bytes()
            return output
        if self.implementation=='old':
            output=self.api.native_reconstruct('airfoil',indices,prediction.cpu().numpy(),self.reference)
            return torch.as_tensor(output,device=self.runtime.device) if self.scope=='B_gpu_gpu' else output
        plan=self.api.build_plan('airfoil',self.reference,indices,reconstruction_id=self.gate['new_id'],channels=1)
        output=plan.apply_tensor(prediction)
        if self.collect_memory:self.last_map_cpu_bytes=plan.cpu_bytes;self.last_map_gpu_bytes=plan.device_bytes()
        return output if self.scope=='B_gpu_gpu' else output.cpu().numpy()

    @torch.no_grad()
    def __call__(self):
        runtime=self.runtime;runtime.clear_dynamic()
        if self.method=='full':
            indices=np.arange(11271,dtype=np.int64)
            prediction=self._prediction(indices,None,self.coords)
            self.last_indices=indices;self.last_compensation=None;self.last_coarse=None
            return prediction.to(torch.float64) if self.scope=='B_gpu_gpu' else prediction.cpu().numpy().astype(np.float64)
        # PI/CPG/CA require physical geometry on CPU. Scope B pays the actual D2H.
        coords=self.gpu_coords.cpu().numpy() if self.scope=='B_gpu_gpu' and self.method!='uniform' else self.coords
        sampling,coarse,_=self._prepare_sampling(coords)
        compensation=None if self.method=='uniform' else sampling.compensation
        prediction=self._prediction(sampling.indices,compensation,coords)
        output=self._reconstruct(sampling.indices,prediction)
        self.last_indices=sampling.indices;self.last_compensation=compensation;self.last_coarse=coarse
        return output

