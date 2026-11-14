"""Record environment, frozen model identities and one-time native FPS setup."""
from __future__ import annotations
import csv,hashlib,json,os,platform,subprocess,time,sys,inspect
from pathlib import Path
import torch,numpy,scipy
from car_request import OUT,E3,load_backbone


def command(args):
    result=subprocess.run(args,capture_output=True,text=True);return dict(command=args,returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())


def main():
    torch.set_num_threads(4);cache=OUT/'runtime/native_fps_cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ['ADASOLVER_NATIVE_CACHE']=str(cache)
    before=list(cache.glob('*.so'));started=time.perf_counter()
    from adasolver.inference.native_fps import load_native_fps
    load_native_fps();elapsed=(time.perf_counter()-started)*1000
    receipt=dict(native_fps_prepare_ms=elapsed,compiled_cache_existed_before=bool(before),cache=str(cache),
        artifacts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in cache.glob('*.so')],
        scope='one-time native FPS compile/load; no geometry or selected-set map is cached here')
    dest=OUT/'config/runtime_preparation.json'
    if not dest.exists():dest.write_text(json.dumps(receipt,indent=2)+'\n')
    else:assert json.loads(dest.read_text())['artifacts']==receipt['artifacts']
    costs={r['backbone']:r for r in csv.DictReader((E3/'results/backbone_training_cost.csv').open(newline=''))};rows=[]
    for arch in ['Transolver3','LinearNO']:
        runtime,model,context,_=load_backbone(arch,'cpu');meta=runtime.metadata;count=sum(p.numel() for p in model.parameters())
        assert count==int(costs[arch]['parameters'])
        cls=type(model);source=Path(inspect.getsourcefile(cls))
        rows.append(dict(backbone=arch,model_class=cls.__module__+'.'+cls.__name__,parameters=count,training_seed=0,
            checkpoint_sha256=meta['checkpoint_sha256'],normalizer_sha256=meta['normalizer_sha256'],
            epochs=int(meta['epochs']),train_samples=meta['train_samples'],validation_samples=meta['validation_samples'],
            global_batch_size=int(meta['global_batch_size']),optimizer_steps=int(meta['total_optimization_steps']),
            training_process_wall_seconds=float(costs[arch]['training_process_wall_seconds']),
            training_cost_provenance='existing frozen E3 training cost; no new Car model training',
            architecture_source=str(source),architecture_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            data_path=str(runtime.data_path),test_used_for_selection=meta['test_used_for_selection'],
            model_dtype=str(next(model.parameters()).dtype),variable_point_policy=meta['variable_n_policy']))
        del model,runtime
    with (OUT/'config/checkpoint_registry.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    env=dict(recorded_at_unix=time.time(),platform=platform.platform(),python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,
        torch_cuda=torch.version.cuda,torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv']),
        cpu=command(['lscpu']),load_average=os.getloadavg(),
        environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ADASOLVER_NATIVE_CACHE']})
    (OUT/'config/environment_cpu_preparation.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));print('Verified frozen Car model parameters and checkpoint/normalizer identities',flush=True)


if __name__=='__main__':main()
