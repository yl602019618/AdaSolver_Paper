"""Frozen Darcy cross-resolution evaluation with persistent per-request records."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
OFF=ROOT/'AdaSolver_official'
sys.path.insert(0,str(OFF));sys.dont_write_bytecode=True
os.environ.setdefault('ADASOLVER_NATIVE_CACHE',str(OUT/'records/native_fps_cache'))
DATA=Path('/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy')
CHECKPOINT=OFF/'checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_best.pt'
MANIFEST=OFF/'checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_manifest.json'
HEAD=OFF/'checkpoints/darcy_ca/darcy_ca_head_grad_lap_cf010_official_best.pt'
HEAD_MANIFEST=OFF/'checkpoints/darcy_ca/darcy_ca_head_grad_lap_cf010_official_manifest.json'
BUDGETS=(.375,.5,.75)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+f'.tmp{os.getpid()}');temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,path)
def grid(n):
 x,y=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
 return np.column_stack((x.ravel(),y.ravel())).astype(np.float32).astype(np.float64)

def compatible_head_manifest():
 metadata=json.loads(HEAD_MANIFEST.read_text())
 if sha(MANIFEST)==metadata['operator_manifest_sha256']:return HEAD_MANIFEST
 original=ROOT/'import_records/before_path_relocation/checkpoints/darcy'/MANIFEST.name
 assert sha(original)==metadata['operator_manifest_sha256']
 old=json.loads(original.read_text());current=json.loads(MANIFEST.read_text())
 differing=[key for key in set(old)|set(current) if old.get(key)!=current.get(key)]
 assert set(differing)<=set(['checkpoint','last_checkpoint','normalizer','command']),differing
 metadata['operator_manifest']=str(original)
 local=OUT/'config/darcy_ca_head_relocated.json';write(local,metadata)
 write(OUT/'config/manifest_relocation_audit.json',dict(original=str(original),current=str(MANIFEST),original_sha256=sha(original),current_sha256=sha(MANIFEST),changed_fields=differing,semantic_fields_equal=True,checkpoint_sha256=sha(CHECKPOINT),head_sha256=sha(HEAD)))
 return local

def prepare():
 import scipy.io,torch
 from adasolver.data.pipe import UnitTransformer
 torch.set_num_threads(4)
 cache=OUT/'records/data_cache';cache.mkdir(parents=True,exist_ok=True)
 if (cache/'complete.json').exists():
  print('Data cache already prepared');return
 train_path=DATA/'piececonst_r421_N1024_smooth1.mat'
 test_path=DATA/'piececonst_r421_N1024_smooth2.mat'
 train=scipy.io.loadmat(train_path,variable_names=['coeff','sol'])
 normalizers={}
 for key in ['coeff','sol']:
  v=torch.from_numpy(np.array(train[key][:1000,::5,::5],copy=True)).float().reshape(1000,-1,1)
  normalizer=UnitTransformer(v)
  normalizers[key]=dict(mean=float(normalizer.mean),std=float(normalizer.std))
 del train,v
 archive=np.load(OFF/'checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_ynorm.npz')
 for k in ['mean','std']:np.testing.assert_allclose(normalizers['sol'][k],archive[k],atol=1e-7,rtol=0)
 test=scipy.io.loadmat(test_path,variable_names=['coeff','sol'])
 assert test['coeff'].shape==(1024,421,421) and test['sol'].shape==(1024,421,421)
 for key in ['coeff','sol']:np.save(cache/f'{key}_421.npy',np.asarray(test[key][:200],dtype=np.float32))
 del test
 write(cache/'normalizers.json',normalizers)
 mapping=[dict(case_id=1000+i,source_file=test_path.name,source_row=i,split_history_status='historical_test_informed_reevaluation') for i in range(200)]
 write(OUT/'config/sample_ids.json',mapping)
 provenance=dict(source_train=str(train_path),source_test=str(test_path),source_shapes=[1024,421,421],
  source_train_sha256=sha(train_path),source_test_sha256=sha(test_path),normalizers=normalizers,
  checkpoint_sha256=sha(CHECKPOINT),head_sha256=sha(HEAD),observation_strides={'85':5,'141':3,'211':2},
  common_evaluation_resolution=211,preparation_finished_at=now())
 write(cache/'complete.json',provenance)
 write(OUT/'config/protocol_lock.json',dict(experiment_id='E4',cases=200,sampling_seeds=[0,1,2],methods=['full','uniform','pi','cpg','ca'],observation_resolutions=[85,141,211],evaluation_resolution=211,budgets=list(BUDGETS),fixed_count=5419,example_source_rows=[0,99,199],expected_unique_requests=27000,protocol_sha256=sha(OUT/'protocol.md'),provenance=provenance))
 (OUT/'data_provenance.md').write_text('# Darcy observation provenance\n\nActual smooth2 421×421 MAT fields, rows 0–199. Observations use original nodal values at strides5/3/2. Truth is the direct 211×211 stride2 field. No synthetic high-resolution input. Normalizers are fit once from smooth1 first1000 cases at original85×85 resolution and numerically match the saved output normalizer.\n\nHistorical model/strategy selection used these smooth2 cases. This is an explicitly test-informed re-evaluation. Frozen CA context-size transfer is reported, not presented as retraining or newly independent validation.\n')
 print(json.dumps(provenance,indent=2))

def reconstruct(source,values,query):
 from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
 linear=LinearNDInterpolator(source,values,fill_value=np.nan)
 result=np.asarray(linear(query)).reshape(-1)
 missing=~np.isfinite(result)
 if missing.any():result[missing]=NearestNDInterpolator(source,values)(query[missing])
 assert np.isfinite(result).all()
 return result,int(missing.sum())

def regions(coeff,coords):
 from scipy.ndimage import distance_transform_edt
 values=coeff.reshape(211,211);interface=np.zeros_like(values,dtype=bool)
 interface[1:]|=np.abs(values[1:]-values[:-1])>1e-6;interface[:-1]|=np.abs(values[1:]-values[:-1])>1e-6
 interface[:,1:]|=np.abs(values[:,1:]-values[:,:-1])>1e-6;interface[:,:-1]|=np.abs(values[:,1:]-values[:,:-1])>1e-6
 boundary=np.min(np.column_stack((coords,1-coords)),axis=1)<=.02
 near=(distance_transform_edt(~interface,sampling=1/210).ravel()<=.02)&~boundary
 return {'boundary':boundary,'interface':near,'smooth':~boundary&~near}

def run(args):
 import torch
 from scipy.spatial import cKDTree
 from adasolver.models import load_darcy_operator
 from adasolver.layers import install_compensation
 from adasolver.inference.pi import sample_darcy_pi
 from adasolver.inference.cpg import DarcyCPG
 from adasolver.inference.ca import DarcyCA,load_darcy_ca_head
 from adasolver.inference.uniform import sample_uniform
 from adasolver.inference import adaptive
 from adasolver.utils import budget_count
 from adasolver.utils.seed import make_numpy_rng
 torch.set_num_threads(args.threads)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch_device=torch.device(args.device)
 def sync():
  if torch_device.type=='cuda':torch.cuda.synchronize(torch_device)
 cache=OUT/'records/data_cache'
 normalizers=json.loads((cache/'normalizers.json').read_text())
 coeff=np.load(cache/'coeff_421.npy',mmap_mode='r');sol=np.load(cache/'sol_421.npy',mmap_mode='r')
 model=load_darcy_operator(CHECKPOINT,MANIFEST,device=torch_device).eval()
 checkpoint_hash=sha(CHECKPOINT);head_hash=sha(HEAD)
 context,patched=install_compensation(model)
 head=load_darcy_ca_head(model,HEAD,compatible_head_manifest(),device=torch_device)
 original_refine=adaptive.refine_once
 def refine_native(*a,**kw):
  kw['fps_distance_backend']='native';return original_refine(*a,**kw)
 adaptive.refine_once=refine_native
 eval_coords=grid(211);eval_tree=cKDTree(eval_coords)
 output=OUT/'records'/f'part{args.shard}{"_smoke" if args.smoke else ""}.jsonl'
 output.parent.mkdir(parents=True,exist_ok=True)
 done={}
 if output.exists():
  for line in output.open():
   r=json.loads(line)
   if r['status']=='ok':done[r['request_id']]=r
 status_path=OUT/'records'/f'part{args.shard}{"_smoke" if args.smoke else ""}_status.json'
 started=time.time();successful=0;failed=0;seen=set()
 with output.open('a',buffering=1) as stream,torch.no_grad():
  for size in [85,141,211]:
   stride={85:5,141:3,211:2}[size];reference=grid(size);N=len(reference)
   t=time.perf_counter();cpg=DarcyCPG(reference,patch_count=32,coarse_fraction=.1,kappa=2,beta=1,acquisition='grad_lap')
   ca=DarcyCA(reference,head,patch_count=32,coarse_fraction=.1,kappa=2,beta=1)
   preparation_seconds=time.perf_counter()-t
   reference_tensor=torch.as_tensor(reference,dtype=torch.float32)
   reg_obs_ids=eval_tree.query(reference,k=1)[1]
   counts={budget_count(N,b):[f'fraction_{b}'] for b in BUDGETS}
   counts.setdefault(5419,[]).append('fixed_5419')
   for local in range(200):
    if local%args.shards!=args.shard:continue
    if args.smoke and local!=args.shard:continue
    case_id=1000+local;coefficient=np.asarray(coeff[local,::stride,::stride]).ravel();target_native=np.asarray(sol[local,::stride,::stride]).ravel()
    target_eval=np.asarray(sol[local,::2,::2]).ravel();common_coeff=np.asarray(coeff[local,::2,::2]).ravel()
    fx=torch.as_tensor((coefficient-normalizers['coeff']['mean'])/normalizers['coeff']['std'],dtype=torch.float32).reshape(1,N,1)
    region_masks=regions(common_coeff,eval_coords)
    for method in ['full','uniform','pi','cpg','ca']:
     for keep,tags in ({N:['full']} if method=='full' else counts).items():
      for seed in ([0] if method=='full' else [0,1,2]):
       if args.smoke and (seed!=0 or (method!='full' and keep!=budget_count(N,.375))):continue
       identity=f'{case_id}/{size}/{method}/{keep}/{seed}';seen.add(identity)
       if identity in done:continue
       row=dict(request_id=identity,experiment_id='E4',evaluator_version='v1_1_equivalent_native_reuse',case_id=case_id,source_row=local,observation_resolution=size,evaluation_resolution=211,
        method=method,retained_points=keep,budget_fraction=keep/N,condition_tags=tags,training_seed=0,head_seed=0 if method=='ca' else None,sampling_seed=seed if method!='full' else None,
        split_history_status='historical_test_informed_reevaluation',checkpoint_sha256=checkpoint_hash,head_sha256=head_hash if method=='ca' else None,
        preparation_seconds=preparation_seconds,precision='float32',status='running')
       try:
        rng=make_numpy_rng(seed,case_id);context.clear();sync()
        if torch_device.type=='cuda':torch.cuda.reset_peak_memory_stats(torch_device)
        begin=time.perf_counter();acquisition_seconds=0.;coarse=np.array([],dtype=np.int64)
        if method=='full': indices=np.arange(N,dtype=np.int64);weights=np.ones(N,dtype=np.float32);metadata={}
        elif method=='uniform':
         sampling=sample_uniform(N,keep,rng);indices=sampling.indices;weights=sampling.compensation;metadata=sampling.metadata
        elif method=='pi':
         sampling=sample_darcy_pi(coefficient,reference,keep,rng,amp=.25,beta=.5,prior='interface_distance',ell=2/84,shape=(size,size));indices=sampling.indices;weights=sampling.compensation;metadata=sampling.metadata
        else:
         policy=cpg if method=='cpg' else ca
         coarse=rng.choice(N,policy.coarse_count(),replace=False)
         pos0=reference_tensor[coarse][None].to(torch_device);fx0=fx[:,coarse].to(torch_device)
         t=time.perf_counter()
         if method=='cpg':
          prediction0=model(pos0,fx0)[0].cpu().numpy();sampling=cpg.refine(coarse,prediction0,keep,rng)
         else:sampling=ca.refine(coarse,pos0,fx0,keep,rng)
         sync();acquisition_seconds=time.perf_counter()-t
         indices=sampling.indices;weights=sampling.compensation;metadata=sampling.metadata
        selection_seconds=time.perf_counter()-begin
        pos=reference_tensor[indices][None].to(torch_device);x=fx[:,indices].to(torch_device)
        if method not in ['full','uniform']:context.set(torch.as_tensor(weights,device=torch_device))
        sync();t=time.perf_counter();pred=model(pos,x)[0,:,0].cpu().numpy()*normalizers['sol']['std']+normalizers['sol']['mean'];sync()
        infer_seconds=time.perf_counter()-t;context.clear();t=time.perf_counter()
        if method=='full' and size==211:pred_eval=pred;fallback=0
        else:pred_eval,fallback=reconstruct(reference[indices],pred,eval_coords)
        reconstruction_seconds=time.perf_counter()-t;sync();total=time.perf_counter()-begin
        if size==211:pred_native=pred_eval
        elif method=='full':pred_native=pred
        else:pred_native,_=reconstruct(reference[indices],pred,reference)
        error=float(np.linalg.norm(pred_eval-target_eval)/max(np.linalg.norm(target_eval),1e-12));native_error=float(np.linalg.norm(pred_native-target_native)/max(np.linalg.norm(target_native),1e-12))
        row.update(status='ok',field_relative_l2_common=error,field_relative_l2_native=native_error,coarse_count=len(coarse),acquisition_seconds=acquisition_seconds,selection_seconds=selection_seconds,
          final_inference_seconds=infer_seconds,reconstruction_seconds=reconstruction_seconds,total_seconds=total,timing_scope='one_direct_accuracy_request_not_repeated_latency',interpolation_fallback_count=fallback,
          selected_indices_sha256=hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(),sampling_metadata=metadata,
          peak_gpu_bytes=torch.cuda.max_memory_allocated(torch_device) if torch_device.type=='cuda' else 0)
        regional={}
        for name,mask in region_masks.items():
         obs_region=mask[reg_obs_ids];selected_count=int(obs_region[indices].sum())
         regional[name]=dict(evaluation_points=int(mask.sum()),observed_region_points=int(obs_region.sum()),selected_points=selected_count,
          retained_fraction=selected_count/max(int(obs_region.sum()),1),relative_l2=float(np.linalg.norm((pred_eval-target_eval)[mask])/max(np.linalg.norm(target_eval[mask]),1e-12)))
        row['regions']=regional
        if local in [0,99,199] and seed==0:
         destination=OUT/'records/examples'/f'case{case_id}_n{size}_{method}_k{keep}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
         np.savez_compressed(destination,reference_coords=reference,coefficient=coefficient,indices=indices,compensation=weights,coarse_indices=coarse,evaluation_coords=eval_coords,prediction=pred_eval,truth=target_eval)
         row['example_path']=str(destination.relative_to(OUT))
        successful+=1
       except Exception as exc:
        context.clear();row.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());failed+=1
       stream.write(json.dumps(row)+'\n');done[identity]=row
       write(status_path,dict(execution_status='running',successful_new=successful,failed_new=failed,completed_total=sum(r['status']=='ok' for r in done.values()),latest=identity,elapsed_seconds=time.time()-started,updated_at=now()))
       print(identity,row['status'],row.get('field_relative_l2_common'),flush=True)
  write(status_path,dict(execution_status='completed' if failed==0 else 'partial',evidence_status='pending',successful_new=successful,failed_new=failed,expected_this_run=len(seen),completed_total=sum(r['status']=='ok' for r in done.values()),elapsed_seconds=time.time()-started,updated_at=now()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);a=p.parse_args()
 if a.prepare:prepare()
 else:run(a)

if __name__=='__main__':main()
