"""Run fixed-point PI beta candidates on calibration or locked final cases."""
from __future__ import annotations

import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='4'
import argparse,csv,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'AdaSolver_official'));sys.dont_write_bytecode=True
from adasolver.models import load_darcy_operator
from adasolver.layers import install_compensation
from adasolver.inference.pi.darcy import sample_darcy_pi,coefficient_interface_distance
from adasolver.inference.uniform import sample_uniform
from adasolver.utils.seed import make_numpy_rng
from inference_speed.gpu_monitor import ExclusiveGPU
from reconstruction import reconstruct_many

FIELDS=['split','case_id','source_row','resolution','method','beta','budget_fraction','retained_points',
        'sampling_seed','field_relative_l2','selected_indices_sha256','checkpoint_sha256','normalizer_id',
        'status','protocol_sha256','selection_sha256','compensation_sha256','reconstruction_id','runner_sha256',
        'output_dtype','legacy_metric_difference']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def grid(n):
    x,y=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    return np.column_stack((x.ravel(),y.ravel())).astype(np.float32).astype(np.float64)


def wait_for_gpu(index,phase):
    tag=f'{phase}_r{ARGS.resolution}'+(f'_s{ARGS.shard}' if ARGS.shards>1 else '')
    while True:
        monitor=ExclusiveGPU(index,OUT/f'runtime/{tag}_gpu.json',interval=2.)
        try:
            monitor.__enter__();return monitor
        except (BlockingIOError,RuntimeError) as error:
            if hasattr(monitor,'lock') and not monitor.lock.closed:monitor.lock.close()
            monitor.nvml.close()
            write(OUT/f'runtime/{tag}.json',dict(status='waiting_gpu',physical_gpu=index,pid=os.getpid(),reason=str(error)))
            time.sleep(10)


class Runtime:
    def __init__(self,protocol,n):
        self.protocol=protocol;self.n=n;self.ref=grid(n);self.query=grid(211)
        self.pos=torch.as_tensor(self.ref,dtype=torch.float32)
        self.norm=protocol['normalizers'];self.model=load_darcy_operator(protocol['checkpoint'],protocol['checkpoint_manifest'],device='cuda:0').requires_grad_(False).eval()
        self.context,_=install_compensation(self.model)
        self.legacy=read(OUT/'data_cache/legacy_reference.json')
        self.max_legacy_error=0.;self.legacy_checks=0

    @torch.no_grad()
    def prediction(self,fx,indices,weights=None):
        self.context.clear()
        if weights is not None:self.context.set(torch.as_tensor(weights,device='cuda:0',dtype=torch.float32))
        try:
            if len(indices)==len(self.pos) and np.array_equal(indices,np.arange(len(self.pos))):
                p=self.pos[None].cuda();x=fx[None].cuda()
            else:p=self.pos[indices][None].cuda();x=fx[indices][None].cuda()
            result=self.model(p,x)[0,:,0].cpu().numpy()*self.norm['sol']['std']+self.norm['sol']['mean']
        finally:self.context.clear()
        assert result.dtype==np.float32 and np.isfinite(result).all()
        return result

    def record(self,phase,sid,source_row,method,budget,seed,indices,field,truth,beta='',weights=None,selection_sha=''):
        error=float(np.linalg.norm(field-truth)/max(np.linalg.norm(truth),1e-12))
        assert np.isfinite(error)
        old_difference=''
        if phase in ['check','test'] and (method!='pi' or float(beta)==.5):
            key=f'{sid}/{self.n}/{method}/{len(indices)}/{0 if seed is None else seed}'
            old=self.legacy[key]
            assert array_sha(indices)==old['selected_indices_sha256'],('E4 selection mismatch',key)
            old_difference=abs(error-old['field_relative_l2_common'])
            assert old_difference<=1e-7+1e-5*abs(old['field_relative_l2_common']),('E4 baseline metric mismatch',key,old_difference)
            self.max_legacy_error=max(self.max_legacy_error,old_difference);self.legacy_checks+=1
        return dict(split=phase,case_id=sid,source_row=source_row,resolution=self.n,method=method,beta=beta,
            budget_fraction=budget,retained_points=len(indices),sampling_seed='' if seed is None else seed,
            field_relative_l2=error,selected_indices_sha256=array_sha(indices),checkpoint_sha256=self.protocol['checkpoint_sha256'],
            normalizer_id=self.protocol['normalizer_id'],status='ok',protocol_sha256=sha(OUT/'protocol.json'),
            selection_sha256=selection_sha,compensation_sha256='' if weights is None else array_sha(weights),
            reconstruction_id='E4_linear_nearest_common211_multichannel_equivalent_v1',runner_sha256=sha(__file__),
            output_dtype=str(field.dtype),legacy_metric_difference=old_difference)

    @torch.no_grad()
    def case(self,phase,sid,source_row,coefficient421,solution421,betas,selection_sha):
        stride={85:5,141:3,211:2}[self.n]
        coeff=np.asarray(coefficient421[::stride,::stride]).ravel()
        truth=np.asarray(solution421[::2,::2]).ravel()
        fx=torch.as_tensor((coeff-self.norm['coeff']['mean'])/self.norm['coeff']['std'],dtype=torch.float32).reshape(-1,1)
        indices=np.arange(self.n*self.n,dtype=np.int64)
        pred=self.prediction(fx,indices)
        field=pred if self.n==211 else reconstruct_many(self.ref,pred,self.query)[:,0]
        rows=[self.record(phase,sid,source_row,'full',1.,None,indices,field,truth,selection_sha=selection_sha)]
        checks=[]
        def old_field_check(method,k,new):
            original=ROOT/f'rebuttal_20261112/experiments/E4_darcy_resolution/records/examples/case{sid}_n{self.n}_{method}_k{k}.npz'
            old=np.load(original)['prediction'];delta=np.abs(new-old)
            assert np.all(delta<=1e-6+1e-5*np.abs(old)),('E4 full field mismatch',original)
            checks.append(dict(case_id=sid,method=method,retained_points=k,max_field_difference=float(delta.max()),
                relative_field_difference=float(np.linalg.norm(new-old)/max(np.linalg.norm(old),1e-30))))
        if phase=='check':old_field_check('full',len(indices),field)
        weight=1.+.25*np.exp(-coefficient_interface_distance(coeff,shape=(self.n,self.n))/(2/84))
        saved_ids=self.protocol['saved_validation_cases'] if phase=='validation' else self.protocol['saved_test_cases']
        for budget in self.protocol['budgets']:
            k=int(round(self.n*self.n*budget))
            for seed in ([0] if phase=='check' else self.protocol['sampling_seeds']):
                u=sample_uniform(self.n*self.n,k,make_numpy_rng(seed,sid));up=self.prediction(fx,u.indices)
                uf=reconstruct_many(self.ref[u.indices],up,self.query)[:,0]
                rows.append(self.record(phase,sid,source_row,'uniform',budget,seed,u.indices,uf,truth,selection_sha=selection_sha))
                if phase=='check':old_field_check('uniform',k,uf)
                sampled=sample_darcy_pi(coeff,self.ref,k,make_numpy_rng(seed,sid),amp=.25,beta=.5,prior='interface_distance',ell=2/84,shape=(self.n,self.n))
                ids=sampled.indices
                np.testing.assert_array_equal(sampled.compensation,np.power(1./weight[ids],.5).astype(np.float32))
                predictions=[];compensations=[]
                for beta in betas:
                    comp=np.power(1./weight[ids],beta).astype(np.float32)
                    if phase=='check':
                        exact=sample_darcy_pi(coeff,self.ref,k,make_numpy_rng(seed,sid),amp=.25,beta=beta,prior='interface_distance',ell=2/84,shape=(self.n,self.n))
                        np.testing.assert_array_equal(ids,exact.indices);np.testing.assert_array_equal(comp,exact.compensation)
                    predictions.append(self.prediction(fx,ids,comp));compensations.append(comp)
                retained=np.column_stack(predictions)
                fields=reconstruct_many(self.ref[ids],retained,self.query)
                for bi,beta in enumerate(betas):
                    rows.append(self.record(phase,sid,source_row,'pi',budget,seed,ids,fields[:,bi],truth,beta,compensations[bi],selection_sha))
                    if phase=='check' and beta==.5:old_field_check('pi',k,fields[:,bi])
                if phase!='check' and sid in saved_ids:
                    folder=OUT/'arrays'/phase;folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'r{self.n}_case{sid}_k{k}_s{seed}.npz',
                        indices=ids,betas=np.asarray(betas),raw_prior_weight=weight[ids],
                        compensation=np.column_stack(compensations),retained_prediction=retained,
                        uniform_indices=u.indices,uniform_retained_prediction=up)
        return rows,checks


def main():
    p=read(OUT/'protocol.json');phase=ARGS.phase;n=ARGS.resolution
    assert read(OUT/'reconstruction_validation.json')['passed']
    assert sha(p['checkpoint'])==p['checkpoint_sha256'] and sha(p['normalizer_file'])==p['normalizer_id']
    assert read(OUT/'data_cache/prepared.json')['protocol_sha256']==sha(OUT/'protocol.json')
    selection_sha='';betas=[.5] if phase=='check' else p['beta_candidates']
    if phase=='test':
        selection=read(OUT/'selection.json');selection_sha=sha(OUT/'selection.json')
        betas=selection['test_betas_by_resolution'][str(n)]
    cases=p['engineering_cases'] if phase=='check' else p['calibration_case_ids'] if phase=='validation' else p['test_case_ids']
    assert 0<=ARGS.shard<ARGS.shards
    if ARGS.shards>1:
        assert phase=='test','Sharding amendment only applies to final test execution'
        cases=[sid for index,sid in enumerate(cases) if index%ARGS.shards==ARGS.shard]
    dataset='validation' if phase=='validation' else 'test'
    coeff=np.load(OUT/f'data_cache/{dataset}_coeff_421.npy',mmap_mode='r')
    sol=np.load(OUT/f'data_cache/{dataset}_sol_421.npy',mmap_mode='r')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();all_rows=[];all_checks=[]
    state=dict(status='running',phase=phase,resolution=n,pid=os.getpid(),physical_gpu=ARGS.gpu,betas=betas,
        expected_cases=len(cases),completed_cases=0,metric_rows=0,protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=selection_sha)
    tag=f'{phase}_r{n}'+(f'_s{ARGS.shard}' if ARGS.shards>1 else '')
    state.update(shard=ARGS.shard,shards=ARGS.shards,case_ids=cases)
    state_path=OUT/f'runtime/{tag}.json'
    gpu=wait_for_gpu(ARGS.gpu,phase)
    try:
        runtime=Runtime(p,n)
        for sid in cases:
            source_row=sid-1000;local=source_row-400 if phase=='validation' else source_row
            case_path=OUT/f'records/cases/{phase}_r{n}_case{sid}.csv'
            if phase!='check' and case_path.exists():
                with case_path.open() as stream:rows=list(csv.DictReader(stream))
                expected=1+9*(1+len(betas));assert len(rows)==expected
                assert all(row['protocol_sha256']==state['protocol_sha256'] and row['selection_sha256']==selection_sha for row in rows)
                checks=[]
            else:
                rows,checks=runtime.case(phase,sid,source_row,coeff[local],sol[local],betas,selection_sha)
                if phase!='check':
                    case_path.parent.mkdir(parents=True,exist_ok=True);temp=case_path.with_suffix('.tmp')
                    with temp.open('w',newline='') as stream:
                        writer=csv.DictWriter(stream,FIELDS);writer.writeheader();writer.writerows(rows)
                    temp.replace(case_path)
            all_rows.extend(rows);all_checks.extend(checks)
            state.update(completed_cases=state['completed_cases']+1,metric_rows=len(all_rows),elapsed_seconds=time.perf_counter()-started,
                         baseline_checks=runtime.legacy_checks,max_legacy_metric_difference=runtime.max_legacy_error)
            write(state_path,state)
            if state['completed_cases']%10==0 or phase=='check':print(json.dumps({k:state[k] for k in ['phase','resolution','completed_cases','expected_cases','metric_rows','elapsed_seconds']}),flush=True)
            gpu.assert_exclusive()
        if phase=='check':
            write(OUT/f'results/engineering_r{n}.json',dict(passed=True,scope='fixed baseline0.5 and old full-field artifacts; no candidate selection',checks=all_checks,
                  max_legacy_metric_difference=runtime.max_legacy_error,baseline_requests=runtime.legacy_checks))
        elif ARGS.shards==1:
            target=OUT/f'records/{phase}_r{n}.csv';temp=target.with_suffix('.tmp')
            with temp.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,FIELDS);writer.writeheader();writer.writerows(all_rows)
            temp.replace(target)
        state.update(status='completed',elapsed_seconds=time.perf_counter()-started);write(state_path,state)
    except Exception:
        state.update(status='failed',traceback=traceback.format_exc());write(state_path,state);raise
    finally:gpu.__exit__(None,None,None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--phase',choices=['check','validation','test'],required=True)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--shard',type=int,default=0)
    ARGS=parser.parse_args();main()
