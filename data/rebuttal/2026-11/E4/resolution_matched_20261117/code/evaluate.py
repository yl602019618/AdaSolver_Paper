"""Matched-resolution Darcy inference with common/native full-field metrics.

Sampling definitions are unchanged from E4. A local observation wrapper records
the actual coarse score and the pre-FPS patch allocation without changing them.
"""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_name]='4'
import argparse,ast,copy,csv,hashlib,json,shutil,sys,time,traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from common import OUT,CODE,ResolutionData,load_model,read,write,digest,array_digest,configure_precision
os.environ.setdefault('ADASOLVER_NATIVE_CACHE',str(OUT/'runtime/native_cache'))
from adasolver.layers import install_compensation
from adasolver.inference import adaptive
from adasolver.inference.cpg import DarcyCPG
from adasolver.inference.ca.darcy import DarcyCA
from adasolver.inference.pi.darcy import sample_darcy_pi,coefficient_interface_distance
from adasolver.inference.uniform import sample_uniform
from adasolver.utils.seed import make_numpy_rng

RESOLUTIONS=(85,141,211)
BUDGETS=(.375,.5,.75)
SAMPLING_SEEDS=(0,1,2)
TEST_IDS=tuple(range(1000,1200))
VALIDATION_IDS=(900,949,999)
EXAMPLE_IDS=(1000,1099,1199)
REGIONS=('boundary','interface','smooth')
METHODS=('full','uniform','pi','cpg','ca')
FIELDNAMES=['dataset','training_resolution','observation_resolution','evaluation_grid','evaluation_resolution',
    'case_id','method','budget_fraction','actual_budget_fraction','retained_points','coarse_points',
    'training_seed','head_seed','sampling_seed','metric','value','unit','checkpoint_sha256',
    'normalizer_sha256','head_sha256','protocol_sha256','selected_indices_sha256','coarse_indices_sha256',
    'compensation_sha256','physical_request_id','source_record','array_path','array_sha256',
    'reconstruction_id','metric_dtype','status']


def policy_protocol_check():
    p=read(OUT/'protocol.json')
    if p.get('resolutions')!=list(RESOLUTIONS):raise ValueError('Unexpected resolution protocol')
    if p.get('budgets')!=list(BUDGETS) or p.get('sampling_seeds')!=list(SAMPLING_SEEDS):
        raise ValueError('Unexpected retained-budget or sampling-seed protocol')
    if p.get('evaluation_resolution',211)!=211:raise ValueError('Common evaluation grid changed')
    if p.get('training_seed')!=0 or p.get('test_case_ids')!=list(TEST_IDS):raise ValueError('Training seed or evaluation population changed')
    for key,value in {'prior':'interface_distance','amp':.25,'ell':2/84,'beta':1.}.items():
        if p['sampling']['pi'][key]!=value:raise ValueError('Locked PI setting differs: '+key)
    for method in ['cpg','ca']:
        for key,value in {'coarse_fraction':.1,'patch_count':32,'kappa':2.,'beta':1.}.items():
            if p['sampling'][method][key]!=value:raise ValueError('Locked '+method+' setting differs: '+key)
    if p['sampling']['cpg']['acquisition']!='grad_lap' or not p['sampling']['coarse_in_final_budget']:
        raise ValueError('CPG acquisition or final coarse budget rule changed')
    return p


def scalar_original():
    path=OUT.parent/'rebuttal_20261112/experiments/E4_darcy_resolution/code/run_resolution.py'
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='reconstruct')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['reconstruct'],str(path),digest(path)


def reconstruct_pair(reference,selected,prediction,common_reference,full=False):
    """Use the E4 linear/nearest rule; queries can reuse one fixed triangulation."""
    values=np.asarray(prediction,dtype=np.float64).reshape(-1)
    source=reference[selected]
    if full:
        native=values.copy()
        if len(reference)==len(common_reference) and np.array_equal(reference,common_reference):
            return native,native,{'common':0,'native':0}
    else:native=None
    linear=LinearNDInterpolator(source,values,fill_value=np.nan);nearest=None
    def apply(query):
        nonlocal nearest
        field=np.asarray(linear(query),dtype=np.float64).reshape(-1)
        missing=~np.isfinite(field)
        if missing.any():
            if nearest is None:nearest=NearestNDInterpolator(source,values)
            field[missing]=nearest(query[missing])
        if not np.isfinite(field).all():raise ValueError('Nonfinite reconstructed field')
        return field,int(missing.sum())
    common,common_missing=apply(common_reference)
    if native is not None:native_missing=0
    elif np.array_equal(reference,common_reference):native=common;native_missing=common_missing
    else:native,native_missing=apply(reference)
    return common,native,{'common':common_missing,'native':native_missing}


def relative_l2(field,truth):
    a=np.asarray(field,dtype=np.float64).reshape(-1);b=np.asarray(truth,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape:raise ValueError((a.shape,b.shape))
    value=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
    if not np.isfinite(value):raise ValueError('Nonfinite physical field metric')
    return value


def physical_regions(common_coeff,common_coords):
    values=np.asarray(common_coeff).reshape(211,211);interface=np.zeros_like(values,dtype=bool)
    interface[1:]|=np.abs(values[1:]-values[:-1])>1e-6;interface[:-1]|=np.abs(values[1:]-values[:-1])>1e-6
    interface[:,1:]|=np.abs(values[:,1:]-values[:,:-1])>1e-6;interface[:,:-1]|=np.abs(values[:,1:]-values[:,:-1])>1e-6
    boundary=np.min(np.column_stack((common_coords,1-common_coords)),axis=1)<=.02
    near=(distance_transform_edt(~interface,sampling=1/210).ravel()<=.02)&~boundary
    return {'boundary':boundary,'interface':near,'smooth':~boundary&~near}


def array_save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)


def clone_rng(state):
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state);return rng


def snapshot_sources():
    names=['evaluate.py','common.py','head_support.py','verify_results.py']
    hashes={name:digest(CODE/name) for name in names}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    folder=OUT/'evaluation/source_snapshots'/identity
    folder.mkdir(parents=True,exist_ok=True)
    for name in names:
        target=folder/name
        if not target.exists():
            temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
            shutil.copyfile(CODE/name,temp);temp.replace(target)
        if digest(target)!=hashes[name]:raise ValueError('Evaluation source snapshot changed')
    write(folder/'manifest.json',dict(source_bundle_sha256=identity,files=hashes))
    return identity


class SamplingTrace:
    """Observe actual allocation calls and actual acquisition values."""
    def __init__(self,policy,coarse):
        self.policy=policy;self.coarse=coarse;self.score=None
        self.add=np.zeros(policy.patch_count,dtype=np.int64)
    def __enter__(self):
        self.fps=adaptive.fps_within
        def capture_fps(candidates,pos,already,n_add,*args,**kwargs):
            candidates=np.asarray(candidates,dtype=np.int64)
            if len(candidates):
                labels=np.unique(self.policy.patch_id[candidates]);assert len(labels)==1
                self.add[int(labels[0])]+=int(n_add)
            return self.fps(candidates,pos,already,n_add,*args,**kwargs)
        adaptive.fps_within=capture_fps
        self.attribute='_score' if isinstance(self.policy,DarcyCPG) else 'acquire'
        self.acquisition=getattr(self.policy,self.attribute)
        def capture_score(*args,**kwargs):
            value=self.acquisition(*args,**kwargs);self.score=np.asarray(value,dtype=np.float64).reshape(-1).copy();return value
        setattr(self.policy,self.attribute,capture_score)
        return self
    def __exit__(self,*exc):
        adaptive.fps_within=self.fps;setattr(self.policy,self.attribute,self.acquisition)
    def allocation(self,selected):
        pid=self.policy.patch_id;count=self.policy.patch_count
        base=np.bincount(pid[self.coarse],minlength=count)
        target=base+self.add;actual=np.bincount(pid[selected],minlength=count)
        if not np.array_equal(target,actual):raise ValueError('Observed pre-FPS allocation differs from final counts')
        patch_size=np.bincount(pid,minlength=count)
        return dict(coarse_patch_counts=base,target_patch_counts=target,actual_patch_counts=actual,
            patch_sizes=patch_size,target_allocation_rate=target[pid]/patch_size[pid],
            target_relative_density=target[pid]/patch_size[pid]/(len(selected)/len(pid)))


class Runtime:
    def __init__(self,resolution,device='cuda:0',require_head=True):
        self.n=int(resolution);self.device=torch.device(device);self.data=ResolutionData(self.n)
        self.source_bundle_sha256=snapshot_sources()
        self.model,self.manifest=load_model(self.n,self.device)
        self.context,_=install_compensation(self.model)
        self.reference=self.data.reference;self.positions=self.data.positions
        self.mean=float(self.data.y_mean.reshape(-1)[0]);self.std=float(self.data.y_std.reshape(-1)[0])
        self.query=np.load(OUT/'data/r211/reference_coords.npy',allow_pickle=False)
        self.obs_to_common=cKDTree(self.query).query(self.reference,k=1)[1]
        self.cpg=DarcyCPG(self.reference,patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.,acquisition='grad_lap')
        self.head_manifest={};self.ca=None
        if require_head:
            from head_support import load_head
            head,self.head_manifest=load_head(self.n,self.model,self.device)
            self.ca=DarcyCA(self.reference,head,patch_count=32,coarse_fraction=.1,kappa=2.,beta=1.)
        self.original_refine=adaptive.refine_once
        def native_refine(*args,**kwargs):
            kwargs['fps_distance_backend']='native';return self.original_refine(*args,**kwargs)
        adaptive.refine_once=native_refine
        self.loaded={}

    def close(self):adaptive.refine_once=self.original_refine

    def load_split(self,split):
        if split not in self.loaded:
            normalized,truth=self.data.load_split(split)
            coeff=np.load(self.data.directory/f'{split}_coeff.npy',mmap_mode='r')
            common_coeff=np.load(OUT/f'data/r211/{split}_coeff.npy',mmap_mode='r')
            common_truth=np.load(OUT/f'data/r211/{split}_solution.npy',mmap_mode='r')
            self.loaded[split]=(normalized,truth,coeff,common_coeff,common_truth)
        return self.loaded[split]

    @torch.no_grad()
    def forward(self,features,indices,weights=None,physical=True):
        self.context.clear()
        if weights is not None:self.context.set(torch.as_tensor(weights,dtype=torch.float32,device=self.device))
        try:
            if len(indices)==len(self.reference) and np.array_equal(indices,np.arange(len(indices))):
                pos=self.positions[None].to(self.device);fx=features[None].to(self.device)
            else:pos=self.positions[indices][None].to(self.device);fx=features[indices][None].to(self.device)
            result=self.model(pos,fx)[0].detach().cpu().numpy()
        finally:self.context.clear()
        if result.shape!=(len(indices),1) or result.dtype!=np.float32 or not np.isfinite(result).all():
            raise ValueError('Prediction does not match actual retained-point shape/FP32 field')
        return (result[:,0]*self.std+self.mean) if physical else result

    @torch.no_grad()
    def choose(self,sid,features,coeff,method,budget,seed,validate_tap=False):
        total=len(self.reference);count=total if method=='full' else int(round(total*budget))
        rng=make_numpy_rng(0 if seed is None else seed,sid)
        coarse=np.empty(0,dtype=np.int64);weights=None;score=np.ones(total,dtype=np.float64)
        support=np.arange(total,dtype=np.int64);allocation={};metadata={};coarse_prediction=np.empty((0,1),dtype=np.float32)
        if method=='full':
            selected=np.arange(total,dtype=np.int64)
            density=np.ones(total,dtype=np.float64);allocation_rate=np.ones(total,dtype=np.float64)
            semantics='identity/full grid'
        elif method=='uniform':
            sampling=sample_uniform(total,count,rng);selected=sampling.indices
            density=np.ones(total,dtype=np.float64);allocation_rate=np.full(total,count/total)
            semantics='uniform nominal inclusion rate; relative proposal density1'
        elif method=='pi':
            sampling=sample_darcy_pi(coeff,self.reference,count,rng,amp=.25,beta=1.,prior='interface_distance',ell=2/84,shape=(self.n,self.n))
            selected=sampling.indices;weights=sampling.compensation;metadata=sampling.metadata
            distance=coefficient_interface_distance(coeff,shape=(self.n,self.n));score=1.+.25*np.exp(-distance/(2/84))
            probability=score/score.sum();np.testing.assert_array_equal(sampling.sampling_density,probability[selected])
            np.testing.assert_array_equal(weights,(1./score[selected]).astype(np.float32))
            density=probability*total;allocation_rate=np.full(total,np.nan)
            semantics='relative nominal proposal density; not exact weighted-without-replacement inclusion probability'
        else:
            policy=self.cpg if method=='cpg' else self.ca
            if policy is None:raise RuntimeError('Corresponding resolution head is unavailable')
            coarse=rng.choice(total,policy.coarse_count(),replace=False).astype(np.int64)
            state=copy.deepcopy(rng.bit_generator.state)
            self.context.clear()
            if method=='cpg':
                coarse_prediction=self.forward(features,coarse,physical=False)
                args=(coarse,coarse_prediction,count)
            else:
                args=(coarse,self.positions[coarse][None].to(self.device),features[coarse][None].to(self.device),count)
            with SamplingTrace(policy,coarse) as trace:sampling=policy.refine(*args,rng)
            selected=sampling.indices;weights=sampling.compensation;metadata=sampling.metadata
            if not np.array_equal(selected[:len(coarse)],coarse):raise ValueError('Coarse points are not retained in the final prefix')
            if validate_tap:
                shadow=policy.refine(*args,clone_rng(state))
                np.testing.assert_array_equal(selected,shadow.indices);np.testing.assert_array_equal(weights,shadow.compensation)
                np.testing.assert_array_equal(sampling.sampling_density,shadow.sampling_density)
            support=coarse;score=trace.score
            if score is None or len(score)!=len(coarse):raise ValueError('Coarse acquisition was not captured')
            allocation=trace.allocation(selected);density=allocation['target_relative_density'];allocation_rate=allocation['target_allocation_rate']
            semantics='pre-FPS planned final patch counts divided by available patch nodes; relative density normalized by overall budget'
        assert len(selected)==count and len(np.unique(selected))==count
        if len(support)==total:lifted=score
        else:lifted=score[cKDTree(self.reference[support]).query(self.reference,k=1)[1]]
        relative_score=lifted/max(float(np.mean(score)),1e-12)
        indicator=np.zeros(total,dtype=np.uint8);indicator[selected]=1
        edges=np.linspace(0,1,17)
        observed_hist=np.histogram2d(self.reference[:,1],self.reference[:,0],bins=(edges,edges))[0]
        selected_hist=np.histogram2d(self.reference[selected,1],self.reference[selected,0],bins=(edges,edges))[0]
        retention_bins=np.divide(selected_hist,observed_hist,out=np.zeros_like(selected_hist),where=observed_hist>0)
        info=dict(coarse_indices=coarse,coarse_prediction=coarse_prediction,
            score_support_indices=support,score_support_values=score,relative_score=relative_score,
            target_relative_density=density,target_allocation_rate=allocation_rate,retained_indicator=indicator,
            bin_observed_count=observed_hist.astype(np.int64),bin_selected_count=selected_hist.astype(np.int64),
            bin_retained_fraction=retention_bins,target_density_semantics=semantics,metadata=metadata)
        info.update(allocation)
        return selected,weights,info

    def request(self,split,sid,method,budget,seed,phase,validate=False):
        x,y,coeff,common_coeff,common_truth=self.load_split(split)
        local=sid-1000 if split=='test' else sid-900
        features=x[local];native_truth=np.asarray(y[local]).reshape(-1)
        truth=np.asarray(common_truth[local]).reshape(-1)
        selected,weights,info=self.choose(sid,features,np.asarray(coeff[local]).reshape(-1),method,budget,seed,validate_tap=validate)
        prediction=self.forward(features,selected,weights)
        common,native,fallback=reconstruct_pair(self.reference,selected,prediction,self.query,full=method=='full')
        error_common=relative_l2(common,truth);error_native=relative_l2(native,native_truth)
        parity=None
        if validate:
            original,source,source_hash=scalar_original()
            expected_common=prediction.astype(np.float64) if method=='full' and self.n==211 else original(self.reference[selected],prediction,self.query)[0]
            expected_native=prediction.astype(np.float64) if method=='full' else original(self.reference[selected],prediction,self.reference)[0]
            dc=np.abs(common-expected_common);dn=np.abs(native-expected_native)
            assert np.all(dc<=1e-6+1e-5*np.abs(expected_common)) and np.all(dn<=1e-6+1e-5*np.abs(expected_native))
            parity=dict(passed=True,max_common_difference=float(dc.max()),max_native_difference=float(dn.max()),
                scalar_source=source,scalar_source_sha256=source_hash,trace_matches_untapped_sampling=method in ['cpg','ca'])
        masks=physical_regions(np.asarray(common_coeff[local]).reshape(-1),self.query)
        regions={}
        for name,mask in masks.items():
            observed=mask[self.obs_to_common];count=int(observed.sum());chosen=int(observed[selected].sum())
            support=observed[info['score_support_indices']];score_values=info['score_support_values'][support]
            allocation_values=info['target_allocation_rate'][observed]
            regions[name]=dict(evaluation_points=int(mask.sum()),observed_points=count,selected_points=chosen,
                retained_fraction=None if count==0 else chosen/count,selected_share=chosen/len(selected),
                support_score_points=len(score_values),mean_support_score=None if not len(score_values) else float(np.mean(score_values)),
                mean_display_relative_score=None if count==0 else float(np.mean(info['relative_score'][observed])),
                mean_target_relative_density=None if count==0 else float(np.mean(info['target_relative_density'][observed])),
                mean_target_allocation_rate=None if count==0 or not np.isfinite(allocation_values).all() else float(np.mean(allocation_values)),
                common_relative_l2=None if not mask.any() else relative_l2(common[mask],truth[mask]))
        seed_label='full' if seed is None else str(seed)
        array_path=OUT/f'records/r{self.n}/arrays/{split}_case{sid}_{method}_k{len(selected)}_s{seed_label}.npz'
        arrays=dict(indices=selected.astype(np.int64),retained_prediction_phys=prediction.astype(np.float32),
                    coarse_indices=info['coarse_indices'].astype(np.int64),compensation=np.empty(0,dtype=np.float32) if weights is None else weights,
                    coarse_score=info['score_support_values'] if method in ['cpg','ca'] else np.empty(0),
                    coarse_prediction_normalized=info['coarse_prediction'])
        if method in ['cpg','ca']:
            arrays.update({key:info[key] for key in ['coarse_patch_counts','target_patch_counts','actual_patch_counts','patch_sizes']})
        example_path=None
        if split=='test' and sid in EXAMPLE_IDS:
            example_path=OUT/f'records/r{self.n}/examples/case{sid}_{method}_k{len(selected)}_s{seed_label}.npz'
            example_arrays={**arrays,**{key:info[key] for key in ['score_support_indices','score_support_values','relative_score','target_relative_density',
                'target_allocation_rate','retained_indicator','bin_observed_count','bin_selected_count','bin_retained_fraction']},
                **dict(reference_coords=self.reference,coefficient=np.asarray(coeff[local]).reshape(-1),common_coords=self.query,
                common_truth=truth,common_prediction=common,native_prediction=native,native_truth=native_truth)}
            array_save(example_path,**example_arrays)
        array_save(array_path,**arrays)
        metadata=dict(info['metadata'])
        beta=None if method in ['full','uniform'] else 1.
        request_id=f'{split}/{sid}/r{self.n}/{method}/{len(selected)}/{seed_label}'
        return dict(request_id=request_id,split=split,case_id=sid,source_row=local if split=='test' else sid,
            training_resolution=self.n,observation_resolution=self.n,method=method,budget_fraction=budget,
            actual_budget_fraction=len(selected)/len(self.reference),retained_points=len(selected),coarse_points=len(info['coarse_indices']),
            training_seed=0,head_seed=0 if method=='ca' else None,sampling_seed=seed,beta=beta,
            field_relative_l2_common=error_common,field_relative_l2_native=error_native,
            output_shapes={'retained':list(prediction.shape),'common':list(common.shape),'native':list(native.shape)},
            checkpoint_sha256=self.manifest['checkpoint_sha256'],normalizer_sha256=self.data.normalizer_sha256,
            head_sha256=self.head_manifest.get('checkpoint_sha256',self.head_manifest.get('head_sha256')) if method=='ca' else None,
            protocol_sha256=digest(OUT/'protocol.json'),selected_indices_sha256=array_digest(selected.astype(np.int64)),
            coarse_indices_sha256=array_digest(info['coarse_indices'].astype(np.int64)),
            compensation_sha256='' if weights is None else array_digest(weights),
            region_definition='common211 physical width0.02, nearest assignment to native observations; fixed before predictions',
            regions=regions,sampling_metadata=metadata,target_density_semantics=info['target_density_semantics'],
            source_array=str(array_path.relative_to(OUT)),source_array_sha256=digest(array_path),
            example_array=None if example_path is None else str(example_path.relative_to(OUT)),
            example_array_sha256=None if example_path is None else digest(example_path),
            fallback_counts=fallback,validation_parity=parity,status='ok',evaluation_code_sha256=digest(__file__),
            evaluation_source_bundle_sha256=self.source_bundle_sha256,
            reconstruction_id='E4_linear_delaunay_nearest_original_rule',metric_dtype='float64')


def csv_export(resolution,phases):
    records=[]
    for phase in phases:
        for path in sorted((OUT/f'records/r{resolution}/{phase}').glob('case*.json')):
            case=read(path)
            if case['execution_status']!='completed':raise RuntimeError('Incomplete case snapshot')
            for row in case['records']:records.append((path,row))
    seen=set();rows=[];region_rows=[]
    for source,row in records:
        if row['request_id'] in seen:raise ValueError('Duplicate physical request')
        seen.add(row['request_id'])
        for grid,metric_n in [('common',211),('native',resolution)]:
            rows.append(dict(dataset='darcy',training_resolution=resolution,observation_resolution=resolution,
                evaluation_grid=grid,evaluation_resolution=metric_n,case_id=row['case_id'],method=row['method'],
                budget_fraction=row['budget_fraction'],actual_budget_fraction=row['actual_budget_fraction'],
                retained_points=row['retained_points'],coarse_points=row['coarse_points'],training_seed=0,
                head_seed=row['head_seed'],sampling_seed=row['sampling_seed'],beta=row['beta'],metric='field_relative_l2',
                value=row['field_relative_l2_'+grid],unit='ratio',checkpoint_sha256=row['checkpoint_sha256'],
                normalizer_sha256=row['normalizer_sha256'],head_sha256=row['head_sha256'],protocol_sha256=row['protocol_sha256'],
                selected_indices_sha256=row['selected_indices_sha256'],coarse_indices_sha256=row['coarse_indices_sha256'],
                compensation_sha256=row['compensation_sha256'],physical_request_id=row['request_id'],source_record=str(source.relative_to(OUT)),
                array_path=row['source_array'],array_sha256=row['source_array_sha256'],
                reconstruction_id=row['reconstruction_id'],metric_dtype=row['metric_dtype'],status=row['status'],
                evaluation_code_sha256=row['evaluation_code_sha256'],evaluation_source_bundle_sha256=row['evaluation_source_bundle_sha256']))
        for name,values in row['regions'].items():
            region_rows.append(dict(resolution=resolution,case_id=row['case_id'],method=row['method'],
                budget_fraction=row['budget_fraction'],actual_budget_fraction=row['actual_budget_fraction'],sampling_seed=row['sampling_seed'],
                region=name,target_density_semantics=row['target_density_semantics'],checkpoint_sha256=row['checkpoint_sha256'],
                physical_request_id=row['request_id'],**values))
    directory=OUT/f'records/r{resolution}'
    for filename,data in [('metrics_records.csv',rows),('region_records.csv',region_rows)]:
        target=directory/filename;temp=target.with_suffix('.tmp')
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
        temp.replace(target)
    return dict(unique_requests=len(records),metric_rows=len(rows),region_rows=len(region_rows),
        metrics_csv=str(directory/'metrics_records.csv'),metrics_sha256=digest(directory/'metrics_records.csv'))


def run(args):
    configure_precision(0,4);policy_protocol_check()
    if args.resolution not in RESOLUTIONS:raise ValueError(args.resolution)
    if args.shard_count<1 or not 0<=args.shard_index<args.shard_count:raise ValueError('Invalid case shard')
    case_ids=TEST_IDS[args.shard_index::args.shard_count]
    validation_ids=VALIDATION_IDS[args.shard_index::args.shard_count]
    if not validation_ids:raise ValueError('Every shard needs a fixed validation case')
    suffix=f'_shard{args.shard_index}of{args.shard_count}'
    methods={'baseline':['full','uniform','pi','cpg'],'ca':['ca'],'all':list(METHODS)}[args.phase]
    runtime=Runtime(args.resolution,args.device,require_head='ca' in methods)
    geometry=OUT/f'evaluation/r{args.resolution}_patch_geometry.npz'
    if geometry.exists():
        with np.load(geometry,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['reference'],runtime.reference)
            np.testing.assert_array_equal(saved['patch_id'],runtime.cpg.patch_id)
            np.testing.assert_array_equal(saved['patch_mass'],runtime.cpg.patch_mass)
    else:array_save(geometry,reference=runtime.reference,patch_id=runtime.cpg.patch_id,patch_mass=runtime.cpg.patch_mass)
    state=dict(execution_status='running',phase=args.phase,resolution=args.resolution,pid=os.getpid(),
        checkpoint_sha256=runtime.manifest['checkpoint_sha256'],normalizer_sha256=runtime.data.normalizer_sha256,
        head_sha256=runtime.head_manifest.get('checkpoint_sha256',runtime.head_manifest.get('head_sha256')),
        methods=methods,expected_cases=len(case_ids),completed_cases=0,completed_requests=0,
        shard_index=args.shard_index,shard_count=args.shard_count,case_ids=list(case_ids),
        evaluation_source_bundle_sha256=runtime.source_bundle_sha256,
        protocol_sha256=digest(OUT/'protocol.json'),evaluation_source_sha256=digest(__file__),
        geometry=str(geometry.relative_to(OUT)),geometry_sha256=digest(geometry),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    status=OUT/f'runtime/evaluation_r{args.resolution}_{args.phase}{suffix}.json';write(status,state);started=time.perf_counter()
    try:
        val_records=[]
        for sid in validation_ids:
            for method in methods:
                for budget in ([1.] if method=='full' else BUDGETS):
                    val_records.append(runtime.request('validation',sid,method,budget,None if method=='full' else 0,'validation_'+args.phase,validate=True))
        validation_path=OUT/f'evaluation/r{args.resolution}_{args.phase}{suffix}_validation.json'
        write(validation_path,dict(passed=True,checkpoint_sha256=runtime.manifest['checkpoint_sha256'],
            head_sha256=state['head_sha256'],validation_ids=list(validation_ids),records=val_records,
            source_sha256=digest(__file__),scope='new checkpoint full-shape, original reconstruction parity and sampling observation-tap parity'))
        if args.validate_only:
            state.update(execution_status='validation_only_completed',elapsed_seconds=time.perf_counter()-started,
                validation_receipt=str(validation_path.relative_to(OUT)))
            write(status,state);return
        expected_per_case=sum(1 if m=='full' else 9 for m in methods)
        for sid in case_ids:
            case_path=OUT/f'records/r{args.resolution}/{args.phase}/case{sid}.json'
            if case_path.exists():
                prior=read(case_path)
                assert prior['execution_status']=='completed' and prior['checkpoint_sha256']==state['checkpoint_sha256']
                assert prior['normalizer_sha256']==state['normalizer_sha256'] and prior['protocol_sha256']==state['protocol_sha256']
                assert prior['head_sha256']==state['head_sha256'];records=prior['records']
                assert len(records)==expected_per_case and {r['method'] for r in records}==set(methods)
                assert {r['case_id'] for r in records}=={sid}
                assert all(r['evaluation_source_bundle_sha256']==state['evaluation_source_bundle_sha256'] for r in records)
            else:
                records=[]
                for method in methods:
                    for budget in ([1.] if method=='full' else BUDGETS):
                        for seed in ([None] if method=='full' else SAMPLING_SEEDS):
                            records.append(runtime.request('test',sid,method,budget,seed,args.phase))
                write(case_path,dict(execution_status='completed',case_id=sid,checkpoint_sha256=state['checkpoint_sha256'],
                    normalizer_sha256=state['normalizer_sha256'],head_sha256=state['head_sha256'],protocol_sha256=state['protocol_sha256'],records=records))
            state['completed_cases']+=1;state['completed_requests']+=len(records)
            state['elapsed_seconds']=time.perf_counter()-started;write(status,state)
            if state['completed_cases']%10==0:print(json.dumps({k:state[k] for k in ['resolution','phase','completed_cases','completed_requests','elapsed_seconds']}),flush=True)
        assert state['completed_requests']==len(case_ids)*expected_per_case
        state.update(execution_status='completed',elapsed_seconds=time.perf_counter()-started);write(status,state)
        receipt=dict(status='complete',passed=True,resolution=args.resolution,phase=args.phase,
            shard_index=args.shard_index,shard_count=args.shard_count,case_ids=list(case_ids),case_count=len(case_ids),
            request_count=state['completed_requests'],metric_rows=state['completed_requests']*2,
            checkpoint_sha256=state['checkpoint_sha256'],head_sha256=state['head_sha256'],
            normalizer_sha256=state['normalizer_sha256'],protocol_sha256=state['protocol_sha256'],
            source_bundle_sha256=state['evaluation_source_bundle_sha256'],validation_receipt=str(validation_path.relative_to(OUT)))
        write(OUT/f'evaluation/r{args.resolution}_{args.phase}{suffix}_receipt.json',receipt)
        if args.shard_count==1:
            phases=['all'] if args.phase=='all' else ['baseline','ca'] if args.phase=='ca' else ['baseline']
            csv_export(args.resolution,phases)
            if args.phase in ['all','ca']:
                from verify_results import verify_resolution
                verify_resolution(args.resolution)
    except Exception:
        state.update(execution_status='failed',error=traceback.format_exc());write(status,state);raise
    finally:runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=RESOLUTIONS,required=True)
    parser.add_argument('--phase',choices=['baseline','ca','all'],default='all');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--shard-count',type=int,default=1);parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--validate-only',action='store_true');run(parser.parse_args())
