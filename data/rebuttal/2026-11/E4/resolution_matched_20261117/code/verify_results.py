"""Independent matrix, artifact and full-field checks for matched Darcy."""
from __future__ import annotations

import argparse,ast,csv,hashlib,json,math,os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_name]='4'
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

OUT=Path(__file__).resolve().parents[1]
os.environ.setdefault('ADASOLVER_NATIVE_CACHE',str(OUT/'runtime/native_cache'))
RESOLUTIONS=(85,141,211)
BUDGETS=(.375,.5,.75)
METHODS=('full','uniform','pi','cpg','ca')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def scalar_reconstruct(coords,values,query):
    field=np.asarray(LinearNDInterpolator(coords,values,fill_value=np.nan)(query)).reshape(-1)
    missing=~np.isfinite(field)
    if missing.any():field[missing]=NearestNDInterpolator(coords,values)(query[missing])
    return field


def rel(field,truth):
    a=np.asarray(field,dtype=np.float64).reshape(-1);b=np.asarray(truth,dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def verify_resolution(resolution,replay_cases=(1000,1099,1199)):
    n=int(resolution);directory=OUT/f'records/r{n}';path=directory/'metrics_records.csv'
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==14800
    model=read(OUT/f'checkpoints/r{n}/manifest.json');head=read(OUT/f'checkpoints/r{n}/head/manifest.json')
    assert model['execution_status']=='completed' and model['epochs_completed']==1000
    assert model['total_optimization_steps']==225000 and model['test_used_for_selection'] is False
    assert model['training_ids']==list(range(900)) and model['validation_ids']==list(range(900,1000))
    assert digest(OUT/f'checkpoints/r{n}/best.pt')==model['checkpoint_sha256']
    norm_hash=digest(OUT/f'data/r{n}/normalizer.npz');assert model['normalizer_sha256']==norm_hash
    assert head['execution_status']=='completed' and head['resolution']==n
    assert head['operator_sha256']==model['checkpoint_sha256'] and head['normalizer_sha256']==norm_hash
    head_hash=head.get('checkpoint_sha256',head.get('head_sha256'))
    assert digest(OUT/f'checkpoints/r{n}/head/best.pt')==head_hash
    expected={(sid,m,b,s,g) for sid in range(1000,1200) for m in METHODS
              for b in ([1.] if m=='full' else BUDGETS)
              for s in ([None] if m=='full' else [0,1,2]) for g in ['common','native']}
    indexed={};physical=defaultdict(list)
    protocol_hash=digest(OUT/'protocol.json')
    for row in rows:
        sid=int(row['case_id']);method=row['method'];budget=float(row['budget_fraction'])
        seed=None if row['sampling_seed']=='' else int(row['sampling_seed']);grid=row['evaluation_grid']
        key=(sid,method,budget,seed,grid);assert key not in indexed;indexed[key]=row
        assert row['dataset']=='darcy' and int(row['training_resolution'])==int(row['observation_resolution'])==n
        assert int(row['evaluation_resolution'])==(211 if grid=='common' else n)
        assert row['checkpoint_sha256']==model['checkpoint_sha256'] and row['normalizer_sha256']==norm_hash
        assert row['protocol_sha256']==protocol_hash and row['training_seed']=='0'
        assert row['head_sha256']==(head_hash if method=='ca' else '')
        assert row['head_seed']==('0' if method=='ca' else '')
        assert row['metric']=='field_relative_l2' and row['unit']=='ratio' and row['metric_dtype']=='float64'
        assert row['status']=='ok' and math.isfinite(float(row['value'])) and float(row['value'])>=0
        assert int(row['retained_points'])==round(n*n*budget)
        assert abs(float(row['actual_budget_fraction'])-int(row['retained_points'])/(n*n))<1e-14
        assert int(row['coarse_points'])==(round(.1*n*n) if method in ['cpg','ca'] else 0)
        physical[row['physical_request_id']].append(row)
    assert set(indexed)==expected and len(physical)==7400
    ref=np.load(OUT/f'data/r{n}/reference_coords.npy');query=np.load(OUT/'data/r211/reference_coords.npy')
    target=np.load(OUT/f'data/r{n}/test_solution.npy',mmap_mode='r')
    common_target=np.load(OUT/'data/r211/test_solution.npy',mmap_mode='r')
    geometry=np.load(OUT/f'evaluation/r{n}_patch_geometry.npz');pid=geometry['patch_id']
    common_coeff=np.load(OUT/'data/r211/test_coeff.npy',mmap_mode='r')
    observed_to_common=cKDTree(query).query(ref,k=1)[1]
    replay_differences=[];arrays_checked=0;case_sources={};region_cache={};source_bundles=set()
    for identity,views in physical.items():
        assert len(views)==2 and {r['evaluation_grid'] for r in views}=={'common','native'}
        row=next(r for r in views if r['evaluation_grid']=='common');sid=int(row['case_id']);method=row['method'];k=int(row['retained_points'])
        source=OUT/row['source_record']
        if source not in case_sources:
            case_sources[source]=read(source);assert case_sources[source]['execution_status']=='completed'
        raw=next(r for r in case_sources[source]['records'] if r['request_id']==identity)
        assert raw['status']=='ok' and raw['selected_indices_sha256']==row['selected_indices_sha256']
        assert raw['beta']==(None if method in ['full','uniform'] else 1.)
        assert raw['normalizer_sha256']==norm_hash and raw['checkpoint_sha256']==model['checkpoint_sha256']
        assert raw['evaluation_code_sha256']==row['evaluation_code_sha256']
        assert raw['evaluation_source_bundle_sha256']==row['evaluation_source_bundle_sha256']
        source_bundles.add(raw['evaluation_source_bundle_sha256'])
        array_path=OUT/row['array_path'];assert digest(array_path)==row['array_sha256']
        with np.load(array_path,allow_pickle=False) as values:
            ids=values['indices'];pred=values['retained_prediction_phys'];coarse=values['coarse_indices'];weights=values['compensation']
            assert ids.dtype==np.int64 and ids.shape==(k,) and len(np.unique(ids))==k
            assert int(ids.min())>=0 and int(ids.max())<n*n
            assert pred.dtype==np.float32 and pred.shape==(k,) and np.isfinite(pred).all()
            assert array_digest(ids)==row['selected_indices_sha256']
            assert array_digest(coarse)==row['coarse_indices_sha256']
            if method=='full':assert np.array_equal(ids,np.arange(n*n))
            if method in ['cpg','ca']:
                assert coarse.shape==(round(.1*n*n),) and np.array_equal(coarse,ids[:len(coarse)])
                assert len(values['coarse_score'])==len(coarse) and np.isfinite(values['coarse_score']).all()
                assert np.array_equal(values['target_patch_counts'],values['actual_patch_counts'])
                assert np.array_equal(values['actual_patch_counts'],np.bincount(pid[ids],minlength=32))
                assert np.array_equal(values['coarse_patch_counts'],np.bincount(pid[coarse],minlength=32))
                assert int(values['target_patch_counts'].sum())==k
            else:assert len(coarse)==0
            if method in ['full','uniform']:assert len(weights)==0 and row['compensation_sha256']==''
            else:assert weights.dtype==np.float32 and weights.shape==(k,) and np.isfinite(weights).all() and (weights>0).all() and array_digest(weights)==row['compensation_sha256']
            if sid in replay_cases:
                pc=pred.astype(np.float64) if method=='full' and n==211 else scalar_reconstruct(ref[ids],pred,query)
                pn=pred.astype(np.float64) if method=='full' else pc if n==211 else scalar_reconstruct(ref[ids],pred,ref)
                errors={'common':rel(pc,common_target[sid-1000]),'native':rel(pn,target[sid-1000])}
                for view in views:
                    delta=abs(float(view['value'])-errors[view['evaluation_grid']]);assert delta<=1e-10+1e-8*abs(errors[view['evaluation_grid']])
                    replay_differences.append(delta)
            arrays_checked+=1
        if n==211:
            assert abs(float(views[0]['value'])-float(views[1]['value']))<1e-14
        if raw['example_array'] is not None:
            ep=OUT/raw['example_array'];assert digest(ep)==raw['example_array_sha256']
            with np.load(ep,allow_pickle=False) as a:
                indicator=a['retained_indicator'];assert indicator.shape==(n*n,) and int(indicator.sum())==k
                assert np.array_equal(np.flatnonzero(indicator),np.sort(ids))
                assert a['relative_score'].shape==a['target_relative_density'].shape==(n*n,)
                assert np.isfinite(a['relative_score']).all() and np.isfinite(a['target_relative_density']).all()
                assert abs(float(a['target_relative_density'].mean())-1)<1e-12
                assert int(a['bin_selected_count'].sum())==k and int(a['bin_observed_count'].sum())==n*n
        regions=raw['regions'];assert set(regions)=={'boundary','interface','smooth'}
        if sid not in region_cache:
            coefficient=np.asarray(common_coeff[sid-1000]).reshape(211,211)
            edge=np.zeros_like(coefficient,dtype=bool)
            for axis in [0,1]:
                changes=np.abs(np.diff(coefficient,axis=axis))>1e-6
                if axis==0:edge[:-1]|=changes;edge[1:]|=changes
                else:edge[:,:-1]|=changes;edge[:,1:]|=changes
            boundary=np.min(np.column_stack((query,1-query)),axis=1)<=.02
            interface=(distance_transform_edt(~edge,sampling=1/210).ravel()<=.02)&~boundary
            region_cache[sid]={'boundary':boundary,'interface':interface,'smooth':~boundary&~interface}
        assert sum(r['evaluation_points'] for r in regions.values())==211**2
        assert sum(r['observed_points'] for r in regions.values())==n*n
        assert sum(r['selected_points'] for r in regions.values())==k
        for name,r in regions.items():
            mask=region_cache[sid][name];observed=mask[observed_to_common]
            assert r['evaluation_points']==int(mask.sum()) and r['observed_points']==int(observed.sum())
            assert r['selected_points']==int(observed[ids].sum())
            assert abs(r['selected_share']-int(observed[ids].sum())/k)<1e-14
            if r['observed_points']:
                assert abs(r['retained_fraction']-r['selected_points']/r['observed_points'])<1e-14
            else:assert r['retained_fraction'] is None
    assert len(source_bundles)==1
    for bundle in source_bundles:
        folder=OUT/'evaluation/source_snapshots'/bundle;manifest=read(folder/'manifest.json')
        assert manifest['source_bundle_sha256']==bundle
        assert hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()==bundle
        for filename,hash_value in manifest['files'].items():assert digest(folder/filename)==hash_value
    result=dict(passed=True,status='complete_matched_resolution_accuracy',resolution=n,training_seed=0,
        checkpoint_sha256=model['checkpoint_sha256'],normalizer_sha256=norm_hash,head_sha256=head_hash,
        test_cases=200,physical_requests=7400,metric_rows=len(rows),source_arrays_checked=arrays_checked,
        common_evaluation_resolution=211,native_evaluation_resolution=n,methods=list(METHODS),
        validation_only_checkpoint_selection=True,coarse_prefix_included=True,pre_FPS_target_counts_preserved=True,
        region_masks_and_selected_counts_independently_recomputed=True,source_bundle_sha256=list(source_bundles)[0],
        independent_replay_case_ids=list(replay_cases),replayed_scalar_metrics=len(replay_differences),
        max_replayed_metric_difference=max(replay_differences),raw_csv=str(path),raw_csv_sha256=digest(path),
        region_csv_sha256=digest(directory/'region_records.csv'),
        validator_sha256=digest(__file__),formal_latency_verified=False)
    write(OUT/f'results/r{n}_accuracy_acceptance.json',result)
    return result


def preflight_sampling():
    """Changed observation-tap and paired-query paths, using engineering fixtures."""
    import sys
    sys.path.insert(0,str(OUT/'code'))
    from common import ResolutionData
    from evaluate import SamplingTrace,reconstruct_pair,scalar_original
    from adasolver.inference import adaptive
    from adasolver.inference.cpg import DarcyCPG
    from adasolver.utils.seed import make_numpy_rng
    original_refine=adaptive.refine_once
    def native(*a,**kw):kw['fps_distance_backend']='native';return original_refine(*a,**kw)
    adaptive.refine_once=native
    original,path,path_hash=scalar_original();records=[]
    try:
        for n in RESOLUTIONS:
            data=ResolutionData(n);reference=data.reference;query=np.load(OUT/'data/r211/reference_coords.npy')
            policy=DarcyCPG(reference,patch_count=32,coarse_fraction=.1,kappa=2,beta=1,acquisition='grad_lap')
            rng=make_numpy_rng(0,900);coarse=rng.choice(n*n,round(.1*n*n),replace=False)
            prediction=(np.sin(3.1*reference[coarse,0])+np.cos(2.7*reference[coarse,1]))[:,None].astype(np.float32)
            import copy
            state=copy.deepcopy(rng.bit_generator.state)
            count=round(n*n*.375)
            with SamplingTrace(policy,coarse) as trace:observed=policy.refine(coarse,prediction,count,rng)
            rng2=np.random.default_rng();rng2.bit_generator.state=state
            baseline=policy.refine(coarse,prediction,count,rng2)
            assert np.array_equal(observed.indices,baseline.indices) and np.array_equal(observed.compensation,baseline.compensation)
            allocation=trace.allocation(observed.indices)
            assert abs(float(allocation['target_relative_density'].mean())-1)<1e-12
            value=(np.sin(reference[observed.indices,0])+.37*np.cos(reference[observed.indices,1])).astype(np.float32)
            common,native_field,_=reconstruct_pair(reference,observed.indices,value,query)
            expected_common=original(reference[observed.indices],value,query)[0]
            expected_native=original(reference[observed.indices],value,reference)[0]
            dc=float(np.max(np.abs(common-expected_common)));dn=float(np.max(np.abs(native_field-expected_native)))
            assert np.all(np.abs(common-expected_common)<=1e-6+1e-5*np.abs(expected_common))
            assert np.all(np.abs(native_field-expected_native)<=1e-6+1e-5*np.abs(expected_native))
            records.append(dict(resolution=n,native_points=n*n,coarse_points=len(coarse),retained_points=count,
                observation_tap_indices_identical=True,compensation_identical=True,target_allocation_counts_match=True,
                max_common_difference=dc,max_native_difference=dn,synthetic_field=True))
    finally:adaptive.refine_once=original_refine
    result=dict(passed=True,scope='CPU engineering validation on actual native reference grids; synthetic field values are not accuracy results',
        original_scalar_source=path,original_scalar_source_sha256=path_hash,records=records)
    write(OUT/'evaluation/policy_reconstruction_preflight.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=RESOLUTIONS)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight_sampling(),indent=2))
    elif args.resolution:
        from evaluate import csv_export
        phases=['all'] if (OUT/f'records/r{args.resolution}/all').exists() else ['baseline','ca']
        csv_export(args.resolution,phases)
        print(json.dumps(verify_resolution(args.resolution),indent=2))
    else:parser.error('resolution or preflight required')
