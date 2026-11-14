"""Independent CPU QA for the native Car/Airfoil score visualization bundles.

Reads original archived arrays and captured CSV/NPZ without changing either.
Writes only the requested QA receipt. No model forward or GPU request is made.
"""
from __future__ import annotations
import argparse,ast,csv,datetime,hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORKSPACE=None
AIRFOIL_DATA=None
QA_CONTRACT=HERE/'qa_contract.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}


def resolve(value,base):
    p=Path(value)
    if p.is_absolute():return p
    candidates=[x for x in [base/p,WORKSPACE/p] if x.exists()]
    if not candidates:raise FileNotFoundError(value)
    if len(candidates)>1 and candidates[0].resolve()!=candidates[1].resolve():raise ValueError('Ambiguous path '+value)
    return candidates[0]


def equal(a,b):return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)
def require(condition,message):
    if not bool(condition):raise AssertionError(message)


class Audit:
    def __init__(self,manifest):
        self.path=manifest;self.base=manifest.parent;self.manifest_bytes=manifest.read_bytes();self.manifest_sha256=hashlib.sha256(self.manifest_bytes).hexdigest();self.m=json.loads(self.manifest_bytes);self.checks=[];self.inputs=[]
        self.car=self.m['dataset'].lower() in ['shapenet-car','shapenet_car']
        self.local_hashes={k:v['sha256'] for k,v in self.m.get('files',{}).items() if isinstance(v,dict) and v.get('sha256')}
        if self.m.get('public_artifact_manifest'):
            public=read(self.base/self.m['public_artifact_manifest'])
            for item in public.get('public_scalar_csvs',[]):self.local_hashes[item['path']]=item['sha256']
            for item in public.get('external_artifacts',[]):self.local_hashes[item['artifact_id'].split('/')[-1]]=item['sha256']
    def note(self,name,status='PASS',**details):self.checks.append(dict(check=name,status=status,**details))
    def checked(self,name,fn):
        try:
            result=fn() or {};self.note(name,**result)
        except Exception as e:self.note(name,'FAIL',error=str(e),error_class=type(e).__name__)
    def file(self,path,expected=None):
        p=resolve(path,self.base);digest=sha(p)
        if expected is None and p.is_relative_to(self.base):expected=self.local_hashes.get(p.relative_to(self.base).as_posix())
        if expected:require(digest==expected,'file SHA differs: '+str(p))
        self.inputs.append(dict(path=str(p.relative_to(WORKSPACE)) if p.is_relative_to(WORKSPACE) else p.name,sha256=digest,bytes=p.stat().st_size))
        return p
    def geometry(self):
        if self.car:
            e5=WORKSPACE/'rebuttal_20261112/experiments/E5_training_seeds'
            sys.path.insert(0,str(e5/'code/vendor'))
            from adasolver.data.shapenet_car_runtime import CarRuntime
            binding=e5/'checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json'
            metadata=read(binding);require(metadata['checkpoint_sha256']==self.m['checkpoint_sha256'],'geometry binding checkpoint')
            sample=CarRuntime(binding).sample(self.m['case_id'])
            self.native_sample=sample
            # The data loader possesses truth, but neither geometry nor score checks use it.
            return sample.pos.numpy(),sample.surface_mask.numpy()
        require(AIRFOIL_DATA is not None,'Airfoil geometry requires --airfoil-data or ADASOLVER_AIRFOIL_DATA')
        root=AIRFOIL_DATA
        x=np.load(root/'NACA_Cylinder_X.npy',mmap_mode='r')[int(self.m['case_id'])]
        y=np.load(root/'NACA_Cylinder_Y.npy',mmap_mode='r')[int(self.m['case_id'])]
        return np.stack((x,y),axis=-1).reshape(-1,2).astype(np.float32),None
    def source_checks(self):
        sources=dict(self.m.get('source_functions_sha256',{}));sources.update(self.m.get('source_code',{}))
        for spec in self.m.get('source_functions',{}).values():sources[spec['path']]=spec['file_sha256']
        require(bool(sources),'no source-function hash records')
        if self.m.get('public_artifact_manifest'):self.file(self.m['public_artifact_manifest'])
        for path,digest in sources.items():self.file(path,digest)
        capture=self.base/('capture.py' if self.car else 'capture_scores.py')
        expected=self.m.get('capture_script_sha256')
        if expected is None and self.m.get('capture_source_identity'):
            observed=self.m['capture_source_identity'];expected=observed.get('sha256')
            if observed.get('executed_source_file_hash')=='not_captured_before_run':self.note('capture_execution_identity','PARTIAL',reason='Current source hash verified; pre-execution capture-script hash was not recorded',source_identity=observed)
        self.file(capture,expected)
        if expected is None:self.note('capture_execution_identity','PARTIAL',reason='Capture file current hash recorded by QA; executed-source hash not supplied in scalar capture_script_sha256 field')
        return dict(source_files=len(sources),capture_current_sha256=sha(capture))
    def method(self,name,spec,xyz,surface):
        n=len(xyz);npz=self.file(spec.get('npz') or spec['arrays']);a=arrays(npz)
        native_path=self.file(spec.get('source_npz') or spec['native_npz'],spec.get('source_npz_sha256') or spec.get('native_npz_sha256'));native=arrays(native_path)
        if spec.get('source_manifest'):self.file(spec['source_manifest'],spec.get('source_manifest_sha256'))
        csv_path=self.file(spec.get('perpoint_csv') or spec['csv'])
        with csv_path.open() as f:
            reader=csv.DictReader(f);headers=reader.fieldnames;raw=list(reader)
        cols=spec['columns'];idx=np.array([int(r[cols['index']]) for r in raw])
        require(len(raw)==n and np.array_equal(idx,np.arange(n)),f'{name}: expected one row per native point in native order')
        coordinates=np.array([[float(r[k]) for k in cols['xyz']] for r in raw])
        captured_xyz=a['xyz' if self.car else 'coords'];require(equal(captured_xyz,xyz),'capture coordinates differ from original native geometry')
        require(equal(coordinates.astype(xyz.dtype),xyz),'CSV coordinates do not round-trip to original native dtype')
        mapped=0
        for key in headers:
            if key in [cols['index'],*cols['xyz']]:continue
            values=np.array([float(r[key]) if r[key] else np.nan for r in raw])
            if key in ['selected_order','coarse_order']:
                native_order=(native['indices'] if key=='selected_order' else native.get('coarse_indices',np.empty(0))).astype(np.int64)
                require(equal(values[native_order],np.arange(len(native_order))),'CSV ordered-rank mapping '+key)
                absent=np.ones(n,bool);absent[native_order]=False
                require(np.all(np.isnan(values[absent])|(values[absent]==-1)),'undefined order encoding '+key)
            elif key in ['coarse_pressure','ca_standardized_output'] and not self.car:
                if key not in a:require(np.isnan(values).all(),'absent coarse/head field populated '+key)
                else:
                    ids=native['coarse_indices'].astype(np.int64);expected=a[key].reshape(-1)
                    require(equal(values[ids].astype(expected.dtype),expected),'CSV coarse/head value mapping '+key)
                    absent=np.ones(n,bool);absent[ids]=False;require(np.isnan(values[absent]).all(),'coarse/head field filled outside coarse set')
            elif key in ['coarse_pressure_proxy','coarse_head_output_raw','coarse_input_pressure_cpu'] and self.car:
                aliases={'coarse_pressure_proxy':'coarse_input_pressure_cpu','coarse_head_output_raw':'ca_head_output_raw_cpu'}
                source=a.get(key,a.get(aliases.get(key,'')))
                if source is None:require(np.isnan(values).all(),'inapplicable coarse/head value populated')
                else:
                    ids=native.get('coarse_indices',np.empty(0)).astype(np.int64);expected=source.reshape(-1)
                    require(equal(values[ids].astype(expected.dtype),expected),'Car CSV coarse/head value map '+key)
                    absent=np.ones(n,bool);absent[ids]=False;require(np.isnan(values[absent]).all(),'Car coarse field outside support')
            elif key=='normalized_prior' and not self.car:
                if name=='uniform':expected=np.full(n,1/n)
                elif name=='pi':expected=a['geometry_prior']/a['geometry_prior'].sum()
                else:expected=np.full(n,np.nan)
                require(equal(values,expected),'derived normalized prior differs from explicit Uniform/PI rule')
            elif key in ['coarse_feature_l2','coarse_feature_mean','coarse_feature_std']:
                features=a.get('ca_block0_features_cpu' if self.car else 'ca_features')
                if features is None:require(np.isnan(values).all(),'feature summary populated without own CA features')
                else:
                    matrix=np.asarray(features,dtype=np.float64).reshape(len(native['coarse_indices']),-1)
                    expected=np.linalg.norm(matrix,axis=1) if key.endswith('l2') else matrix.mean(axis=1) if key.endswith('mean') else matrix.std(axis=1,ddof=0)
                    ids=native['coarse_indices'].astype(np.int64);require(equal(values[ids],expected),'native feature descriptive summary '+key)
                    absent=np.ones(n,bool);absent[ids]=False;require(np.isnan(values[absent]).all(),'CA feature summary outside coarse support')
            else:
                require(key in a,'CSV column has no matching captured array: '+key)
                require(values.shape==a[key].shape and equal(values,a[key]),'CSV/NPZ mismatch '+key)
            mapped+=1
        original_indices=native['indices'].astype(np.int64);coarse=native.get('coarse_indices',np.empty(0)).astype(np.int64)
        exported_indices=a['actual_selected_indices' if self.car else 'indices'].astype(np.int64)
        require(equal(original_indices,exported_indices),'captured ordered selected list differs from archived actual indices')
        require(equal(a.get('coarse_indices',np.empty(0)).astype(np.int64),coarse),'captured coarse ordered indices differ')
        selected=np.zeros(n,bool);selected[original_indices]=True;coarse_mask=np.zeros(n,bool);coarse_mask[coarse]=True
        require(equal(a['selected'],selected) and equal(a['coarse'],coarse_mask),'per-point selected/coarse mask differs from original indices')
        require(len(original_indices)==spec['selected_count'] and len(np.unique(original_indices))==len(original_indices),'selected cardinality')
        declared=spec.get('actual_selected_ordered_sha256') or spec.get('actual_selected_indices_sha256')
        require(array_sha(original_indices)==declared,'ordered selected hash differs')
        declared_coarse=spec.get('actual_coarse_ordered_sha256') or spec.get('actual_coarse_indices_sha256')
        if len(coarse) or declared_coarse:require(array_sha(coarse)==declared_coarse,'ordered coarse hash differs')
        density_column='compensation_density_selected' if self.car else 'reported_sampling_density'
        weight_column='compensation_weight_selected' if self.car else 'compensation'
        require(equal(a[weight_column][original_indices],native['compensation']),'actual selected compensation differs')
        require(np.isnan(a[weight_column][~selected]).all(),'undefined compensation was filled outside selected support')
        require(np.isnan(a[density_column][~selected]).all(),'undefined native density was filled outside selected support')
        if self.car:require(equal(a[density_column][original_indices],native['sampling_density']),'native selected density differs')
        self.note(name+'.native_point_identity',points=n,selected=len(original_indices),coarse=len(coarse),coarse_retained=int(selected[coarse].sum()),csv_fields_exact=mapped,coordinate_dtype=str(xyz.dtype),selected_ordered_sha256=declared)
        sem=spec.get('field_semantics',self.m.get('field_semantics',{}))
        if not sem:self.note(name+'.field_semantics','PARTIAL',reason='No explicit per-field support/normalization metadata')
        else:self.note(name+'.field_semantics',fields=sem)
        if self.car:self.car_fields(name,spec,a,native,original_indices,coarse,surface,native_path)
        else:self.airfoil_fields(name,spec,a,native,original_indices,coarse,native_path)
    def car_fields(self,name,spec,a,native,indices,coarse,surface,native_path):
        cfg=spec['configuration'];n=len(surface);count=len(indices)
        if name in ['uniform','pi']:
            for field in ['raw_score_cpu','normalized_score_native','normalized_score_cpu']:require(np.isnan(a[field]).all(),'inapplicable model score populated '+field)
            if name=='uniform':require(equal(a['raw_prior'],np.ones(n)) and equal(a['allocation_density'],np.full(n,count/n)),'uniform prior/allocation')
            else:
                require(equal(a['raw_prior'],(a['distance_to_surface']<cfg['r']).astype(float)),'PI hard-shell prior differs')
                require(np.isnan(a['allocation_density']).all(),'PI undefined full density was fabricated')
                require(np.isclose(a['native_sampling_density'].sum(),1.,atol=1e-12,rtol=0),'PI selected weights not sum1 as declared')
                if 'target_density' not in a:self.note(name+'.paper_density_mapping','PARTIAL',reason='Native hard-shell prior and selected normalization verified; no full-domain lambda mapping is supplied')
            self.note(name+'.model_score','NOT_APPLICABLE',reason='Only explicitly named prior is present')
        else:
            require(equal(a['normalized_score_native'],native['score']),'archived score changed')
            raw=np.maximum(a['raw_score_cpu'],0);expected=np.maximum(raw/max(float(np.quantile(raw,.95)),1e-12),cfg['score_floor'])**cfg['score_power']
            require(equal(a['normalized_score_cpu'],expected),'CPU raw->normalized native rule differs')
            from adasolver.inference.cpg.shapenet_car import surface_scalar_score
            rebuilt=surface_scalar_score(a['xyz'],surface,coarse,a['coarse_input_pressure_cpu'],feature=cfg['feature'],k=cfg['neighborhood'],surface_band=cfg['surface_band'])
            require(equal(rebuilt,a['raw_score_cpu']),'Car coarse pressure -> native raw-score function mismatch')
            difference=np.abs(a['normalized_score_cpu']-native['score']);bound=1e-6+1e-5*np.abs(native['score']);violations=int((difference>bound).sum())
            self.note(name+'.CPU_replay_parity','PASS' if violations==0 else 'PARTIAL',violations=violations,max_absolute_difference=float(difference.max()),relative_norm_difference=float(np.linalg.norm(difference)/np.linalg.norm(native['score'])),atol=1e-6,rtol=1e-5,archived_score_preserved=True)
            strata=a['stratum'];mass=a['stratum_mass'];counts=a['stratum_selected_counts']
            require(equal(np.bincount(strata,minlength=len(mass)),mass),'stratum native reference mass')
            require(equal(np.bincount(strata[indices],minlength=len(mass)),counts),'stratum actual selected count')
            allocation=counts[strata]/mass[strata];require(equal(allocation,a['allocation_density']),'allocation is not actual quota/reference count')
            require(equal(allocation[indices],native['sampling_density']),'allocation does not match archived selected density')
            require(np.isclose(allocation.sum(),count,atol=1e-9,rtol=0),'full allocation should sumK')
            scale=max(float(np.quantile(native['score'][surface],.95)),1e-12);physics=np.clip(native['score']/scale,0,1)
            priority=(1+cfg['wall_amplitude']*np.exp(-a['distance_to_surface']/cfg['wall_scale']))*(1+cfg['physics_strength']*physics)
            require(equal(physics,a['physics_term']) and equal(priority,a['allocation_priority']),'paper-priority weight transform differs')
            self.note(name+'.paper_priority_formula',anchor='eq:app_density_car' if name=='cpg' else 'eq:app_ca_car_density',unnormalized_weight_match=True)
            if 'target_density' not in a:self.note(name+'.paper_density_mapping','PARTIAL',reason='Native stratum allocation is sumK and selected density is not the paper mean-one full-reference lambda; no substituted global density is accepted',allocation_sum=float(allocation.sum()))
            if name=='ca':
                require(a['ca_block0_features_cpu'].shape==(len(coarse),256),'CA feature/coarse alignment')
                h=read(WORKSPACE/'rebuttal_20261112/experiments/E5_training_seeds/checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json')
                require(h['head_sha256']==self.m['head_sha256'] and h['operator_sha256']==self.m['checkpoint_sha256'],'CA head/backbone identity')
                decoded=a['ca_head_output_raw_cpu']*h['target_std']+h['target_mean']
                require(np.allclose(decoded,a['ca_head_pressure_physical_cpu'],atol=1e-7,rtol=1e-6),'CA standardized->physical decoding')
                self.note(name+'.feature_head_lineage',feature_shape=list(a['ca_block0_features_cpu'].shape),physical_decode_checked=True,head_sha256=h['head_sha256'])
        if 'normalized_allocation_density' in a and name!='pi':
            p=a['normalized_allocation_density'];require(np.allclose(p,a['allocation_density']/count,atol=1e-18,rtol=1e-13),'normalized quota measure is not allocation/K')
            require(np.isclose(p.sum(),1,atol=1e-12,rtol=0),'derived quota measure sum1')
            if 'target_to_uniform_reference_ratio' in a:
                ratio=a['target_to_uniform_reference_ratio'];require(np.allclose(ratio,n*p,atol=1e-14,rtol=1e-13) and np.isclose(ratio.mean(),1,atol=1e-12,rtol=0),'derived quota/reference ratio')
                if name in ['cpg','ca']:
                    reference_correction=ratio[indices]**(-cfg['beta']);native_weight=native['compensation'].astype(float)
                    factor=(count/n)**cfg['beta']
                    require(np.allclose(reference_correction,factor*native_weight,atol=1e-7,rtol=1e-6),'quota-measure inverse ratio does not differ by claimed common scalar')
                    self.note(name+'.derived_quota_measure',sum_p=float(p.sum()),mean_reference_ratio=float(ratio.mean()),inverse_correction_global_factor=factor,interpretation='derived from realized integer quotas; not equivalent to priority t/mean(t) or independent point inclusion probabilities',model_global_scale_forward_tested=False)
        if 'target_density' in a:
            if name=='uniform':weight=np.ones(n)
            elif name=='pi':
                shell=a['distance_to_surface']<cfg['r'];gamma=cfg['frac_extra'];weight=(1-gamma)+gamma*shell/shell.mean()
            else:weight=a['allocation_priority']
            target=weight/weight.sum();require(np.allclose(target,a['target_density'],atol=1e-18,rtol=1e-13),'designed target probability does not match paper/native prior weight')
            require(np.allclose(n*target,a['target_reference_ratio'],atol=1e-13,rtol=1e-13),'paper reference-density ratio N*p')
            require(np.isclose(target.sum(),1,atol=1e-12,rtol=0) and np.isclose(a['target_reference_ratio'].mean(),1,atol=1e-12,rtol=0),'designed target normalization')
            self.note(name+'.designed_paper_target',sum_probability=float(target.sum()),mean_reference_ratio=float(a['target_reference_ratio'].mean()),formula='t/sum(t) and N*t/sum(t)',source='uniform prior' if name=='uniform' else 'frozen hard-shell mixture; explicit design distribution' if name=='pi' else 'native wall-and-score priority from archived score',actual_FPS_or_fallback_inclusion_probability_claimed=False)
        raw_path=native_path.parent.parent/'raw.jsonl'
        original=next(r for r in map(json.loads,raw_path.read_text().splitlines()) if r['sample_id']==self.m['case_id'] and r['seed']==self.m['sampling_seed'] and r['budget_fraction']==self.m['budget'])
        self.file(raw_path)
        for key,value in spec['native_online_generation_seconds'].items():require(value==original[key] and np.isfinite(value) and value>=0,'native timing not bound to raw record '+key)
        self.note(name+'.generation_cost',source='original native raw-record stage seconds',stage_seconds=spec['native_online_generation_seconds'],CPU_observer_wall_seconds=spec['cpu_trace_seconds'],limitation='Native stage boundaries only; no claim of isolated GPU busy time or repeated request latency')
    def airfoil_fields(self,name,spec,a,native,indices,coarse,native_path):
        cfg=spec['config'];eligible=a['eligible'];selected=a['selected'];n=len(selected)
        require(np.isfinite(a['proposal_probability']).all() and (a['proposal_probability']>=0).all(),'proposal must be finite nonnegative')
        require(np.all(a['proposal_probability'][~eligible]==0),'forced/excluded points should have zero additional-draw proposal probability')
        require(np.isnan(a['sampling_weight'][~eligible]).all(),'undefined sampling weight filled outside eligible support')
        require(np.isclose(a['proposal_probability'][eligible].sum(),1,atol=1e-12,rtol=0),'eligible proposal sum not1')
        expected=a['sampling_weight'][eligible]/a['sampling_weight'][eligible].sum();require(np.allclose(expected,a['proposal_probability'][eligible],atol=1e-15,rtol=1e-13),'proposal differs from actual eligible weight normalization')
        if name in ['cpg','ca']:
            sys.path.insert(0,str(WORKSPACE/'linearno_followup_20261114/standard_benchmarks/airfoil/code/vendor'))
            from adasolver.inference.cpg.airfoil2d_features import pressure_gradient,pressure_acquisition
            mandatory=np.flatnonzero(a['mandatory'])
            coarse_gradient=pressure_gradient(a['coords'][coarse],a['coarse_pressure'],cfg['feature_k'])
            require(equal(coarse_gradient,a['coarse_raw_gradient_score']),'Airfoil coarse pressure->LSQ gradient mismatch')
            rebuilt=pressure_acquisition(a['coords'],coarse,a['coarse_pressure'],mandatory,feature=cfg['feature'],k=cfg['feature_k'],ell=cfg['ell'],pressure_offset=cfg['pressure_offset'])
            require(equal(rebuilt,a['raw_score']),'Airfoil coarse-pressure gradient lift -> raw score mismatch')
            score=np.maximum(a['raw_score'],0);expected=score/max(float(np.quantile(score,.95)),1e-12)
            require(equal(a['normalized_score'],expected),'raw->q95 score differs')
            full_weights=(cfg['uniform_floor']+expected)**cfg['score_gamma']
            require(equal(a['sampling_weight'][eligible],full_weights[eligible]),'gamma-powered eligible sampling weight differs')
            reported=(cfg['uniform_floor']+expected[indices])/np.sum(cfg['uniform_floor']+expected)
            require(equal(a['reported_sampling_density'][indices],reported),'native reported density differs from source formula')
            retained=int(selected[coarse].sum())
            self.note(name+'.paper_score_weight_formula',anchor='eq:app_density_airfoil_ratio' if name=='cpg' else 'eq:app_ca_airfoil_density',weight_formula_match=True,eligible_conditional_probability_match=True)
            if retained!=len(coarse):self.note(name+'.paper_coarse_retention','PARTIAL',coarse_points=len(coarse),retained=retained,reason='Native weighted configuration retains mandatory surface only; fixed paper prose says S0 is retained')
            else:self.note(name+'.paper_coarse_retention',coarse_points=len(coarse),retained=retained)
            if name=='ca':
                require(a['ca_features'].shape==(1,len(coarse),128) and a['ca_bottleneck'].shape==(1,len(coarse),8) and a['ca_standardized_output'].shape==(1,len(coarse),1),'CA feature/head shapes')
                h=read(resolve(cfg['head_manifest'],self.base));require(h['head_sha256']==self.m['head_sha256'] and h['operator_sha256']==self.m['checkpoint_sha256'],'CA head/backbone binding')
                self.file(cfg['head_checkpoint'],h['head_sha256'])
                decoded=a['ca_standardized_output'][0].astype(np.float64)*h['target_std']+h['target_mean']
                require(np.allclose(decoded,a['coarse_pressure'],atol=1e-7,rtol=1e-6),'CA decoded coarse pressure differs')
                self.note(name+'.feature_head_lineage',feature_shape=list(a['ca_features'].shape),bottleneck_shape=list(a['ca_bottleneck'].shape),bottleneck_semantics='raw first linear-layer output, before GELU',decoded_pressure_checked=True)
        else:
            require(np.isnan(a['raw_score']).all() and np.isnan(a['normalized_score']).all(),'inapplicable model score populated')
            self.note(name+'.model_score','NOT_APPLICABLE',reason='No model refinement score for this native sampler')
            if name=='pi':
                expected=1+cfg['amp']*np.exp(-a['surface_distance']/cfg['ell']);require(equal(expected,a['geometry_prior']),'PI geometry exponential prior differs')
                require(equal(a['sampling_weight'][eligible],expected[eligible]),'PI eligible weights differ from prior')
                self.note(name+'.paper_prior_formula',anchor='eq:app_pi_airfoil_density',same_unnormalized_weight=True)
        if name!='uniform':
            mass=a['represented_mass'][indices];realized=mass.mean()/mass
            require(np.allclose(realized,a['realized_density'][indices],atol=1e-14,rtol=1e-13),'realized selected coverage formula differs')
            require(np.allclose(realized**(-cfg['beta']),a['compensation'][indices],atol=1e-14,rtol=1e-13),'coverage compensation differs')
            require(np.isclose(mass.sum(),n,atol=1e-8,rtol=0),'selected Voronoi coverage mass does not sumN')
        self.note(name+'.density_normalization',eligible_count=int(eligible.sum()),proposal_sum=float(a['proposal_probability'].sum()),meaning='initial conditional proposal on eligible residual candidates; not final inclusion probability or selected coverage density')
        if spec.get('online_generation_cost') is None:self.note(name+'.online_generation_cost','PARTIAL',CPU_observer_wall_seconds=spec['score_capture_wall_seconds'],reason='Only instrumented CPU capture; no native online/GPU request timing claimed')
        if 'paper_formula_status' not in self.m and 'paper_consistency_status' not in self.m:self.note(name+'.paper_full_density_display','PARTIAL',reason='Source weight and eligible normalization verified; no explicit full-reference mean-one lambda artifact/mapping status supplied')
    def truth(self):
        if self.car:
            tree=ast.parse((self.base/'capture.py').read_text());forbidden={'target_phys','target_norm','truth','y_true','ground_truth'}
            accesses=[node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute) and node.attr in forbidden]
            require(not accesses,'capture reads target/truth attributes '+str(accesses))
            require(self.m.get('truth_loaded_by_dataset_loader') is True and self.m.get('truth_read_by_capture_or_score') is False,'Car truth possession/use not correctly distinguished')
            return dict(truth_loaded=True,truth_used_for_score=False,scope='source main accesses native geometry, normalized model input and predictions/head output; loader truth presence disclosed; no live denial guard in Car')
        receipt=read(self.file('truth_isolation.json'));capture=self.base/'capture_scores.py';text=capture.read_text()
        require(receipt['actual_capture_truth_access_attempts']==0 and receipt['score_graph_truth_access'] is False,'truth access during score generation')
        require({r['kind'] for r in receipt['negative_control_rejections']}=={'target_method','q_attribute','truth_file'},'missing truth guard negative checks')
        require(not any(r['file']=='NACA_Cylinder_Q.npy' for r in receipt['score_capture_file_accesses']),'truth file opened during score capture')
        require(text.index('write(HERE / "score_manifest.json", manifest)')<text.index('phase["name"] = "error_only"')<text.index('truth = np.asarray'),'diagnostic truth phase precedes saved score bundle')
        for method in ['uniform','pi','cpg','ca']:
            data=arrays(self.base/f'{method}_score_arrays.npz');require(not any('truth' in key or 'error' in key for key in data),'truth/error array inside score bundle')
            error_path=self.file(f'{method}_error_only.npz');d=arrays(error_path);require(np.allclose(d['absolute_error'],np.abs(d['prediction']-d['truth']),atol=0,rtol=0),'diagnostic error identity differs')
            csv_path=self.base/f'{method}_error_points.csv'
            if csv_path.exists():
                self.file(csv_path)
                with csv_path.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
                require(not any('truth' in k or 'prediction' in k for k in fields),'public diagnostic CSV contains more than error/geometry')
                col=next(k for k in fields if 'error' in k)
                values=np.array([float(r[col]) for r in rows]);require(equal(values,d['absolute_error'].reshape(-1)),'diagnostic CSV error values differ')
        return dict(truth_used_for_score=False,denial_tests=3,actual_score_access_attempts=0,error_phase_after_score_persistence=True,diagnostic_arrays_separate=True)
    def car_cpu_diagnostic(self):
        directory=self.base/'ca_cpu_diagnostic';path=directory/'run_manifest.json'
        if not path.exists():return dict(status='NOT_APPLICABLE',reason='No separate CPU diagnostic supplied')
        m=read(self.file(path));d=arrays(self.file(directory/m['npz'],m['files'][m['npz']]['sha256']));self.file(directory/m['perpoint_csv'],m['files'][m['perpoint_csv']]['sha256'])
        source=self.file(self.base/'cpu_ca_run.py',m['source_script_sha256']);text=source.read_text()
        require(text.index('prediction_norm=ca.final(')<text.index('truth=item.target_phys.numpy()')<text.index('with np.load(source/"arrays"'),'CPU diagnostic consumes truth or archived GPU arrays before own final prediction')
        require(m['checkpoint_sha256']==self.m['checkpoint_sha256'] and m['head_sha256']==self.m['head_sha256'] and m['normalizer_sha256']==self.m['normalizer_sha256'],'independent CPU diagnostic checkpoint/head mismatch')
        require(equal(d['xyz'],self.native_sample.pos.numpy()),'CPU diagnostic native coordinates')
        n=len(d['xyz']);ids=d['actual_selected_indices'];ci=d['coarse_indices'];cfg=m['configuration']
        from adasolver.inference.pi.shapenet_car import car_surface_distance
        distance=car_surface_distance(d['xyz'],d['surface']);physics=np.clip(d['normalized_score']/max(float(np.quantile(d['normalized_score'][d['surface']],.95)),1e-12),0,1)
        target_weight=(1+cfg['wall_amplitude']*np.exp(-distance/cfg['wall_scale']))*(1+cfg['physics_strength']*physics);target=target_weight/target_weight.sum()
        with (directory/m['perpoint_csv']).open() as f:reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
        require(len(rows)==n and np.array_equal([int(r['index']) for r in rows],np.arange(n)),'CPU diagnostic CSV point coverage')
        coords=np.array([[float(r[k]) for k in ['x','y','z']] for r in rows]);require(equal(coords.astype(d['xyz'].dtype),d['xyz']),'CPU diagnostic CSV coordinates')
        for key in columns:
            if key in ['index','x','y','z']:continue
            values=np.array([float(r[key]) if r[key] else np.nan for r in rows])
            if key in ['selected_order','coarse_order']:
                ordered=ids if key=='selected_order' else ci;require(equal(values[ordered],np.arange(len(ordered))),'CPU CSV rank')
                absent=np.ones(n,bool);absent[ordered]=False;require(np.all(np.isnan(values[absent])|(values[absent]==-1)),'CPU CSV absent rank')
            elif key in ['coarse_head_output_raw','coarse_pressure_proxy']:
                expected=d['ca_head_raw' if key=='coarse_head_output_raw' else key]
                require(equal(values[ci].astype(expected.dtype),expected),'CPU CSV coarse pressure/head mapping')
                absent=np.ones(n,bool);absent[ci]=False;require(np.isnan(values[absent]).all(),'CPU CSV coarse field outside support')
            elif key in ['target_density','target_reference_ratio']:
                require(np.allclose(values,target if key=='target_density' else n*target,atol=1e-13,rtol=1e-13),'CPU diagnostic designed target does not use its own score')
            elif key in ['coarse_feature_l2','coarse_feature_mean','coarse_feature_std']:
                matrix=d['ca_block0_features'].astype(np.float64)
                expected=np.linalg.norm(matrix,axis=1) if key.endswith('l2') else matrix.mean(axis=1) if key.endswith('mean') else matrix.std(axis=1,ddof=0)
                require(equal(values[ci],expected),'CPU diagnostic feature summary '+key)
                absent=np.ones(n,bool);absent[ci]=False;require(np.isnan(values[absent]).all(),'CPU feature summary outside coarse support')
            elif key=='predicted_surface_pressure':
                require(equal(values[d['surface']],d['complete_prediction_physical'][d['surface'],3]) and np.isnan(values[~d['surface']]).all(),'CPU diagnostic surface pressure scalar mapping')
            else:require(key in d and equal(values,d[key]),'CPU diagnostic CSV/NPZ field '+key)
        for key,hkey in [('actual_selected_indices','selected_ordered_sha256'),('coarse_indices','coarse_ordered_sha256'),('normalized_score','score_sha256'),('raw_score','raw_score_sha256'),('complete_prediction_physical','complete_prediction_sha256')]:require(array_sha(d[key])==m[hkey],'CPU diagnostic array hash '+key)
        expected=np.maximum(np.maximum(d['raw_score'],0)/max(float(np.quantile(np.maximum(d['raw_score'],0),.95)),1e-12),cfg['score_floor'])**cfg['score_power'];require(equal(expected,d['normalized_score']),'CPU diagnostic raw->normalized score')
        from adasolver.inference.cpg.shapenet_car import surface_scalar_score
        rebuilt=surface_scalar_score(d['xyz'],d['surface'],ci,d['coarse_pressure_proxy'],feature=cfg['feature'],k=cfg['neighborhood'],surface_band=cfg['surface_band']);require(equal(rebuilt,d['raw_score']),'CPU diagnostic coarse proxy->raw score')
        head=read(WORKSPACE/'rebuttal_20261112/experiments/E5_training_seeds/checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json')
        require(d['ca_block0_features'].shape==(len(ci),256),'CPU diagnostic own feature alignment')
        require(np.allclose(d['ca_head_raw']*head['target_std']+head['target_mean'],d['coarse_pressure_proxy'],atol=1e-7,rtol=1e-6),'CPU diagnostic head decoding')
        strata=d['stratum'];mass=np.bincount(strata);counts=np.bincount(strata[ids],minlength=len(mass));rate=counts[strata]/mass[strata]
        require(equal(rate,d['allocation_density']) and np.allclose(rate/len(ids),d['normalized_allocation_density'],atol=1e-18,rtol=1e-13),'CPU diagnostic quota density')
        # Reconstruct only from this diagnostic's stored final prediction. No model forward.
        from adasolver.metrics.shapenet_car import reconstruct_car_fields
        e5=WORKSPACE/'rebuttal_20261112/experiments/E5_training_seeds';model=read(e5/'checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json')
        normalizer=np.load(model['normalizer']);mean=normalizer['mean_out'];std=normalizer['std_out']
        velocity,pressure,surface_ids=reconstruct_car_fields(ids,d['final_prediction_normalized'],d['xyz'],d['surface'],prediction_is_normalized=True,output_mean=mean,output_std=std,k=4)
        field=np.zeros((n,4));field[:,:3]=velocity;field[surface_ids,3]=pressure;require(equal(field,d['complete_prediction_physical']),'CPU field differs from its own retained prediction reconstruction')
        truth=self.native_sample.target_phys.numpy();surface=d['surface'];vel_error=np.linalg.norm(field[:,:3]-truth[:,:3],axis=1);pressure_error=np.abs(field[surface,3]-truth[surface,3])
        require(equal(vel_error,d['velocity_absolute_error']) and equal(pressure_error,d['surface_pressure_absolute_error'][surface]) and np.isnan(d['surface_pressure_absolute_error'][~surface]).all(),'CPU diagnostic own error fields')
        v=float(np.linalg.norm(field[:,:3]-truth[:,:3])/np.linalg.norm(truth[:,:3]));p=float(np.linalg.norm(field[surface,3]-truth[surface,3])/np.linalg.norm(truth[surface,3]));score=v+.5*p
        require(np.allclose([v,p,score],[m['metrics']['velocity_rel_l2'],m['metrics']['pressure_rel_l2'],m['metrics']['field_error']],atol=1e-12,rtol=1e-12),'CPU diagnostic metrics')
        archived=arrays(resolve(self.m['methods']['ca']['source_npz'],self.base));delta=np.abs(d['normalized_score']-archived['score']);violations=int((delta>1e-6+1e-5*np.abs(archived['score'])).sum())
        require(violations==m['comparison_to_archived_GPU']['normalized_score_violations'],'CPU/GPU diagnostic comparison count')
        return dict(own_CPU_chain_artifacts_consistent=True,own_field_error=score,truth_used_only_in_diagnostic=True,archived_GPU_used_only_after_own_prediction=True,archived_GPU_score_violations=violations,ordered_selected_indices_equal=equal(ids,archived['indices']),diagnostic_wall_seconds=m['diagnostic_wall_seconds'],formal_timing=False)
    def run(self):
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run with CUDA_VISIBLE_DEVICES empty; validator never requests GPU')
        self.file(self.path,self.manifest_sha256)
        self.checked('source_identity',self.source_checks)
        xyz,surface=self.geometry()
        methods=self.m['methods'];methods=methods.items() if isinstance(methods,dict) else [(m['method'],m) for m in methods]
        for name,spec in methods:self.checked(name+'.complete_checks',lambda name=name,spec=spec:self.method(name,spec,xyz,surface))
        self.checked('truth_boundary_source_review',self.truth)
        if self.car:self.checked('independent_CPU_CA_diagnostic',self.car_cpu_diagnostic)
        if self.m.get('paper_formula_status')=='PARTIAL':self.note('declared_paper_formula_status','PARTIAL',reason=self.m.get('paper_formula_reason'))
        if str(self.m.get('paper_consistency_status','')).startswith('PARTIAL'):self.note('declared_paper_consistency_status','PARTIAL',reason=self.m.get('paper_consistency_status'))
        if sha(self.path)!=self.manifest_sha256:self.note('manifest_stability','PARTIAL',reason='Producer changed manifest during QA; receipt binds the initial snapshot and should be rerun after producer finishes')
        fail=sum(x['status']=='FAIL' for x in self.checks);partial=sum(x['status']=='PARTIAL' for x in self.checks)
        return dict(status='STALE_OR_CONFLICT' if fail else 'PARTIAL' if partial else 'VERIFIED_COMPLETE',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=self.m['dataset'],case_id=self.m['case_id'],checkpoint_sha256=self.m['checkpoint_sha256'],manifest_sha256=self.manifest_sha256,validator_sha256=sha(__file__),contract_sha256=sha(QA_CONTRACT),gpu_used=False,model_forward_executed=False,source_data_modified=False,failed_checks=fail,partial_checks=partial,checks=self.checks,source_inputs=self.inputs)


def main():
    global WORKSPACE,AIRFOIL_DATA,QA_CONTRACT
    p=argparse.ArgumentParser(description='CPU source QA using explicitly supplied external workspace and geometry data')
    p.add_argument('--workspace',type=Path,default=os.environ.get('ADASOLVER_WORKSPACE'),help='External original experiment workspace; or ADASOLVER_WORKSPACE')
    p.add_argument('--airfoil-data',type=Path,default=os.environ.get('ADASOLVER_AIRFOIL_DATA'),help='Folder containing NACA_Cylinder_X/Y.npy; or ADASOLVER_AIRFOIL_DATA')
    p.add_argument('--contract',type=Path,default=QA_CONTRACT,help='The accompanying qa_contract.json')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.workspace is None:p.error('--workspace or ADASOLVER_WORKSPACE is required')
    WORKSPACE=args.workspace.expanduser().resolve()
    AIRFOIL_DATA=args.airfoil_data.expanduser().resolve() if args.airfoil_data else None
    QA_CONTRACT=args.contract.expanduser().resolve()
    if not WORKSPACE.is_dir():p.error('The supplied workspace directory does not exist')
    if not QA_CONTRACT.is_file():p.error('The QA contract does not exist; pass --contract')
    result=Audit(args.manifest.resolve()).run();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','dataset','failed_checks','partial_checks']}))
    raise SystemExit(1 if result['failed_checks'] else 0)


if __name__=='__main__':main()
