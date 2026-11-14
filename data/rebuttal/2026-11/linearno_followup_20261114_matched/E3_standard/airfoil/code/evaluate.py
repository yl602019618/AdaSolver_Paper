"""Paired old/new reconstruction from fixed predictions; no test-based selection."""
from runtime import *
import argparse,time,math,csv,shutil
from adasolver.evaluation.airfoil2d import force_record,pressure_surface_force

EVALUATION_SOURCE_BYTES={name:(OUT/'code'/name).read_bytes() for name in ['evaluate.py','runtime.py','common.py']}
EVALUATION_SOURCE_HASHES={name:hashlib.sha256(value).hexdigest() for name,value in EVALUATION_SOURCE_BYTES.items()}


def attach_reconstruction():
    gate=read(OUT/'config/evaluation_gate.json')
    if not gate.get('accepted'):raise RuntimeError('No accepted reconstruction gate')
    path=Path(gate['module']);assert sha(path)==gate['module_sha256']
    sys.path.insert(0,str(path.parent))
    import reconstruct_api
    if reconstruct_api.IMPLEMENTATION_VERSION!='1.1':raise RuntimeError('Unexpected loaded reconstruction version')
    if reconstruct_api.IMPLEMENTATION_SOURCE_SHA256!=gate['module_sha256']:raise RuntimeError('Loaded reconstruction source differs from gate')
    snapshot=OUT/'config/reconstruction_sources'/gate['module_sha256'];snapshot.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ['reconstruct_api.py','native_grid_reference.py','native_car_reference.py','timing_helper.py']:
        original=path.parent/name;target=snapshot/name
        if target.exists() and sha(target)!=sha(original):raise RuntimeError('Frozen reconstruction dependency changed')
        if not target.exists():shutil.copyfile(original,target)
        sources[name]=dict(source=str(original),snapshot=str(target),sha256=sha(target))
    write(snapshot/'manifest.json',dict(implementation_version=reconstruct_api.IMPLEMENTATION_VERSION,
          actual_loaded_module=str(Path(reconstruct_api.__file__).resolve()),sources=sources))
    return reconstruct_api,gate


def metrics(field,coords,truth):
    value=np.asarray(field,dtype=np.float64)
    rel=float(np.linalg.norm(value-truth)/np.linalg.norm(truth))
    true=force_record(pressure_surface_force(coords,truth));pred=force_record(pressure_surface_force(coords,value))
    ratio=None if pred['Cd_over_Cl'] is None or true['Cd_over_Cl'] is None else abs(pred['Cd_over_Cl']-true['Cd_over_Cl'])
    return dict(field_relative_l2=rel,cd_abs_error=abs(pred['Cd']-true['Cd']),cl_abs_error=abs(pred['Cl']-true['Cl']),
                cdcl_abs_error=ratio,prediction=pred,truth=true)


def compare_pair(old,new,coords,truth):
    delta=np.abs(new-old);bound=1e-6+1e-5*np.abs(old)
    old_metrics,new_metrics=metrics(old,coords,truth),metrics(new,coords,truth)
    d={k:None if old_metrics[k] is None or new_metrics[k] is None else abs(old_metrics[k]-new_metrics[k])
       for k in ['field_relative_l2','cd_abs_error','cl_abs_error','cdcl_abs_error']}
    valid=int(np.count_nonzero(delta>bound))==0
    for k,v in d.items():
        if v is None:valid &= old_metrics[k] is None and new_metrics[k] is None
        else:valid &= v<=1e-8+1e-5*abs(old_metrics[k])
    return dict(max_abs=float(delta.max()),relative_norm=float(np.linalg.norm(delta)/max(np.linalg.norm(old),1e-300)),
                abs_quantiles=np.quantile(delta,[.5,.9,.99,1.]).tolist(),elementwise_violations=int(np.count_nonzero(delta>bound)),
                metric_differences=d,passed=bool(valid)),old_metrics,new_metrics


def evaluate_group(runtime,api,gate,ids,configs,stage,validation=False,with_full=True):
    dest=OUT/'records'/runtime.architecture/stage;dest.mkdir(parents=True,exist_ok=True)
    status_path=dest/'manifest.json'
    if status_path.exists() and read(status_path).get('execution_status')=='completed':return read(status_path)
    if (dest/'metrics.jsonl').exists():raise FileExistsError('Partial run must be explicitly resumed/repaired')
    state=dict(architecture=runtime.architecture,execution_status='running',evidence_status='pending',stage=stage,
               checkpoint=runtime.metadata['checkpoint'],checkpoint_sha256=runtime.operator_hash,
               normalizer_sha256=runtime.data.normalizer_sha256,sample_ids=ids,sampling_seeds=[0,1,2],
               budgets=[.375,.5,.75],reconstruction_gate=gate,metric_target='field',force_metrics='field-target diagnostics only',
               expected_requests=len(ids)*(int(with_full)+9*len(configs)),configurations=configs,with_full=with_full,
               evaluation_sources=EVALUATION_SOURCE_HASHES)
    code_snapshot=dest/'sources';code_snapshot.mkdir(exist_ok=True)
    for name,value in EVALUATION_SOURCE_BYTES.items():(code_snapshot/name).write_bytes(value)
    write(status_path,state);count=0;violations=[];pair_checks=[];start=time.perf_counter()
    with (dest/'metrics.jsonl').open('x') as stream:
        for index,sid in enumerate(ids,1):
            runtime.clear_dynamic();coords=runtime.data.coords(sid);truth=runtime.data.target(sid).astype(np.float64)
            if with_full:
                full=runtime.predict_full(sid).cpu().numpy().astype(np.float64)
                full_metrics=metrics(full,coords,truth)
                full_array=f'case{sid}_full.npz';np.savez_compressed(dest/full_array,field=full)
                stream.write(json.dumps(dict(case_id=sid,method='full',sampling_seed=None,budget_fraction=1.,retained_points=11271,
                        reconstruction_id='full_native_identity_float64',old=full_metrics,new=full_metrics,
                        checkpoint_sha256=runtime.operator_hash,normalizer_sha256=runtime.data.normalizer_sha256,arrays=full_array),allow_nan=False)+'\n');count+=1
            pairs={}
            for seed in [0,1,2]:
                for budget in [.375,.5,.75]:
                    for cfg in configs:
                        pred,sampling,coarse,metadata=runtime.predict_subset(sid,seed,budget,cfg)
                        indices=sampling.indices;prediction=pred.cpu().numpy()
                        old=api.native_reconstruct('airfoil',indices,prediction,runtime.reference)
                        plan=api.build_plan('airfoil',runtime.reference,indices,channels=1,device=runtime.device)
                        new=plan.apply_tensor(pred).cpu().numpy()
                        agreement,old_metric,new_metric=compare_pair(old,new,coords,truth)
                        name=cfg['name'];key=(seed,budget)
                        ih=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
                        ch=None if coarse is None else hashlib.sha256(coarse.astype(np.int64).tobytes()).hexdigest()
                        if name=='cpg_beta0':pairs[key]=(ih,ch,metadata['coarse_prediction_hash'])
                        elif name=='cpg':
                            equal=pairs[key]==(ih,ch,metadata['coarse_prediction_hash']);assert equal
                            pair_checks.append(dict(case_id=sid,seed=seed,budget=budget,coarse_and_final_identical=True))
                        if not agreement['passed']:violations.append(dict(case_id=sid,method=name,seed=seed,budget=budget,agreement=agreement))
                        array_name=f'case{sid}_s{seed}_b{budget}_{name}.npz'
                        arrays=dict(indices=indices,coarse_indices=np.asarray([] if coarse is None else coarse),
                                    compensation=sampling.compensation,prediction=prediction)
                        if validation:arrays.update(old_field=old,new_field=new,absolute_difference=np.abs(new-old))
                        np.savez_compressed(dest/array_name,**arrays)
                        row=dict(case_id=sid,method=name,method_family=cfg['method'],beta=None if name=='uniform' else cfg['beta'],
                                 sampling_seed=seed,budget_fraction=budget,retained_points=len(indices),coarse_points=0 if coarse is None else len(coarse),
                                 indices_hash=ih,coarse_indices_hash=ch,coarse_prediction_hash=metadata.get('coarse_prediction_hash'),
                                 checkpoint_sha256=runtime.operator_hash,normalizer_sha256=runtime.data.normalizer_sha256,
                                 head_sha256=metadata.get('head_sha256'),
                                 old_reconstruction_id=gate['old_id'],new_reconstruction_id=gate['new_id'],
                                 old=old_metric,new=new_metric,agreement=agreement,arrays=array_name)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');count+=1
            stream.flush();state.update(completed_cases=index,completed_requests=count,elapsed_seconds=time.perf_counter()-start);write(status_path,state)
            print(stage,runtime.architecture,index,'/',len(ids),count,'requests',flush=True)
            if validation and violations:break
    state.update(execution_status='completed' if count==state['expected_requests'] and not violations else 'failed',
                 evidence_status='pending',completed_requests=count,equivalence_violations=violations,
                 paired_beta_groups=len(pair_checks),elapsed_seconds=time.perf_counter()-start)
    write(dest/'beta_pair_checks.json',pair_checks);write(status_path,state)
    if violations:raise RuntimeError(f'{len(violations)} reconstruction equivalence failures; not relaxing tolerance')
    return state


def export_metrics(architecture,gate,sources,phase):
    rows=[];seen=set()
    fields=['dataset','backbone','checkpoint_id','training_seed','head_seed','sampling_seed','method','beta',
            'budget_fraction','case_id','metric_target','test_population','reconstruction_id','metric','value','unit',
            'status','selected_indices_hash','coarse_indices_hash','coarse_prediction_hash','retained_points',
            'normalizer_sha256','head_sha256','full_alias','source_file']
    label={'t3':'Transolver3','linearno':'LinearNO_point_temp','linearno_conv':'LinearNO_conv_temp_fullgrid'}[architecture]
    for source in sources:
        for line in source.open():
            record=json.loads(line)
            key=(record['case_id'],record['method'],record['sampling_seed'],record['budget_fraction'])
            if key in seen:raise ValueError('duplicate actual prediction request')
            seen.add(key)
            for variant,rid in [('old',gate['old_id']),('new',gate['new_id'])]:
                for metric in ['field_relative_l2','cd_abs_error','cl_abs_error','cdcl_abs_error']:
                    value=record[variant][metric]
                    rows.append(dict(dataset='airfoil2d',backbone=label,checkpoint_id=record['checkpoint_sha256'],
                        training_seed=0,head_seed=0 if record['method']=='ca' else None,sampling_seed=record['sampling_seed'],
                        method=record['method'],beta=record.get('beta'),budget_fraction=record['budget_fraction'],
                        case_id=record['case_id'],metric_target='field',test_population='published_cl_gt_0p02_110',
                        reconstruction_id=rid,metric=metric,value=value,
                        unit='ratio' if metric=='field_relative_l2' else 'coefficient_ratio' if metric=='cdcl_abs_error' else 'coefficient',
                        status='ok' if value is not None else 'nonfinite_metric_preserved',
                        selected_indices_hash=record.get('indices_hash',hashlib.sha256(np.arange(11271,dtype=np.int64).tobytes()).hexdigest()),
                        coarse_indices_hash=record.get('coarse_indices_hash'),coarse_prediction_hash=record.get('coarse_prediction_hash'),
                        retained_points=record['retained_points'],normalizer_sha256=record['normalizer_sha256'],
                        head_sha256=record.get('head_sha256'),
                        full_alias=record['method']=='full' and variant=='new',source_file=str(source)))
    target=OUT/f'records/{architecture}/metrics_{phase}.csv'
    temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    if target.exists():
        if target.read_bytes()!=temporary.read_bytes():raise FileExistsError(f'Immutable phase CSV differs: {target}')
        temporary.unlink()
    else:temporary.replace(target)
    compatibility=OUT/f'records/{architecture}/metrics.csv'
    temporary=compatibility.with_name(compatibility.name+f'.{os.getpid()}.tmp');shutil.copyfile(target,temporary);temporary.replace(compatibility)
    return dict(actual_unique_prediction_requests=len(seen),metric_rows=len(rows),metrics_csv=str(target),
                metrics_sha256=sha(target),compatibility_metrics_csv=str(compatibility),export_phase=phase,
                source_files=[str(p) for p in sources],full_computed_once_per_case=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['t3','linearno','linearno_conv'],required=True)
    p.add_argument('--phase',choices=['preliminary','complete'],default='complete');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import update_registry
    update_registry.main()
    assert read(OUT/f'results/{a.architecture}_identity_validation.json')['passed']
    import audit_training_sampling
    audit_training_sampling.ensure_final()
    api,gate=attach_reconstruction();runtime=Runtime(a.architecture)
    split=read(SPLITS);validation_ids=gate['validation_case_ids'];assert set(validation_ids)<=set(split['validation_ids'])
    if a.architecture=='linearno_conv':
        evaluate_group(runtime,api,gate,validation_ids,[],'validation_complete',validation=True)
        result=evaluate_group(runtime,api,gate,split['airfoil']['primary_test_ids'],[],'test_complete')
        sources=[OUT/f'records/{a.architecture}/test_complete/metrics.jsonl']
    else:
        configs=configurations(a.architecture,include_ca=False)
        evaluate_group(runtime,api,gate,validation_ids,configs,'validation_preliminary',validation=True)
        result=evaluate_group(runtime,api,gate,split['airfoil']['primary_test_ids'],configs,'test_preliminary')
        sources=[OUT/f'records/{a.architecture}/test_preliminary/metrics.jsonl']
        if a.phase=='complete':
            ca=[c for c in configurations(a.architecture) if c['name']=='ca']
            evaluate_group(runtime,api,gate,validation_ids,ca,'validation_ca',validation=True,with_full=False)
            result=evaluate_group(runtime,api,gate,split['airfoil']['primary_test_ids'],ca,'test_ca',with_full=False)
            sources.append(OUT/f'records/{a.architecture}/test_ca/metrics.jsonl')
    exported=export_metrics(a.architecture,gate,sources,a.phase)
    result.update(exported,phase=a.phase,reconstruction_id=gate['new_id'],old_reconstruction_id=gate['old_id'])
    write(OUT/f'results/{a.architecture}_{a.phase}_completion.json',result)
    if a.phase=='complete':
        import validate_accuracy
        validate_accuracy.validate(a.architecture)
    import aggregate
    aggregate.main()
    import update_registry
    update_registry.main()


if __name__=='__main__':main()
