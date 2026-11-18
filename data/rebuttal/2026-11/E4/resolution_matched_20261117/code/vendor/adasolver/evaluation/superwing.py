"""One full-reference evaluator for SuperWing Uniform and adaptive policies."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from adasolver.data.superwing import SuperWingAdapter
from adasolver.data.superwing_runtime import SuperWingRuntime
from adasolver.inference.superwing_contract import (BUDGETS, SEEDS, NFULL, NCOARSE, PROTOCOL,
    identity, indices_identity, coarse_indices, rng_for, validate_indices, validate_policy, write_json)
from adasolver.metrics.superwing import evaluate_field
from adasolver.utils.budget import budget_count
from adasolver.models.checkpoint import sha256_file


def record_key(record):
    return record['config_id'], record['sample_id'], record['seed'], record['budget_fraction']


def load_records(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def uniform_config():
    return dict(method='uniform', target='shared')


def open_run(output, runtime, ids, seeds, policies, split, stage):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from adasolver.inference.superwing_contract import ROOT
    sources=['adasolver/data/superwing.py','adasolver/data/superwing_runtime.py',
             'adasolver/metrics/superwing.py','adasolver/models/transolver3.py',
             'adasolver/layers/compensation.py','adasolver/layers/first_block.py',
             'adasolver/utils/interpolation.py','adasolver/evaluation/superwing.py',
             'adasolver/inference/superwing_sampling.py','adasolver/inference/superwing_contract.py',
             'adasolver/inference/pi/superwing.py','adasolver/inference/cpg/superwing.py',
             'adasolver/inference/ca/superwing.py','adasolver/models/adapter.py','adasolver/utils/budget.py']
    if any(p.get('cd_revision')=='v2' or p.get('sampler') in ('weighted_remaining','stratified_weighted') for p in policies):
        sources.append('adasolver/inference/superwing_cd.py')
    if any(p.get('cd_signal')=='density_influence' for p in policies):
        sources.append('adasolver/inference/superwing_influence.py')
    if runtime.config['protocol']['id'] != PROTOCOL:
        sources.extend(['adasolver/metrics/superwing_cdcl.py','adasolver/inference/superwing_cdcl.py'])
    if any(p.get('density_revision')=='v2' or p.get('budget_parameters') for p in policies):
        sources.extend(['adasolver/inference/superwing_density.py','adasolver/evaluation/superwing_replay.py'])
    manifest = dict(protocol=runtime.config['protocol']['id'], signature=runtime.signature, signature_fields=runtime.signature_fields,
                    source_sha256={p:sha256_file(ROOT/p) for p in sources},
                    sample_ids=list(map(int, ids)), seeds=list(seeds), policies=policies, split=split, stage=stage,
                    environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                     device=str(runtime.device), gpu=torch.cuda.get_device_name(runtime.device)
                                     if runtime.device.type=='cuda' else None, cpu_threads=torch.get_num_threads()))
    path = output/'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ('protocol','signature','sample_ids','seeds','policies','split','stage'):
            if previous[key] != manifest[key]:
                raise ValueError(f'resume manifest mismatch: {key}; use a new output directory')
    else:
        write_json(path, manifest)
    records = load_records(output/'raw.jsonl')
    done = {record_key(r) for r in records}
    if len(done) != len(records) or any(r['status'] != 'ok' for r in records):
        raise ValueError('existing raw records contain duplicates or failed rows')
    return output, done


def run_fullmesh(config, *, split='calibration', sample_ids=None, output, device='cuda:0'):
    adapter = SuperWingAdapter(config['data_path'], config['normalizer_path'], config['split_path'])
    runtime = SuperWingRuntime(config, device)
    ids = list(adapter.load_split(split).sample_ids if sample_ids is None else sample_ids)
    policy = dict(method='fullmesh', target='shared')
    output, done = open_run(output, runtime, ids, (0,), [policy], split, 'fullmesh_reference')
    cid = identity(policy)
    rows = load_records(output/'raw.jsonl')
    warmed = False
    with (output/'raw.jsonl').open('a', buffering=1) as handle:
        for row, sample_id in enumerate(ids):
            if (cid,sample_id,0,1.0) in done:
                continue
            start = time.perf_counter()
            sample = adapter.load(sample_id)
            geometry_seconds = time.perf_counter()-start
            if not warmed:
                runtime.predict(sample.geometry.features())
                warmed = True
            prediction, seconds = runtime.predict(sample.geometry.features())
            metrics = evaluate_field(prediction, sample.target, sample.geometry.weights, sample.structured)
            if config['protocol']['id'] != PROTOCOL:
                from adasolver.metrics.superwing_cdcl import from_record
                metrics.update(from_record(metrics))
            record = dict(protocol=config['protocol']['id'], protocol_signature=runtime.signature, stage='fullmesh_reference',
                          split=split, method='fullmesh', metric_target='shared', config_id=cid,
                          sample_id=sample_id, shape_id=sample.geometry.shape_id, seed=0,
                          budget_fraction=1.0, selected_count=NFULL, budget_count=NFULL, full_count=NFULL,
                          geometry_seconds=geometry_seconds, final_seconds=seconds, status='ok', **metrics)
            handle.write(json.dumps(record, allow_nan=False)+'\n')
            rows.append(record)
            if (row+1)%50==0 or row+1==len(ids):
                print(f'fullmesh {split}: {row+1}/{len(ids)}', flush=True)
    keys = [key for key,value in rows[0].items() if isinstance(value,float) and key not in ('budget_fraction',)]
    summary = dict(split=split, sample_count=len(rows), mean={k:float(np.mean([r[k] for r in rows])) for k in keys},
                   complete=len(rows)==len(ids), protocol_signature=runtime.signature)
    if split=='calibration':
        expected=runtime.metadata['best_calibration_relative_l2']
        summary.update(legacy_calibration_relative_l2=expected,
                       legacy_mean_absolute_difference=abs(summary['mean']['field_relative_l2']-expected))
    write_json(output/'summary.json', summary)
    return summary


def run_evaluation(config, policies, *, split, output, device='cuda:0', sample_ids=None,
                   seeds=SEEDS, stage='development', save_fields=(),replay_records=None):
    for policy in policies:
        validate_policy(policy)
        if policy['target']=='cdcl' and config['protocol']['id']==PROTOCOL:
            raise ValueError('Cd/Cl policies require the independent Cd/Cl protocol')
    adapter = SuperWingAdapter(config['data_path'], config['normalizer_path'], config['split_path'])
    runtime = SuperWingRuntime(config, device)
    ids = list(adapter.load_split(split).sample_ids if sample_ids is None else sample_ids)
    if not ids or len(set(ids))!=len(ids) or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('sample IDs and seeds must be nonempty and unique')
    if not set(ids).issubset(adapter.splits[split]):
        raise ValueError('requested IDs are outside the named split')
    if config['protocol']['id'] != PROTOCOL:
        threshold=config['protocol']['cdcl_cl_threshold']
        if np.any(np.abs(adapter.arrays['index'][ids,9]) < threshold):
            raise ValueError('Cd/Cl run contains samples outside its frozen population')
    if split=='test' and stage!='frozen_test':
        raise ValueError('test evaluation requires stage=frozen_test and frozen policies')
    cids = [identity(p) for p in policies]
    if len(set(cids)) != len(cids):
        raise ValueError('duplicate policy configurations')
    if split=='test':
        for p in policies:
            if p['method']!='uniform' and not p.get('frozen',False):
                raise ValueError('only frozen policies may be evaluated on test')
    output, done = open_run(output,runtime,ids,seeds,policies,split,stage)
    if replay_records is not None:
        from adasolver.evaluation.superwing_replay import seed_replay
        reused=seed_replay(output,runtime,ids,seeds,policies,split,stage,replay_records,done)
        print(f'reused {reused} verified same-execution records',flush=True)
    (output/'indices').mkdir(exist_ok=True)
    heads = {}
    warmed = False
    with (output/'raw.jsonl').open('a', buffering=1) as handle:
        for row, sample_id in enumerate(ids):
            expected = {(cid,sample_id,s,r) for cid in cids for s in seeds for r in BUDGETS}
            if expected.issubset(done):
                continue
            start = time.perf_counter()
            sample = adapter.load(sample_id)
            geometry = sample.geometry
            geometry_seconds = time.perf_counter()-start
            if not warmed:
                runtime.predict(geometry.features(np.arange(12288)))
                warmed = True
            pi_scores = {}
            for seed in seeds:
                s0 = coarse_indices(sample_id,seed)
                acquisition = {}
                for policy,cid in zip(policies,cids):
                    if all((cid,sample_id,seed,r) in done for r in BUDGETS):
                        continue
                    method,target = policy['method'],policy['target']
                    score = cache = None
                    timing = dict(coarse_or_block_seconds=0.0,head_seconds=0.0,score_seconds=0.0)
                    if method=='pi':
                        from adasolver.inference.pi.superwing import physics_score
                        key=identity(policy) if target=='cdcl' else policy['prior']
                        if key not in pi_scores:
                            started=time.perf_counter()
                            if target=='cdcl':
                                from adasolver.inference.superwing_cdcl import physics_score as ratio_physics
                                value=ratio_physics(geometry,policy)
                            else:
                                value=physics_score(geometry,key)
                            pi_scores[key]=(value,time.perf_counter()-started)
                        score,timing['score_seconds']=pi_scores[key]
                    elif method in ('cpg','ca'):
                        from adasolver.inference.cpg.superwing import acquire_score
                        score,cache,timing=acquire_score(runtime,geometry,s0,policy,acquisition,heads)
                    elif method!='uniform':
                        raise ValueError(f'unknown SuperWing method {method}')
                    for budget_index,fraction in enumerate(BUDGETS):
                        from adasolver.inference.superwing_density import for_budget
                        active_policy=for_budget(policy,budget_index)
                        key=(cid,sample_id,seed,fraction)
                        if key in done:
                            continue
                        if runtime.device.type=='cuda':
                            torch.cuda.reset_peak_memory_stats(runtime.device)
                        start=time.perf_counter()
                        count=budget_count(NFULL,fraction)
                        sampling_start=time.perf_counter()
                        if method=='uniform':
                            indices=rng_for(sample_id,seed,20,budget_index).choice(NFULL,count,replace=False)
                            comp=None
                            sampling=dict(sampling_seconds=time.perf_counter()-sampling_start,
                                          compensation_seconds=0.0,score_support='none')
                        else:
                            from adasolver.inference.superwing_sampling import select
                            indices,comp,sampling=select(score,count,policy,sample_id,seed,budget_index,
                                                         s0 if method!='pi' or policy.get('retain_coarse') else None,
                                                         geometry=geometry)
                        validate_indices(indices,count)
                        if method in ('cpg','ca') and not np.isin(s0,indices).all():
                            raise ValueError('refinement must preserve all coarse points')
                        features=geometry.features(indices)
                        prediction,final_seconds=runtime.predict(features,comp,cache,
                                                                np.arange(NCOARSE) if cache is not None else None)
                        started=time.perf_counter()
                        reconstructed,fallback=adapter.reconstruct(indices,prediction)
                        reconstruction_seconds=time.perf_counter()-started
                        started=time.perf_counter()
                        metrics=evaluate_field(reconstructed,sample.target,geometry.weights,sample.structured)
                        if config['protocol']['id'] != PROTOCOL:
                            from adasolver.metrics.superwing_cdcl import from_record
                            metrics.update(from_record(metrics))
                        metric_seconds=time.perf_counter()-started
                        acquisition_seconds=sum(timing.values())
                        total_seconds=time.perf_counter()-start+geometry_seconds+acquisition_seconds
                        identifier=f'{cid[:12]}_{sample_id}_{seed}_{budget_index}'
                        sidecar=output/'indices'/f'{identifier}.npz'
                        np.savez_compressed(sidecar,selected=indices.astype(np.int32),
                                            coarse=s0.astype(np.int32) if method in ('cpg','ca') or policy.get('retain_coarse') else np.empty(0,np.int32))
                        record=dict(protocol=config['protocol']['id'],protocol_signature=runtime.signature,stage=stage,split=split,
                                    method=method,metric_target=target,config_id=cid,sample_id=sample_id,
                                    shape_id=geometry.shape_id,seed=seed,budget_fraction=fraction,
                                    full_count=NFULL,budget_count=count,selected_count=len(indices),
                                    coarse_fraction=0.1 if method in ('cpg','ca') else 0.0,
                                    coarse_count=NCOARSE if method in ('cpg','ca') else 0,
                                    uniform_base_count=NCOARSE if method in ('cpg','ca') or policy.get('retain_coarse') else 0,
                                    selected_indices_identity=indices_identity(indices),
                                    coarse_indices_identity=indices_identity(s0) if method in ('cpg','ca') else None,
                                    indices_sidecar=str(sidecar.relative_to(output)),
                                    operator_sha256=runtime.metadata['checkpoint_sha256'],
                                    head_sha256=policy.get('head_sha256'),
                                    compensation_backend='identity' if method=='uniform' or sampling.get('score_uniform_fallback') else 'native_lattice_nearest_cell_mass',
                                    compensation_beta=active_policy.get('beta',0),
                                    interpolation_backend='scipy_linear_nearest',reconstruction_coordinate_system='native_lattice',
                                    interpolation_fallback_count=fallback,geometry_seconds=geometry_seconds,
                                    acquisition_cache_hit=bool(acquisition.get('_last_cache_hit',False)) if method in ('cpg','ca') else False,
                                    acquisition_interpolation_fallback_count=acquisition.get('_last_fallback_count',0) if method in ('cpg','ca') else 0,
                                    acquisition_seconds=acquisition_seconds,final_seconds=final_seconds,
                                    reconstruction_seconds=reconstruction_seconds,metric_seconds=metric_seconds,
                                    total_seconds=total_seconds,
                                    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type=='cuda' else 0,
                                    status='ok',**timing,**sampling,**metrics)
                        handle.write(json.dumps(record,allow_nan=False)+'\n')
                        done.add(key)
                        if sample_id in save_fields:
                            directory=output/'fields'
                            directory.mkdir(exist_ok=True)
                            np.savez_compressed(directory/f'{identifier}.npz',prediction=reconstructed,
                                                score=score if score is not None else np.ones(NFULL),
                                                compensation=np.ones(len(indices)) if comp is None else comp)
            if (row+1)%8==0 or row+1==len(ids):
                print(f'{split}: {row+1}/{len(ids)} samples, {len(done)} records, {len(policies)} policies',flush=True)
    write_json(output/'completion.json',dict(complete=len(done)==len(ids)*len(seeds)*len(BUDGETS)*len(policies),
                                           record_count=len(done),expected_count=len(ids)*len(seeds)*len(BUDGETS)*len(policies)))
    return output
