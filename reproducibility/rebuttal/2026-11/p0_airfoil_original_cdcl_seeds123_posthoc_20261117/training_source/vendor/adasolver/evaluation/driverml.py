"""One full-reference evaluator for both DriverML targets and all sampling methods."""
from __future__ import annotations
import json
import fcntl
import os
import time
from pathlib import Path
import numpy as np
import torch
from adasolver.data.driverml import DriverMLAdapter
from adasolver.data.driverml_runtime import DriverMLOperator, protocol_signature, index_identity
from adasolver.inference.driverml_contract import ROOT, BUDGETS, TARGETS, coarse_count, rng_for, identity
from adasolver.inference.driverml_sampling import DriverMLSelection, CoarsePrediction, select_from_score
from adasolver.inference.pi.driverml import geometry_prior
from adasolver.inference.cpg.driverml import coarse_acquisition
from adasolver.inference.ca.driverml import load_head, acquire_pressure
from adasolver.inference.uniform import sample_uniform
from adasolver.metrics.driverml import pressure_relative_l2, integrate_cd, relative_cd_error, decode_pressure, decode_wss
from adasolver.utils.budget import budget_count
from adasolver.utils.surface_reconstruction import nearest_surface_map
from adasolver.models.driverml import file_sha256


def defaults(method, target):
    if method not in ('uniform', 'pi', 'cpg', 'ca') or target not in TARGETS:
        raise ValueError('unknown DriverML method or target')
    return dict(method=method, target=target, amp=8.0, beta=0.5, neighborhood=16,
                prior='normal_variation' if target == 'pressure_field' else 'drag_sensitivity',
                feature='lap_proxy', sensitivity=target == 'cd', wss_weight=0.0)


def record_key(record):
    return (record['metric_target'], record['method'], record['run_id'], record['seed'], record['budget_fraction'])


def score_cache(config, signature, method_config, run_id, seed):
    fields = {key: method_config[key] for key in ('method','target','prior','feature','sensitivity','neighborhood','wss_weight')}
    if method_config['method']=='ca':
        fields['head_sha256']=method_config['head_sha256']
    # Geometry is seed independent; CPG keeps exactly the same S0 for all candidates.
    key = identity(dict(protocol=signature, strategy=fields, run_id=run_id,
                        seed=seed if method_config['method'] != 'pi' else None))
    directory = ROOT / 'temp/driverml/acquisition_cache' / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory / 'score.npy', directory / 'metadata.json'


def cached_score(score_path, meta_path, build):
    """One producer per acquisition so concurrent candidates share identical scores."""
    with meta_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if score_path.exists() and meta_path.exists():
            return np.load(score_path), json.loads(meta_path.read_text()), True
        score, metadata = build()
        temporary = score_path.with_name(f'score_{os.getpid()}.npy')
        np.save(temporary, score)
        temporary.replace(score_path)
        temp_meta = meta_path.with_name(f'metadata_{os.getpid()}.json')
        temp_meta.write_text(json.dumps(metadata))
        temp_meta.replace(meta_path)
        return score, metadata, False


def run_evaluation(config, method_config, *, split, output, device='cuda:0', run_ids=None, seeds=(0,1,2),
                   budgets=BUDGETS, workers=8, stage='calibration', save_indices=False,
                   shard_index=0, shards=1,capture_dir=None):
    target, method = method_config['target'], method_config['method']
    method_config=dict(method_config)
    defaults(method, target)
    if tuple(budgets) != BUDGETS:
        raise ValueError('DriverML curves must contain the three fixed final budgets')
    if method == 'ca':
        if not method_config.get('head_checkpoint') or not method_config.get('head_manifest'):
            raise ValueError('CA requires an explicit pressure head checkpoint and manifest')
        for key in ('head_checkpoint','head_manifest'):
            path=Path(method_config[key]).expanduser()
            method_config[key]=str(path if path.is_absolute() else ROOT/path)
        method_config['head_sha256']=file_sha256(method_config['head_checkpoint'])
    if any(seed not in (0,1,2) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('invalid evaluation seeds')
    adapter = DriverMLAdapter(config['data_path'], config['normalizer_path'], config['labels_path'], config['split_path'])
    allowed = adapter.load_split(split).sample_ids
    ids = tuple(allowed if run_ids is None else run_ids)
    if not set(ids).issubset(allowed) or len(set(ids)) != len(ids):
        raise ValueError('requested run IDs must be unique members of the declared split')
    if stage == 'official' and (split != 'test' or ids != adapter.load_split('test').sample_ids or tuple(seeds) != (0,1,2)):
        raise ValueError('official runs require the entire test split and all three seeds')
    if not 1 <= shards <= len(ids) or not 0 <= shard_index < shards:
        raise ValueError('invalid DriverML evaluation shard')
    global_ids = ids
    ids = ids[shard_index::shards]
    signature, provenance = protocol_signature(config)
    config_id = identity(method_config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            rec = json.loads(line)
            if rec['protocol_signature'] != signature or rec['config_id'] != config_id or rec['stage'] != stage:
                raise ValueError('output already contains an incompatible experiment')
            key = record_key(rec)
            if key in completed:
                raise ValueError('duplicate records in output')
            completed.add(key)
    inf = config['inference']
    pressure = DriverMLOperator(config['checkpoints']['pressure'], 1, device, inf['chunk_size'], inf['dtype'])
    wss = DriverMLOperator(config['checkpoints']['wss'], 3, device, inf['chunk_size'], inf['dtype']) if target == 'cd' or method_config['wss_weight'] else None
    head=head_metadata=None
    if method=='ca':
        head,head_metadata=load_head(method_config['head_checkpoint'],method_config['head_manifest'],pressure.metadata['sha256'],device,
                                     normalizer_sha256=file_sha256(config['normalizer_path']),allow_diagnostic=stage=='diagnostic')
        if method_config.get('head_feature_dim',head_metadata['feature_dim'])!=head_metadata['feature_dim']:
            raise ValueError('declared CA feature dimension differs from the head checkpoint')
    labels = adapter.cd_labels()
    manifest = dict(protocol_signature=signature, provenance=provenance, method_config=method_config,
                    split=split, run_ids=ids, global_sample_ids=global_ids, shard_index=shard_index, shards=shards,
                    seeds=list(seeds), budgets=list(budgets), stage=stage)
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with output.open('a', buffering=1) as handle:
        for run_id in ids:
            requested = {(target, method, run_id, seed, fraction) for seed in seeds for fraction in budgets}
            if requested.issubset(completed):
                continue
            load_start = time.perf_counter()
            sample = adapter.load(run_id, include_wss=target=='cd')
            geometry = sample.geometry()
            io_seconds = time.perf_counter()-load_start
            pi_score = None
            geometry_seconds = 0.0
            if method == 'pi':
                score_path, meta_path = score_cache(config, signature, method_config, run_id, 0)
                def build_geometry():
                    started = time.perf_counter()
                    values = geometry_prior(geometry, method_config['prior'], method_config['neighborhood'], workers)
                    return values, {'geometry_seconds': time.perf_counter()-started}
                pi_score, cached_meta, _ = cached_score(score_path, meta_path, build_geometry)
                geometry_seconds = cached_meta['geometry_seconds']
            for seed in seeds:
                coarse = None
                point_cache = None
                score = pi_score
                acquisition = {'coarse_pressure_seconds': 0.0, 'coarse_wss_seconds': 0.0, 'score_seconds': 0.0}
                acquisition_cache_hit = False
                if method == 'cpg':
                    coarse = rng_for(seed, run_id, 0).choice(sample.point_count, coarse_count(sample.point_count), replace=False)
                    score_path, meta_path = score_cache(config, signature, method_config, run_id, seed)
                    def build_coarse_score():
                        timings = dict(coarse_pressure_seconds=0.0, coarse_wss_seconds=0.0, score_seconds=0.0)
                        features = geometry.features(torch.from_numpy(coarse))
                        pred, timings['coarse_pressure_seconds'] = pressure.predict(features, permutation_seed=1000000+run_id*10+seed)
                        p0 = decode_pressure(pred[:, 0], adapter.stats['pressure_mean'], adapter.stats['pressure_std']).numpy()
                        w0 = None
                        if method_config['wss_weight']:
                            pred, timings['coarse_wss_seconds'] = wss.predict(features, permutation_seed=1000000+run_id*10+seed)
                            w0 = decode_wss(pred, adapter.stats['wss_mean'], adapter.stats['wss_std']).numpy()
                        prediction = CoarsePrediction(coarse, p0, sample.point_count, w0)
                        started = time.perf_counter()
                        values = coarse_acquisition(geometry, prediction, feature=method_config['feature'],
                                                   sensitivity=method_config['sensitivity'], k=method_config['neighborhood'],
                                                   wss_weight=method_config['wss_weight'], workers=workers)
                        timings['score_seconds'] = time.perf_counter()-started
                        return values, timings
                    score, acquisition, acquisition_cache_hit = cached_score(score_path, meta_path, build_coarse_score)
                elif method=='ca':
                    coarse=rng_for(seed,run_id,0).choice(sample.point_count,coarse_count(sample.point_count),replace=False)
                    coarse_features=geometry.features(torch.from_numpy(coarse))
                    p0,point_cache,acquisition=acquire_pressure(pressure,head,head_metadata,coarse_features)
                    w0=None
                    if method_config['wss_weight']:
                        wnorm,acquisition['coarse_wss_seconds']=wss.predict(coarse_features,permutation_seed=1000000+run_id*10+seed)
                        w0=decode_wss(wnorm,adapter.stats['wss_mean'],adapter.stats['wss_std']).numpy()
                    direct_score=head_metadata['target']=='normalized_log1p_cpg_lap_proxy'
                    if direct_score and (method_config['feature']!='lap_proxy' or method_config['neighborhood']!=head_metadata['score_neighborhood'] or method_config['wss_weight']):
                        raise ValueError('direct CA score must use its trained neighborhood and no coarse WSS term')
                    prediction=None if direct_score else CoarsePrediction(coarse,p0,sample.point_count,w0)
                    score_path,meta_path=score_cache(config,signature,method_config,run_id,seed)
                    def build_head_score():
                        started=time.perf_counter()
                        if direct_score:
                            from adasolver.inference.driverml_sampling import lift_score
                            values=lift_score(geometry.pos.numpy(),geometry.pos.numpy()[coarse],p0,workers=workers)
                            if method_config['sensitivity']:values*=np.abs(geometry.normal[:,0].numpy())*geometry.area.numpy()
                        else:
                            values=coarse_acquisition(geometry,prediction,feature=method_config['feature'],sensitivity=method_config['sensitivity'],
                                                      k=method_config['neighborhood'],wss_weight=method_config['wss_weight'],workers=workers)
                        return values,{'score_seconds':time.perf_counter()-started}
                    score,cached_meta,acquisition_cache_hit=cached_score(score_path,meta_path,build_head_score)
                    acquisition['score_seconds']=cached_meta['score_seconds']
                for budget_id, fraction in enumerate(budgets):
                    key = (target, method, run_id, seed, fraction)
                    if key in completed:
                        continue
                    pressure.sync()
                    wall_start = time.perf_counter()
                    if pressure.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(pressure.device)
                    count = budget_count(sample.point_count, fraction)
                    rng = rng_for(seed, run_id, 1, budget_id)
                    started = time.perf_counter()
                    if method == 'uniform':
                        selected = sample_uniform(sample.point_count, count, rng)
                        selection = DriverMLSelection(selected.indices, selected.metadata)
                    else:
                        selection = select_from_score(score, count, rng, amp=method_config['amp'], coarse=coarse,
                                                      metadata={'method': method, 'feature': method_config['feature']},
                                                      score_power=method_config.get('score_power',1.0))
                    sampling_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    surface = nearest_surface_map(geometry.pos.numpy(), selection.indices, workers=workers)
                    selected = selection.finalize(surface, beta=method_config['beta'], uniform=method == 'uniform')
                    density_seconds = time.perf_counter()-started
                    indices = torch.from_numpy(selected.indices)
                    features = geometry.features(indices)
                    permutation_seed = 2000000+run_id*100+seed*3+budget_id
                    if method=='ca':
                        pred,pressure_seconds=pressure.predict_cached(features,point_cache,selected.compensation,permutation_seed)
                    else:
                        pred, pressure_seconds = pressure.predict(features, selected.compensation, permutation_seed)
                    p = decode_pressure(pred[:, 0], adapter.stats['pressure_mean'], adapter.stats['pressure_std'])
                    w = None
                    wss_seconds = 0.0
                    if target == 'cd':
                        pred, wss_seconds = wss.predict(features, selected.compensation, permutation_seed)
                        w = decode_wss(pred, adapter.stats['wss_mean'], adapter.stats['wss_std'])
                    started = time.perf_counter()
                    p_full = surface.reconstruct(p)
                    w_full = surface.reconstruct(w) if w is not None else None
                    reconstruction_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    if target == 'pressure_field':
                        metric = {'pressure_field_relative_l2': pressure_relative_l2(p_full, sample.pressure)}
                    else:
                        physical = config['physics']
                        force = integrate_cd(sample.normal, sample.area, p_full, w_full, sample.a_ref,
                                             physical['rho'], physical['velocity'])
                        metric = {**force, 'cd_gt': labels[run_id], 'cd_relative_error': relative_cd_error(force['cd'], labels[run_id]),
                                  'wss_source': 'predicted'}
                    if not all(np.isfinite(v) for v in metric.values() if isinstance(v, (int,float))):
                        raise ValueError('non-finite DriverML metric')
                    metric_seconds = time.perf_counter()-started
                    wall_seconds = time.perf_counter()-wall_start
                    rec = dict(protocol_id=config['protocol']['id'], protocol_signature=signature, config_id=config_id,
                               stage=stage, split=split, metric_target=target, method=method, run_id=run_id, seed=seed,
                               shard_index=shard_index, shards=shards,
                               budget_fraction=fraction, full_count=sample.point_count, budget_count=count, selected_count=len(indices),
                               coarse_fraction=0.1 if coarse is not None else 0.0, coarse_count=0 if coarse is None else len(coarse),
                               coarse_indices_id=None if coarse is None else index_identity(coarse), final_indices_id=index_identity(selected.indices),
                               **metric, **acquisition, sampling_seconds=sampling_seconds, density_seconds=density_seconds,
                               pressure_inference_seconds=pressure_seconds, wss_inference_seconds=wss_seconds,
                               reconstruction_seconds=reconstruction_seconds, metric_seconds=metric_seconds,
                               acquisition_seconds=sum(acquisition.values()), geometry_seconds=geometry_seconds, io_seconds=io_seconds,
                               acquisition_cache_hit=acquisition_cache_hit,
                               head_sha256=method_config.get('head_sha256'),
                               head_feature_dim=head_metadata['feature_dim'] if head_metadata else None,
                               head_parameter_count=head_metadata['parameter_count'] if head_metadata else None,
                               head_feature_depth=head_metadata.get('feature_depth',1) if head_metadata else None,
                               head_input_normalization=head_metadata.get('input_normalization','none') if head_metadata else None,
                               head_target=head_metadata['target'] if head_metadata else None,
                               score_power=method_config.get('score_power',1.0),
                               pressure_inference_order='permuted',
                               cache_mode='first_block_pointwise_permuted' if method=='ca' else 'none',
                               wall_seconds=wall_seconds, total_seconds=wall_seconds+sum(acquisition.values())+geometry_seconds,
                               peak_gpu_bytes=torch.cuda.max_memory_allocated(pressure.device) if pressure.device.type=='cuda' else 0,
                               density_backend=selected.metadata['density_backend'], beta=selected.metadata['beta'],
                               reconstruction_backend='ckdtree_nearest', reconstruction_fallback_count=0,
                               reconstruction_coordinate_system='physical', inference_mode='global_chunked',
                               chunk_size=inf['chunk_size'], dtype=inf['dtype'], status='ok')
                    if save_indices:
                        sidecar = output.parent/f'{method}_{target}_{run_id}_{seed}_{budget_id}_indices.npy'
                        np.save(sidecar, selected.indices)
                        rec['indices_path'] = str(sidecar)
                    if capture_dir is not None:
                        from adasolver.visualization.driverml_data import export_snapshot
                        export_snapshot(sample,selected,surface,p_full,score,rec,capture_dir)
                    handle.write(json.dumps(rec)+'\n')
                    completed.add(key)
                    print(json.dumps({k:rec[k] for k in ['method','metric_target','run_id','seed','budget_fraction','total_seconds',*metric]}), flush=True)
                    del surface, features, p_full, w_full, pred
                del point_cache
            del sample, geometry
    return output
