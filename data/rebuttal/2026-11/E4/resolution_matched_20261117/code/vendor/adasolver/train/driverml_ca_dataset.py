"""Deployment-matched first-block contexts with pointwise pressure supervision."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from adasolver.data.driverml import DriverMLAdapter
from adasolver.data.driverml_runtime import DriverMLOperator,index_identity
from adasolver.inference.driverml_contract import load_config,coarse_count,rng_for,identity
from adasolver.layers.first_block import first_block_features
from adasolver.inference.ca.driverml import acquisition_features
from adasolver.models.driverml import file_sha256


def build_shard(config_path, output, tasks, *, device='cuda:0', rows=16384, train_seed=17,feature_depth=1):
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    cfg=load_config(config_path)
    adapter=DriverMLAdapter(cfg['data_path'],cfg['normalizer_path'],cfg['labels_path'],cfg['split_path'])
    runtime=DriverMLOperator(cfg['checkpoints']['pressure'],1,device,cfg['inference']['chunk_size'],'float32')
    normalizer_id=file_sha256(cfg['normalizer_path'])
    output=Path(output)
    for split,run_id in tasks:
        if split not in ('train','calibration') or run_id not in adapter.load_split(split).sample_ids:
            raise ValueError('CA datasets use only train and development validation IDs')
        destination=output/split/f'run_{run_id}.npz';meta=destination.with_suffix('.json')
        seed=train_seed if split=='train' else 0
        key=identity(dict(run_id=run_id,split=split,seed=seed,rows=rows,operator=runtime.metadata['sha256'],
                          normalizer=normalizer_id,target='normalized_pressure',version=1))
        if feature_depth!=1:key=identity(dict(base=key,feature_depth=feature_depth))
        if destination.exists() and meta.exists():
            if json.loads(meta.read_text())['key']!=key: raise ValueError('CA dataset shard identity mismatch')
            continue
        started=time.perf_counter()
        sample=adapter.load(run_id,include_wss=False)
        n0=coarse_count(sample.point_count)
        coarse=rng_for(seed,run_id,0).choice(sample.point_count,n0,replace=False)
        supervised=rng_for(seed,run_id,2).choice(n0,min(rows,n0),replace=False)
        inputs=sample.features(torch.from_numpy(coarse))
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            features,_=acquisition_features(runtime,inputs,depth=feature_depth,capture=False)
            x=features[0,torch.from_numpy(supervised).to(device)].float().cpu().numpy()
        target=(sample.pressure[torch.from_numpy(coarse[supervised])].numpy()-adapter.stats['pressure_mean'])/adapter.stats['pressure_std']
        if not np.isfinite(x).all() or not np.isfinite(target).all():
            raise ValueError('non-finite pressure-head supervision')
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez(destination,features=x,target=target.astype(np.float32),point_indices=coarse[supervised])
        info=dict(key=key,run_id=run_id,split=split,seed=seed,full_count=sample.point_count,coarse_count=n0,
                  coarse_fraction=.1,coarse_indices_id=index_identity(coarse),supervision_rows=len(supervised),
                  input_hidden_dim=256,target='normalized_pressure',feature_depth=feature_depth,seconds=time.perf_counter()-started)
        meta.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
        del features,inputs,sample,x
