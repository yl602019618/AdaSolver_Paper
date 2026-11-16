"""Deployment-matched, full-fit SuperWing first-block feature banks."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from adasolver.data.superwing import SuperWingAdapter
from adasolver.data.superwing_runtime import SuperWingRuntime
from adasolver.inference.superwing_contract import NCOARSE,coarse_indices,write_json
from adasolver.models.checkpoint import sha256_file


def build_bank(config,*,split,output,device,rank=0,world=1,limit=None):
    if split not in ('fit','head_val'):
        raise ValueError('head banks may only use fit and head_val')
    adapter=SuperWingAdapter(config['data_path'],config['normalizer_path'],config['split_path'])
    runtime=SuperWingRuntime(config,device)
    all_ids=adapter.splits[split]
    role='full' if limit is None else 'diagnostic'
    ids=(all_ids if limit is None else all_ids[:limit])[rank::world]
    view_seed=11 if split=='fit' else 29
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    metadata=dict(split=split,sample_ids=ids,sample_count=len(ids),rows=len(ids)*NCOARSE,
                  coarse_count=NCOARSE,coarse_fraction=.1,feature_depth=1,input_hidden_dim=128,
                  feature_dtype='float32',target='Cp_GT',target_dtype='float32',view_seed=view_seed,
                  operator_sha256=runtime.metadata['checkpoint_sha256'],normalizer_sha256=runtime.metadata['normalizer_sha256'],
                  split_sha256=sha256_file(config['split_path']),dataset_role=role,rank=rank,world=world,
                  status='building',completed_samples=0)
    meta_path=output/'manifest.json'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        for key in metadata:
            if key not in ('status','completed_samples') and previous[key]!=metadata[key]:
                raise ValueError(f'feature-bank resume mismatch: {key}')
        if previous['status']=='complete':
            return previous
        metadata['completed_samples']=previous['completed_samples']
    write_json(meta_path,metadata)
    mode='r+' if (output/'features.npy').exists() else 'w+'
    features=np.lib.format.open_memmap(output/'features.npy',mode=mode,dtype=np.float32,shape=(len(ids),NCOARSE,128))
    target=np.lib.format.open_memmap(output/'target.npy',mode=mode,dtype=np.float32,shape=(len(ids),NCOARSE))
    indices=np.lib.format.open_memmap(output/'indices.npy',mode=mode,dtype=np.int32,shape=(len(ids),NCOARSE))
    started=time.perf_counter()
    for row in range(metadata['completed_samples'],len(ids)):
        sample_id=ids[row]
        g=adapter.geometry(sample_id)
        selected=coarse_indices(sample_id,view_seed)
        h,_,_=runtime.acquire_features(g.features(selected),capture=False)
        values=h[0].cpu().numpy()
        cp=np.asarray(adapter.arrays['data'][sample_id,0]).reshape(-1)[selected]
        if not np.isfinite(values).all() or not np.isfinite(cp).all():
            raise ValueError('non-finite CA feature/pressure target')
        features[row]=values
        target[row]=cp
        indices[row]=selected
        if (row+1)%64==0 or row+1==len(ids):
            features.flush();target.flush();indices.flush()
            metadata['completed_samples']=row+1
            write_json(meta_path,metadata)
            print(f'bank {split} shard {rank}: {row+1}/{len(ids)} ({time.perf_counter()-started:.1f}s)',flush=True)
    metadata.update(status='complete',generation_seconds=time.perf_counter()-started,
                    feature_file_bytes=(output/'features.npy').stat().st_size)
    write_json(meta_path,metadata)
    return metadata
