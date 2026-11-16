"""Frozen DriverML operator execution and dataset-level provenance."""
from __future__ import annotations
from contextlib import nullcontext
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from adasolver.layers import install_compensation
from adasolver.models.driverml import load_driver_checkpoint, file_sha256
from adasolver.inference.driverml_contract import identity
from adasolver.layers.first_block import cached_first_block_forward


class DriverMLOperator:
    def __init__(self, path, out_dim, device='cuda:0', chunk_size=131072, dtype='float32'):
        if dtype not in ('float32', 'bfloat16') or chunk_size < 1:
            raise ValueError('invalid DriverML inference dtype or chunk size')
        self.device = torch.device(device)
        self.chunk_size, self.dtype = chunk_size, dtype
        self.model, self.metadata = load_driver_checkpoint(path, out_dim, self.device)
        self.context, self.patched = install_compensation(self.model)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, features, compensation=None, permutation_seed=0):
        n = len(features)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(permutation_seed))
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(n)
        self.sync()
        start = time.perf_counter()
        self.context.set(None if compensation is None else torch.as_tensor(compensation)[permutation])
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16) if self.dtype == 'bfloat16' else nullcontext()
        attention = sdpa_kernel(SDPBackend.MATH) if self.dtype == 'float32' else nullcontext()
        try:
            with autocast, attention:
                prediction = self.model(features[permutation].unsqueeze(0).to(self.device), chunk_size=self.chunk_size)
            output = prediction[0].float().cpu()[inverse]
            self.sync()
        finally:
            self.context.clear()
        if not torch.isfinite(output).all():
            raise ValueError('non-finite frozen DriverML prediction')
        return output, time.perf_counter()-start

    @torch.inference_mode()
    def predict_cached(self, features, cache, compensation, permutation_seed=0):
        if self.dtype != 'float32':
            raise ValueError('the DriverML CA cache currently uses the validated float32 path')
        n=len(features)
        permutation=torch.randperm(n,generator=torch.Generator().manual_seed(permutation_seed))
        inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(n)
        self.sync();started=time.perf_counter()
        inputs=features[permutation][None].to(self.device)
        positions=inverse[:cache.inputs.shape[1]].to(self.device)
        weights=torch.as_tensor(compensation)[permutation].to(self.device)
        with sdpa_kernel(SDPBackend.MATH):
            prediction=cached_first_block_forward(self.model,inputs,positions,cache,weights,chunk_size=self.chunk_size)
        result=prediction[0].float().cpu()[inverse];self.sync()
        if not torch.isfinite(result).all(): raise ValueError('non-finite cached DriverML prediction')
        return result,time.perf_counter()-started


def protocol_signature(config):
    keys = {key: file_sha256(config[key]) for key in ('normalizer_path', 'split_path', 'labels_path')}
    keys['checkpoints'] = {field: file_sha256(path) for field, path in config['checkpoints'].items()}
    keys.update(data_path=config['data_path'], protocol=config['protocol'], inference=config['inference'],
                metrics=config['metrics'], physics=config['physics'],
                implementation='global_raw_statistics_v1')
    return identity(keys), keys


def index_identity(indices):
    import hashlib
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
