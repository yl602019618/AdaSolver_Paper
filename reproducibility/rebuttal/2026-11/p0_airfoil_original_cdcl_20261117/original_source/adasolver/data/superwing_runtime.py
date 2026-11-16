"""Frozen, compensated SuperWing execution using the existing T3 backbone."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from adasolver.inference.superwing_contract import MODEL, identity
from adasolver.layers.compensation import install_compensation
from adasolver.layers.first_block import first_block_features, cached_first_block_forward
from adasolver.models.adapter import Transolver3ModelSpec
from adasolver.models.checkpoint import load_operator, sha256_file


class SuperWingRuntime:
    def __init__(self, config, device='cuda:0'):
        self.device = torch.device(device)
        self.config = config
        with open(config['operator_manifest']) as handle:
            self.metadata = json.load(handle)
        if self.metadata['model'] != MODEL or self.metadata['benchmark'] != 'superwing':
            raise ValueError('SuperWing operator manifest architecture mismatch')
        self.model = load_operator(config['checkpoint'], Transolver3ModelSpec(**MODEL),
                                   self.device, self.metadata['checkpoint_sha256'])
        self.model.eval().requires_grad_(False)
        if sha256_file(config['normalizer_path']) != self.metadata['normalizer_sha256']:
            raise ValueError('normalizer does not belong to this operator')
        self.context, self.patched = install_compensation(self.model)
        self.signature_fields = dict(protocol=config['protocol'], operator=self.metadata['checkpoint_sha256'],
                                     normalizer=self.metadata['normalizer_sha256'],
                                     split=sha256_file(config['split_path']), dtype='float32',
                                     attention_backend='math', implementation='superwing_v1')
        self.signature = identity(self.signature_fields)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, features, compensation=None, cache=None, coarse_positions=None):
        self.sync()
        start = time.perf_counter()
        features = torch.as_tensor(features, device=self.device, dtype=torch.float32)[None]
        comp = None if compensation is None else torch.as_tensor(compensation, device=self.device)
        try:
            with sdpa_kernel(SDPBackend.MATH):
                if cache is None:
                    self.context.set(comp)
                    result = self.model.net(features, use_checkpoint=False, input_list=False)
                else:
                    result = cached_first_block_forward(self.model, features, coarse_positions, cache,
                                                         comp, chunk_size=131072)
            prediction = result[0].cpu().numpy()
            self.sync()
        finally:
            self.context.clear()
        if not np.isfinite(prediction).all():
            raise ValueError('non-finite SuperWing prediction')
        return prediction, time.perf_counter()-start

    @torch.inference_mode()
    def acquire_features(self, features, capture=True):
        self.context.clear()
        self.sync()
        start = time.perf_counter()
        inputs = torch.as_tensor(features, dtype=torch.float32, device=self.device)[None]
        with sdpa_kernel(SDPBackend.MATH):
            h, cache = first_block_features(self.model, inputs, capture=capture)
        self.sync()
        return h, cache, time.perf_counter()-start
