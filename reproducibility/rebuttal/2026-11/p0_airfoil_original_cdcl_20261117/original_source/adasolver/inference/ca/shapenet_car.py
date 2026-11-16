"""One-output pressure head and its frozen 10% block-0 acquisition."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from adasolver.inference.shapenet_car_contract import coarse_count, validate_indices
from adasolver.layers.block0_cache import Block0Cache, coarse_features, cached_operator_forward
from adasolver.models.checkpoint import sha256_file


class CarPressureHead(nn.Module):
    def __init__(self, input_dim=256, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, features):
        return self.net(features)


class _AcquisitionGraph:
    """One reusable shape; returned tensors own their storage across replays."""
    def __init__(self, ca, pos, fx, nfull):
        self.signature = (tuple(pos.shape), tuple(fx.shape), pos.dtype, fx.dtype, pos.device, nfull)
        self.pos, self.fx = pos.clone(), fx.clone()
        with torch.cuda.device(pos.device):
            current = torch.cuda.current_stream(pos.device)
            stream = torch.cuda.Stream(device=pos.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    ca._acquire_eager(self.pos, self.fx, nfull, capture=True)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.pressure, self.cache = ca._acquire_eager(self.pos, self.fx, nfull, capture=True)
            current.wait_stream(stream)

    def run(self, pos, fx, capture):
        with torch.cuda.device(pos.device):
            self.pos.copy_(pos)
            self.fx.copy_(fx)
            self.graph.replay()
            pressure = self.pressure.clone()
            if not capture:
                return pressure, None
            # A later query must not overwrite an earlier query's cache.
            c = self.cache
            cache = Block0Cache(c.hidden.clone(), c.normalized.clone(), c.slice_weights.clone(),
                                self.pos.clone(), self.fx.clone(), c.operator_id)
            return pressure, cache


class ShapeNetCarCA:
    def __init__(self, operator, head_manifest, binding, device="cpu", *, use_cuda_graph=True):
        self.operator = operator
        path = Path(head_manifest)
        self.metadata = json.loads(path.read_text())
        meta = self.metadata
        for key in ("operator_sha256", "normalizer_sha256"):
            if meta.get(key) != binding[key]:
                raise ValueError(f"CA head belongs to a different {key}")
        if meta.get("target") != "surface_pressure" or meta.get("coarse_fraction") != .1:
            raise ValueError("Car CA requires a surface-pressure head trained at 10%")
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        if sha256_file(checkpoint) != meta["head_sha256"]:
            raise ValueError("CA head checkpoint hash mismatch")
        self.head = CarPressureHead(meta["input_dim"], meta["width"]).to(device)
        self.head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        self.head.eval().requires_grad_(False)
        self.use_cuda_graph = use_cuda_graph
        self._acquisition_graph = None

    @torch.no_grad()
    def feature_pass(self, pos, fx, nfull, *, capture=True):
        if pos.shape[1] != coarse_count(nfull):
            raise ValueError("CA feature extraction requires the actual 10% subset")
        return coarse_features(self.operator, pos, fx, capture=capture, nfull=nfull)

    @torch.no_grad()
    def head_pass(self, features):
        """The learned MLP alone; block-0 work is exposed by feature_pass."""
        pressure = self.head(features)[0, :, 0]
        return pressure * self.metadata["target_std"] + self.metadata["target_mean"]

    @torch.no_grad()
    def _acquire_eager(self, pos, fx, nfull, *, capture=True):
        features, cache = self.feature_pass(pos, fx, nfull, capture=capture)
        return self.head_pass(features), cache

    @torch.no_grad()
    def prepare_acquisition(self, pos, fx, nfull):
        """Warm/capture once at model startup; the coarse budget stays exactly 10%."""
        if pos.shape[1] != coarse_count(nfull):
            raise ValueError("CA feature extraction requires the actual 10% subset")
        context = getattr(self.operator.net.blocks[0].Attn, "_compensation_context", None)
        if context is not None and context.weights is not None:
            raise ValueError("prepare the uniform coarse graph without compensation")
        if self.use_cuda_graph and pos.is_cuda:
            signature = (tuple(pos.shape), tuple(fx.shape), pos.dtype, fx.dtype, pos.device, nfull)
            if self._acquisition_graph is None or self._acquisition_graph.signature != signature:
                self._acquisition_graph = _AcquisitionGraph(self, pos, fx, nfull)

    @torch.no_grad()
    def acquire(self, pos, fx, nfull, *, capture=True):
        context = getattr(self.operator.net.blocks[0].Attn, "_compensation_context", None)
        if self.use_cuda_graph and pos.is_cuda and (context is None or context.weights is None):
            self.prepare_acquisition(pos, fx, nfull)
            return self._acquisition_graph.run(pos, fx, capture)
        return self._acquire_eager(pos, fx, nfull, capture=capture)

    def final(self, pos, fx, coarse, final, cache, compensation):
        return cached_operator_forward(self.operator, pos, fx, coarse, final, cache, compensation)
