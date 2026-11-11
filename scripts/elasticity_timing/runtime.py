"""Charge online Voronoi weights with the same nearest-neighbor ties as accuracy."""
import time

import torch

from adasolver.inference.adaptive import subset_mass
from adasolver.utils.interpolation import reconstruct_field
from inference_speed.runtime import GraphCall
from .gpu_density import Elasticity10Plan


class ElasticityAdaptiveRuntime:
    def __init__(self, case, prepared_mass=None):
        self.case = case
        self.executor = case.executor
        self.device = self.executor.device
        self.plan = Elasticity10Plan(case)
        self.policy = self.plan.policy + "; original SciPy Voronoi ties and compensation"
        self.mass = self.adapter = None
        self.graph = None
        self.graph_setup_seconds = 0.
        self.stage_graphs = {}
        self.original_name = self.fast_name = "Delaunay linear + nearest fallback"

    def _acquire(self, inputs):
        coarse = inputs[:, self.case.coarse_tensor]
        if self.case.method == "cpg":
            return (self.executor.coarse(coarse, optimized=False),)
        return self.executor.acquire(coarse, optimized=False)

    def _acquire_and_select(self, inputs):
        acquired = self._acquire(inputs)
        return (self.plan.select_indices(acquired[0]), *acquired[1:])

    def _weights(self, indices):
        host_indices = indices.cpu().numpy()
        mass = subset_mass(host_indices, self.case.reference, 972)
        weights = (mass / mass.mean()) ** self.plan.config.beta
        weights = torch.as_tensor(weights.astype("float32"), device=self.device)
        return host_indices, weights

    @torch.no_grad()
    def query(self, mode="sampled"):
        if self.graph is None:
            began = time.perf_counter()
            modules = torch.nn.ModuleList([self.executor.operator, self.executor.head])
            self.graph = GraphCall(self._acquire_and_select, (self.case.resident,), modules)
            self.graph_setup_seconds += time.perf_counter() - began
        selected = self.graph(self.case.resident)
        indices, cache = selected[0], selected[1:]
        host_indices, weights = self._weights(indices)
        inputs = self.case.resident[:, indices]
        before = self.executor.graph_setup_seconds
        if self.case.method == "ca":
            output = self.executor.cached(inputs, weights, cache, optimized=True)
        else:
            output = self.executor.direct(inputs, weights, optimized=True)
        self.graph_setup_seconds += self.executor.graph_setup_seconds - before
        prediction = self.case.decode(output)[0].cpu().numpy()
        if mode != "sampled":
            prediction = reconstruct_field(host_indices, prediction, self.case.reference)[0]
        return host_indices, prediction

    def _stage_run(self, key, fn, *args):
        if key not in self.stage_graphs:
            modules = torch.nn.ModuleList([self.executor.operator, self.executor.head])
            self.stage_graphs[key] = GraphCall(fn, args, modules)
        return self.stage_graphs[key](*args)

    @torch.no_grad()
    def stages(self):
        result = {}

        def mark(name, fn):
            torch.cuda.synchronize(self.device)
            began = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(self.device)
            result[name] = 1000 * (time.perf_counter() - began)
            return value

        acquired = mark("coarse_ms", lambda: self._stage_run("coarse", self._acquire, self.case.resident))

        def sample():
            indices = self._stage_run("sampling", self.plan.select_indices, acquired[0])
            host_indices, weights = self._weights(indices)
            return indices, host_indices, weights

        indices, host_indices, weights = mark("density_sampling_compensation_ms", sample)

        def final(inputs, idx, w, *cache):
            if self.case.method == "ca":
                return self.executor.cached(inputs[:, idx], w, cache, optimized=False)
            return self.executor.direct(inputs[:, idx], w, optimized=False)

        output = mark("fine_ms", lambda: self._stage_run("final", final, self.case.resident,
                                                        indices, weights, *acquired[1:]))
        prediction = mark("decode_ms", lambda: self._stage_run("decode", self.case.decode, output))[0]
        values = mark("sampled_export_ms", lambda: prediction.cpu().numpy())
        for mode in ("original", "fast"):
            mark("reconstruction_" + mode + "_ms",
                 lambda: reconstruct_field(host_indices, values, self.case.reference))
            result["full_export_" + mode + "_ms"] = 0.
        return result
