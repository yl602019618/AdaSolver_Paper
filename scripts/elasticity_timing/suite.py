"""Use the same operator, stress head, density and RNGs as accuracy evaluation."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from adasolver.evaluation.elasticity_budget10 import ElasticityRuntime
from adasolver.inference.base import SamplingResult
from adasolver.inference.elasticity_contract import coarse_count, sample_rng
from adasolver.inference.elasticity_density import DensityConfig, ElasticityDensity, scalar_score
from adasolver.inference.pi.elasticity import sample_elasticity_pi
from adasolver.inference.uniform import sample_uniform
from inference_speed.cases import Case
from inference_speed.runtime import FrozenExecutor


class Elasticity10Suite:
    def __init__(self, root, device):
        root = Path(root).resolve()
        relative_bundle = Path("configs/methods/elasticity_budget10_frozen_comparison.json")
        bundle = json.loads((root / relative_bundle).read_text())
        if bundle["status"] != "frozen" or bundle["coarse_cap_count"] != 97:
            raise ValueError("Expected the frozen 97-point comparison")
        self.configs = bundle["methods"]
        self.runtime = ElasticityRuntime(device, batch_size=1)
        head_config = copy.deepcopy(self.configs["ca"])
        for key in ("head_checkpoint", "head_manifest"):
            head_config[key] = str(root / head_config[key])
        self.head = self.runtime.head(head_config)
        self.executor = FrozenExecutor(self.runtime.model, self.runtime.context, device, self.head)
        self.binding = {
            "checkpoint": "checkpoints/elasticity/elasticity_transolver3_varpts_nmin040_best.pt",
            "policy": relative_bundle.as_posix(),
            "configurations": self.configs,
            "coarse_count": 97,
            "n_full": 972,
            "head_bottleneck": self.head.bottleneck,
            "head_parameters": sum(p.numel() for p in self.head.parameters()),
            "profiling_split": "historical_test",
        }

    def sample_ids(self):
        return tuple(range(1800, 2000))

    def cases(self, local_index, seeds, budgets=(.375, .5, .75)):
        sid = self.sample_ids()[local_index]
        coords, target = self.runtime.sample(sid)
        inputs = torch.from_numpy(coords[None])
        reference = np.ascontiguousarray(coords, dtype=np.float64)
        start = time.perf_counter()
        geometry = ElasticityDensity(reference)
        geometry.patches(32)
        geometry_seconds = time.perf_counter() - start
        mean, std = float(self.runtime.mean), float(self.runtime.std)

        def decode(value):
            return value * std + mean

        def score_decode(value):
            # Accuracy evaluation decodes to FP32 before computing FP64 scores.
            return np.asarray(value, np.float32) * np.float32(std) + np.float32(mean)

        for seed in seeds:
            coarse = sample_rng(sid, seed).choice(972, coarse_count(), replace=False)
            for budget in (*budgets, 1.):
                count = round(972 * budget)
                methods = ("full",) if budget == 1. else ("uniform", "pi", "cpg", "ca")
                for method in methods:
                    config = self.configs.get(method, {})
                    density = DensityConfig(**config["density"]) if method in {"cpg", "ca"} else None

                    def select(field, optimized, method=method, density=density, count=count, seed=seed):
                        if method == "full":
                            return SamplingResult(np.arange(972), np.ones(972), np.ones(972, np.float32))
                        if method == "uniform":
                            return sample_uniform(972, count, sample_rng(sid, seed))
                        if method == "pi":
                            return sample_elasticity_pi(reference, count, sample_rng(sid, seed),
                                                        **self.configs["pi"]["pi"])
                        score = scalar_score(reference[coarse], field, density.score, density.neighbors)
                        return geometry.refine(coarse, score, count, density,
                                               sample_rng(sid, seed, refinement=True))

                    yield Case(
                        "elasticity", sid, method, budget, seed, inputs, reference,
                        self.executor, select, coarse.copy() if density is not None else None,
                        decode, score_decode, preparation_seconds=geometry_seconds, target=target,
                        gpu_spec={"density": density, "geometry": geometry, "head": self.head,
                                  "output_mean": mean, "output_std": std,
                                  "elasticity_budget10": True},
                    )
