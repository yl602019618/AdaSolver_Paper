"""CA revisions reuse the official model, acquisition, cache, reconstruction and metric.

The original evaluation module and all frozen v1 source hashes are preserved.
"""
from __future__ import annotations
import time
import numpy as np
import torch

from adasolver.evaluation.plasticity2d_full import Full20Evaluator, sync
from adasolver.inference.ca.plasticity2d_revision import refine_ca, validate_config
from adasolver.inference.plasticity2d_contract import (
    BUDGETS, METRIC_ID, N_COARSE, N_FULL, indices_hash, object_hash, rng_for,
)
from adasolver.inference.plasticity2d_sampling import REFERENCE, solution_score
from adasolver.layers.first_block import cached_first_block_forward
from adasolver.metrics.plasticity2d import full20_metrics
from adasolver.utils.interpolation import reconstruct_field


class Full20CARevisionEvaluator(Full20Evaluator):
    @torch.no_grad()
    def predict(self, split_name, local_index, seed, budget_id, config, *, reuse_acquisition=False):
        if "revision" not in config:
            return super().predict(split_name, local_index, seed, budget_id, config,
                                   reuse_acquisition=reuse_acquisition)
        validate_config(config)
        split = self.data.splits[split_name]
        sid = split.sample_ids[local_index]
        budget, count = BUDGETS[budget_id], round(BUDGETS[budget_id]*N_FULL)
        pos, fx = split.pos[local_index:local_index+1], split.fx[local_index:local_index+1]
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.context.clear()
        sync(self.device)
        started = time.perf_counter()
        coarse, features, cache, acquisition_seconds, score_seconds, head_id = self.acquire(
            "ca", sid, seed, pos, fx, config, reuse_acquisition)
        score = solution_score(features, config)
        sample_start = time.perf_counter()
        profile = self.data.input_normalizer.decode(fx)[0, :, 0].numpy().reshape(101, 31)[:, 0]
        refine_seed = int(rng_for(seed, sid, 22, budget_id).integers(2**31))
        sample = refine_ca(coarse, score, profile, count, refine_seed, config)
        indices, weights = sample.indices, sample.compensation
        compensation_seconds = sample.metadata["compensation_seconds"]
        sampling_seconds = time.perf_counter()-sample_start-compensation_seconds
        final_pos, final_fx = pos[:, indices].to(self.device), fx[:, indices].to(self.device)
        weights_tensor = torch.as_tensor(weights, device=self.device)
        self.context.set(weights_tensor)
        mapping = np.empty(N_FULL, np.int64)
        mapping[indices] = np.arange(count)
        sync(self.device)
        final_start = time.perf_counter()
        prediction = cached_first_block_forward(
            self.operator, torch.cat((final_pos, final_fx), dim=-1), mapping[coarse], cache,
            weights_tensor, chunk_size=count,
        )
        sync(self.device)
        final_seconds = time.perf_counter()-final_start
        sparse = prediction[0].cpu().numpy()
        self.context.clear()
        reconstruct_start = time.perf_counter()
        reconstructed, fallback = reconstruct_field(indices, sparse, REFERENCE)
        reconstruction_seconds = time.perf_counter()-reconstruct_start
        total_seconds = time.perf_counter()-started
        record = {
            "benchmark": "plasticity2d", "task_version": "full20_v1", "method_revision": "ca_v2",
            "metric_id": METRIC_ID, "method": "ca", "config_id": object_hash(config), "effective_config": config,
            "sample_id": sid, "split": split_name, "seed": seed,
            "rng_scheme": "SeedSequence(seed,sample_id,stage,budget_id)_v1",
            "budget_fraction": budget, "budget_count": count, "selected_count": len(indices),
            "selected_indices": indices.tolist(), "selected_indices_hash": indices_hash(indices),
            "coarse_count": N_COARSE, "coarse_fraction": .1, "coarse_indices_hash": indices_hash(coarse),
            "operator_id": self.metadata["checkpoint_sha256"], "head_id": head_id,
            "truth_access_inference": False, "input_normalizer_id": self.metadata["input_normalizer_sha256"],
            "interpolation_backend": "scipy_linear_nearest_fallback", "reconstruction_coordinates": "index_lattice",
            "interpolation_fallback_count": fallback, "fallback_fraction": fallback/N_FULL,
            "compensation_type": sample.metadata["compensation_type"], "beta": config.get("beta", 1.),
            "acquisition_seconds": acquisition_seconds, "score_seconds": score_seconds,
            "sampling_seconds": sampling_seconds, "compensation_seconds": compensation_seconds,
            "final_inference_seconds": final_seconds, "reconstruction_seconds": reconstruction_seconds,
            "total_seconds": total_seconds, "timing_mode": "search_cached" if reuse_acquisition else "independent_request",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if torch.device(self.device).type == "cuda" else None,
            "sampling_metadata": sample.metadata, "status": "ok",
            **full20_metrics(reconstructed, split.target[local_index].numpy()),
        }
        arrays = {"indices": indices, "density": sample.sampling_density, "compensation": weights,
                  "prediction": reconstructed, "coarse": coarse}
        return record, arrays
