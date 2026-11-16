"""Explicit split/config evaluation for the <=10% Elasticity experiments."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from adasolver.data.elasticity import ElasticityAdapter
from adasolver.inference.elasticity_contract import BUDGETS, SPLITS, coarse_count, sample_rng, split_ids
from adasolver.inference.elasticity_density import DensityConfig, ElasticityDensity, scalar_score
from adasolver.inference.pi.elasticity import cavity_boundary_distance, sample_elasticity_pi
from adasolver.inference.uniform import sample_uniform
from adasolver.layers import install_compensation
from adasolver.layers.first_block import first_block_features, cached_first_block_forward
from adasolver.metrics.rauec import relative_area_under_error_curve
from adasolver.models import load_elasticity_operator
from adasolver.models.checkpoint import sha256_file
from adasolver.utils import budget_count
from adasolver.utils.interpolation import reconstruct_field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATOR = ROOT / "checkpoints/elasticity/elasticity_transolver3_varpts_nmin040_best.pt"
DEFAULT_MANIFEST = ROOT / "checkpoints/elasticity/elasticity_transolver3_varpts_nmin040_manifest.json"
DEFAULT_NORMALIZER = ROOT / "checkpoints/elasticity/elasticity_transolver3_varpts_nmin040_ynorm.npz"
DEFAULT_DATA = None


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_config(path):
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def summarize(rows, uniform_curve=None):
    keys = [(r["sample_id"], r["seed"], r["budget_fraction"]) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError("raw records must be nonempty and unique")
    ids, seeds = sorted({k[0] for k in keys}), sorted({k[1] for k in keys})
    if set(keys) != {(i, s, b) for i in ids for s in seeds for b in BUDGETS}:
        raise ValueError("incomplete sample/seed/budget product")
    curve = [float(np.mean([r["relative_l2"] for r in rows if r["budget_fraction"] == b])) for b in BUDGETS]
    if not np.isfinite(curve).all():
        raise ValueError("nonfinite error curve")
    result = {"sample_ids": ids, "samples": len(ids), "seeds": seeds, "records": len(rows),
              "budgets": list(BUDGETS), "curve": curve}
    if uniform_curve is not None:
        result["rAUEC"] = relative_area_under_error_curve(BUDGETS, curve, uniform_curve)
    return result


class ElasticityRuntime:
    def __init__(self, device="cuda:0", *, operator=DEFAULT_OPERATOR, manifest=DEFAULT_MANIFEST,
                 normalizer=DEFAULT_NORMALIZER, data_path=DEFAULT_DATA, batch_size=32):
        self.device = torch.device(device)
        self.operator_path, self.manifest_path = str(operator), str(manifest)
        self.metadata = json.loads(Path(manifest).read_text())
        if self.metadata.get("nmin_count") != 389:
            raise ValueError("budget10 experiments require the official Nmin=389 operator")
        self.operator_hash = sha256_file(operator)
        self.normalizer_hash = sha256_file(normalizer)
        if self.normalizer_hash != self.metadata["normalizer_sha256"]:
            raise ValueError("operator normalizer identity mismatch")
        archive = np.load(normalizer)
        self.mean, self.std = np.float32(archive["mean"].item()), np.float32(archive["std"].item())
        self.model = load_elasticity_operator(operator, manifest, self.device).requires_grad_(False).eval()
        self.context, _ = install_compensation(self.model)
        self.adapter = ElasticityAdapter(data_path or os.environ.get("ADASOLVER_ELASTICITY_DATA")
                                         or self.metadata["data_path"])
        self.batch_size = batch_size
        self.samples, self.geometry, self.acquisition, self.heads, self.pi_distance = {}, {}, {}, {}, {}

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def sample(self, sid):
        if sid not in self.samples:
            data = self.adapter.load_ids([sid])
            self.samples[sid] = (data.pos[0].numpy(), data.target[0].numpy())
        return self.samples[sid]

    def prefetch(self, ids):
        missing = [sid for sid in ids if sid not in self.samples]
        if missing:
            data = self.adapter.load_ids(missing)
            for index, sid in enumerate(missing):
                self.samples[sid] = (data.pos[index].numpy(), data.target[index].numpy())

    def head(self, config):
        key = (config["head_checkpoint"], config["head_manifest"])
        if key not in self.heads:
            from adasolver.inference.ca.elasticity_stress import load_stress_head
            self.heads[key] = load_stress_head(*key, operator_sha256=self.operator_hash,
                                             normalizer_sha256=self.normalizer_hash, device=self.device)[0]
        return self.heads[key]

    @torch.no_grad()
    def acquire(self, config, pairs):
        method = config["method"]
        fraction = config["density"].get("coarse_fraction", .1)
        n0 = coarse_count(972, fraction)
        identity = (method, fraction, config.get("head_checkpoint"), config.get("head_manifest"))
        missing = [(sid, seed) for sid, seed in pairs if (identity, sid, seed) not in self.acquisition]
        for start in range(0, len(missing), self.batch_size):
            group = missing[start:start+self.batch_size]
            chosen = [sample_rng(sid, seed).choice(972, n0, replace=False) for sid, seed in group]
            pos = torch.as_tensor(np.stack([self.sample(sid)[0][ix] for (sid, _), ix in zip(group, chosen)]),
                                  device=self.device)
            self.context.clear()
            self.sync()
            began = time.perf_counter()
            if method == "cpg":
                field = self.model(pos, None)
            elif method == "ca":
                features, _ = first_block_features(self.model, pos)
                field = self.head(config)(features)
            else:
                raise ValueError("unknown acquisition method")
            field = (field * float(self.std) + float(self.mean)).cpu().numpy()
            self.sync()
            seconds = (time.perf_counter() - began) / len(group)
            for key, indices, values in zip(group, chosen, field):
                self.acquisition[(identity, *key)] = (indices, values, seconds)
        return {pair: self.acquisition[(identity, *pair)] for pair in pairs}

    @torch.no_grad()
    def evaluate(self, config, ids, seeds=(0, 1, 2), *, split="development", uniform_curve=None,
                 output=None, save_indices=False, online_cache=False):
        method = config["method"]
        start_id, end_id = SPLITS[split]
        if not ids or any(not start_id <= i < end_id for i in ids):
            raise ValueError("sample IDs do not belong to declared split")
        if split in {"confirmation", "independent_holdout", "historical_test"} and config.get("status") != "frozen":
            raise ValueError("confirmation/final evaluation requires a frozen config")
        if method not in {"uniform", "pi", "cpg", "ca"}:
            raise ValueError("unknown method")
        self.prefetch(ids)
        c = DensityConfig(**config["density"]) if method in {"cpg", "ca"} else None
        pairs = [(sid, seed) for seed in seeds for sid in ids]
        cache = self.acquire(config, pairs) if c is not None and not online_cache else {}
        cid, rows, arrays = config_id(config), [], {}
        started = time.perf_counter()
        for budget in BUDGETS:
            count = budget_count(972, budget)
            for start in range(0, len(pairs), self.batch_size):
                group = pairs[start:start+self.batch_size]
                first_cache = None
                if c is not None and online_cache:
                    n0 = coarse_count(972, c.coarse_fraction)
                    coarse_sets = [sample_rng(sid, seed).choice(972, n0, replace=False) for sid, seed in group]
                    coarse_coords = np.stack([self.sample(sid)[0][ix]
                                              for (sid, _), ix in zip(group, coarse_sets)])
                    self.context.clear()
                    self.sync()
                    began = time.perf_counter()
                    if method == "ca":
                        from adasolver.inference.ca.elasticity_stress import sparse_stress_features
                        features, first_cache = sparse_stress_features(self.model, coarse_coords)
                        field = self.head(config)(features)
                    else:
                        coarse_pos = torch.as_tensor(coarse_coords, device=self.device)
                        field = self.model(coarse_pos, None)
                    field = (field * float(self.std) + float(self.mean)).cpu().numpy()
                    self.sync()
                    acq_seconds = (time.perf_counter()-began)/len(group)
                    for pair, ix, values in zip(group, coarse_sets, field):
                        cache[pair] = (ix, values, acq_seconds)
                sampled, sampling_times = [], []
                for sid, seed in group:
                    coords, _ = self.sample(sid)
                    began = time.perf_counter()
                    if method == "uniform":
                        selection = sample_uniform(972, count, sample_rng(sid, seed))
                    elif method == "pi":
                        selection = sample_elasticity_pi(coords, count, sample_rng(sid, seed), **config.get("pi", {}))
                    else:
                        coarse, values, _ = cache[(sid, seed)]
                        score = scalar_score(coords[coarse], values, c.score, c.neighbors)
                        if sid not in self.geometry:
                            self.geometry[sid] = ElasticityDensity(coords)
                        selection = self.geometry[sid].refine(coarse, score, count, c,
                                                              sample_rng(sid, seed, refinement=True))
                    sampling_times.append(time.perf_counter() - began)
                    sampled.append(selection)
                final_coords = np.stack([self.sample(sid)[0][v.indices] for (sid, _), v in zip(group, sampled)])
                weights = (None if method == "uniform" else
                           torch.as_tensor(np.stack([v.compensation for v in sampled]), device=self.device))
                self.context.set(weights)
                self.sync()
                began = time.perf_counter()
                if first_cache is not None:
                    from adasolver.inference.ca.elasticity_stress import cached_stress_forward
                    prediction = cached_stress_forward(self.model, final_coords, first_cache, weights)
                else:
                    pos = torch.as_tensor(final_coords, device=self.device)
                    prediction = self.model(pos, None)
                prediction = prediction * float(self.std) + float(self.mean)
                self.sync()
                inference_seconds = (time.perf_counter() - began) / len(group)
                prediction = prediction.cpu().numpy()
                self.context.clear()
                for j, ((sid, seed), sampling) in enumerate(zip(group, sampled)):
                    coords, target = self.sample(sid)
                    began = time.perf_counter()
                    restored, fallback = reconstruct_field(sampling.indices, prediction[j], coords)
                    reconstruction_seconds = time.perf_counter() - began
                    numerator, denominator = float(np.linalg.norm(restored-target)), float(np.linalg.norm(target))
                    acq = cache[(sid, seed)][2] if c is not None else 0.
                    row = {"sample_id": sid, "seed": seed, "split": split, "method": method, "config_id": cid,
                           "budget_fraction": budget, "budget_count": count,
                           "coarse_count": coarse_count(972, c.coarse_fraction) if c is not None else 0,
                           "relative_l2": numerator / max(denominator, 1e-12),
                           "relative_l2_numerator": numerator, "relative_l2_denominator": denominator,
                           "interpolation_fallback_count": fallback, "sampling_seconds": sampling_times[j],
                           "acquisition_seconds": acq, "inference_seconds": inference_seconds,
                           "reconstruction_seconds": reconstruction_seconds,
                           "stage_sum_seconds": acq+sampling_times[j]+inference_seconds+reconstruction_seconds,
                           "execution": "cached_final" if first_cache is not None else "uncached",
                           "timing_batch_size": len(group), "sampling_metadata": sampling.metadata}
                    if save_indices:
                        key = f"s{sid}_seed{seed}_n{count}"
                        arrays[key] = sampling.indices
                        if c is not None:
                            arrays[key+"_coarse"] = cache[(sid, seed)][0]
                        row["indices_key"] = key
                    rows.append(row)
        result = summarize(rows, uniform_curve)
        result.update(benchmark="elasticity", method=method, split=split, config=config, config_id=cid,
                      operator_sha256=self.operator_hash, normalizer_sha256=self.normalizer_hash,
                      evaluation_wall_seconds=time.perf_counter()-started,
                      operator_checkpoint_selection="historical_test_used_for_operator_best",
                      timing_note="stage timings are batched estimates; use separate batch=1 runtime report")
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            raw_path = output.with_suffix(".raw.jsonl")
            raw_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            result["raw_file"] = str(raw_path)
            if arrays:
                index_path = output.with_suffix(".indices.npz")
                np.savez_compressed(index_path, **arrays)
                result["indices_file"] = str(index_path)
            output.write_text(json.dumps(result, indent=2)+"\n")
        return result, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=SPLITS, default="development")
    p.add_argument("--samples", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--uniform", type=Path)
    p.add_argument("--save-indices", action="store_true")
    p.add_argument("--online-cache", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    runtime = ElasticityRuntime(args.device, batch_size=args.batch_size)
    uniform = json.loads(args.uniform.read_text())["curve"] if args.uniform else None
    result, _ = runtime.evaluate(load_config(args.config), split_ids(args.split, args.samples), args.seeds,
                                 split=args.split, uniform_curve=uniform, output=args.output,
                                 save_indices=args.save_indices, online_cache=args.online_cache)
    print(json.dumps({k: v for k, v in result.items() if k != "sample_ids"}, indent=2))


if __name__ == "__main__":
    main()
