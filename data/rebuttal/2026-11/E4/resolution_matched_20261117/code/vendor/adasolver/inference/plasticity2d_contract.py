"""Explicit full20 configuration, artifact and sampling identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

N_FULL = 3131
N_COARSE = 313
BUDGETS = (0.375, 0.5, 0.75)
METRIC_ID = "displacement_full20_relative_l2_v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/benchmarks/plasticity2d_full.yaml"


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def indices_hash(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_config(path=DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("benchmark") != "plasticity2d" or cfg.get("task_version") != "full20_v1":
        raise ValueError("expected Plasticity full20_v1 config")
    model = cfg["model"]
    if any(model.get(k) != v for k, v in {
        "out_dim": 40, "frame_start": 0, "predicted_frames": 20,
        "space_dim": 2, "fun_dim": 1, "displacement_channels": 2,
    }.items()):
        raise ValueError("full20 model layout mismatch")
    ev = cfg["evaluation"]
    if (tuple(ev["budgets"]) != BUDGETS or ev["coarse_count"] != N_COARSE
            or ev["coarse_fraction"] != 0.1 or ev["metric_id"] != METRIC_ID):
        raise ValueError("full20 evaluation contract mismatch")
    if cfg["training"]["nmin_fraction"] != 0.4:
        raise ValueError("official variable-N training requires nmin=0.4")
    occupied = set()
    for name, (start, stop) in cfg["splits"].items():
        ids = set(range(start, stop))
        if start < 0 or stop > 987 or not ids or occupied & ids:
            raise ValueError(f"invalid or overlapping split {name}")
        occupied.update(ids)
    return cfg


def rng_for(seed: int, sample_id: int, stage: int, budget_id: int = 0):
    return np.random.default_rng(np.random.SeedSequence([seed, sample_id, stage, budget_id]))


def validate_method_config(config):
    method = config.get("method")
    if method not in ("uniform", "pi", "cpg", "ca"):
        raise ValueError("unknown full20 method")
    allowed = {"method"}
    if method != "uniform":
        allowed |= {"gamma", "power", "beta"}
        if not 0 <= config.get("gamma", .8) < 1 or config.get("power", 1.) <= 0:
            raise ValueError("invalid density gamma/power")
        if not 0 <= config.get("beta", 1.) <= 1:
            raise ValueError("invalid compensation beta")
    if method == "pi":
        allowed |= {"ell", "curvature_weight"}
        if config.get("ell", .1) <= 0 or config.get("curvature_weight", 0.) < 0:
            raise ValueError("invalid PI prior parameters")
    if method in ("cpg", "ca"):
        allowed |= {"score", "neighbors", "lift", "patch_count", "kappa", "placement", "laplacian_weight",
                    "coarse_fraction", "coarse_count"}
        if config.get("coarse_fraction", .1) != .1 or config.get("coarse_count", 313) != 313:
            raise ValueError("full20 coarse budget is fixed at 10%, 313 points")
        if config.get("score", "g") not in ("g", "l", "gl", "increment"):
            raise ValueError("unknown variation score")
        if config.get("lift", "nearest") not in ("nearest", "idw") or config.get("placement", "fps") not in ("random", "fps"):
            raise ValueError("unknown lift or placement")
        if not 0 <= config.get("laplacian_weight", .5) <= 1 or config.get("kappa", 4.) < 1:
            raise ValueError("invalid variation mixture or kappa")
        if not isinstance(config.get("neighbors",16),int) or not 6 <= config.get("neighbors",16) <= 312:
            raise ValueError("neighbors must be an integer in [6,312]")
        if not isinstance(config.get("patch_count",24),int) or not 1 <= config.get("patch_count",24) <= 313:
            raise ValueError("invalid patch count")
    if method == "ca":
        allowed.add("head_manifest")
        if not config.get("head_manifest"):
            raise ValueError("CA requires its operator-bound head manifest")
    if set(config)-allowed:
        raise ValueError(f"unknown full20 method options: {sorted(set(config)-allowed)}")
    for value in config.values():
        if isinstance(value,(float,int)) and not np.isfinite(value):
            raise ValueError("method parameters must be finite")


def validate_indices(indices, count, coarse=None):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != count
            or len(np.unique(indices)) != count or np.any(indices < 0) or np.any(indices >= N_FULL)):
        raise ValueError("invalid full20 selected point count or indices")
    if coarse is not None:
        validate_indices(coarse, N_COARSE)
        if not np.isin(coarse, indices).all():
            raise ValueError("final set must contain all 313 coarse points")


def load_data(config, normalizer=None, arrays=None):
    from adasolver.data.plasticity2d import Plasticity2DAdapter, PlasticityLayout
    return Plasticity2DAdapter(
        config["data_path"], {k: tuple(v) for k, v in config["splits"].items()}, arrays,
        layout=PlasticityLayout(frame_start=0, predicted_frames=20),
        input_normalizer_path=normalizer,
    ).load()
