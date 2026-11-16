"""ShapeNet-Car budget10 protocol shared by training, search and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "budget10_v1"
BUDGETS = (0.375, 0.5, 0.75)
SEEDS = (0, 1, 2)
COARSE_FRACTION = 0.10
DEFAULT_MANIFEST = ROOT / "checkpoints/shapenet_car/shapenet_car_transolver3_varpts_nmin040_e400_s0_manifest.json"
DEFAULT_SPLITS = ROOT / "results/shapenet_car/budget10_v1/splits.json"


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def indices_hash(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def coarse_count(nfull: int, fraction: float = COARSE_FRACTION) -> int:
    if not math.isfinite(fraction) or not math.isclose(fraction, .1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("ShapeNet-Car budget10 requires exactly 10% coarse fraction")
    count = math.floor(nfull * fraction)
    if count < 2:
        raise ValueError("10% coarse budget requires at least two points")
    return count


def validate_indices(indices, nfull: int, count: int | None = None) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if count is not None and len(values) != count:
        raise ValueError(f"indices do not match configured coarse count / budget: {len(values)} != {count}")
    if not len(values) or len(np.unique(values)) != len(values):
        raise ValueError("indices must be nonempty and unique")
    if np.any(values < 0) or np.any(values >= nfull):
        raise ValueError("indices outside reference cloud")
    return values.astype(np.int64, copy=False)


def rng_for(sample_id: str, seed: int, stream: str = "coarse", budget: float | None = None):
    # Stable across filtering, process restarts and candidate iteration order.
    key = f"{PROTOCOL}:{sample_id}:{seed}:{stream}:{budget}"
    integer = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(integer)


def sample_fold(sample_id: str) -> int:
    parts = str(sample_id).split("/")
    if len(parts) != 2 or not parts[0].startswith("param") or parts[1] in {"", ".", ".."}:
        raise ValueError(f"invalid native Car sample ID: {sample_id}")
    return int(parts[0][5:])


def make_splits(sample_ids, operator_metadata: dict) -> dict:
    ids = sorted(str(sid) for sid in sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    test = [sid for sid in ids if sample_fold(sid) == 0]
    pool = [sid for sid in ids if sample_fold(sid) != 0]
    count = max(1, round(len(pool) * operator_metadata["validation_fraction"]))
    order = np.random.default_rng(operator_metadata["split_seed"]).permutation(len(pool))
    val_indices = set(order[:count].tolist())
    train = [sid for i, sid in enumerate(pool) if i not in val_indices]
    validation = [sid for i, sid in enumerate(pool) if i in val_indices]
    if len(validation) < 3:
        raise ValueError("at least three operator validation samples required")
    partition = np.random.default_rng(20260916).permutation(validation).tolist()
    nhead = len(validation) // 2
    ncal = (len(validation) - nhead) // 2
    result = {
        "protocol_version": PROTOCOL,
        "operator_checkpoint_sha256": operator_metadata["checkpoint_sha256"],
        "operator_split_seed": operator_metadata["split_seed"], "development_split_seed": 20260916,
        "operator_train": train, "operator_validation": validation,
        "head_train": train, "head_validation": partition[:nhead],
        "calibration": partition[nhead:nhead+ncal], "confirmation": partition[nhead+ncal:],
        "test": test,
        "disclosure": "Operator validation and normalization used the non-test pool. Fold0 has historical evaluations; no new selection on fold0.",
    }
    result["split_hash"] = json_hash(result)
    return result


def validate_splits(splits: dict, operator_metadata: dict) -> None:
    value = {k: v for k, v in splits.items() if k != "split_hash"}
    if splits.get("split_hash") != json_hash(value):
        raise ValueError("split manifest hash mismatch")
    if splits.get("operator_checkpoint_sha256") != operator_metadata["checkpoint_sha256"]:
        raise ValueError("splits belong to a different operator")
    seen = set()
    for role in ("head_train", "head_validation", "calibration", "confirmation", "test"):
        ids = splits[role]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"split {role} is empty, duplicate, or overlapping")
        if any((sample_fold(sid) == 0) != (role == "test") for sid in ids):
            raise ValueError(f"fold 0 leakage in split {role}")
        seen.update(ids)


def method_config(method: str, **overrides) -> dict:
    if method not in {"uniform", "pi", "cpg", "ca", "fullmesh"}:
        raise ValueError(f"unsupported Car method {method}")
    config = {"method": method, "selection_target": "field"}
    if method == "pi":
        config.update(strategy="two_stage", r=.075, frac_extra=.5, beta=.5,
                      amp=8., ell=.05)
        if overrides.get("strategy") == "two_stage_tempered":
            config.update(proposal_alpha=1.)
    if method in {"cpg", "ca"}:
        config.update(coarse_fraction=.1, feature="gradient", signal="pressure",
                      patch_count=24, neighborhood=4, normalization="q95",
                      score_floor=.1, score_power=1., kappa=4., beta=.75,
                      placement="fps", surface_band=.1, cd_weight=False)
        if overrides.get("strategy") == "surface_stratified":
            config.update(strategy="surface_stratified", wall_scale=.1, wall_amplitude=8.,
                          physics_strength=1., score_bins=4, placement="random")
        if overrides.get("strategy") == "cd_shell":
            config.update(strategy="cd_shell", shell_radius=.1, frac_extra=.375,
                          physics_strength=1., compensation="shell_mass", placement="guided_shell", beta=.5)
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"unknown Car method options: {sorted(unknown)}")
    config.update(overrides)
    if config["selection_target"] not in {"field", "cd"}:
        raise ValueError("selection target must be field or cd")
    if method in {"cpg", "ca"}:
        coarse_count(100, config["coarse_fraction"])
        if config.get("strategy") not in {"surface_stratified", "cd_shell"} and config["placement"] != "fps":
            raise ValueError("Car budget10 implements FPS placement only")
        if config.get("strategy") == "surface_stratified":
            if (config["placement"] not in {"random", "fps"} or config["wall_scale"] <= 0 or
                    config["wall_amplitude"] < 0 or config["physics_strength"] < 0 or
                    config["score_bins"] < 1):
                raise ValueError("invalid surface-stratified configuration")
        if config.get("strategy") == "cd_shell":
            if (config["placement"] != "guided_shell" or config["shell_radius"] <= 0 or
                    not 0 <= config["frac_extra"] < 1 or config["physics_strength"] < 0 or
                    config["compensation"] not in {"shell_mass", "voronoi"} or config["signal"] != "pressure"):
                raise ValueError("invalid Cd shell configuration")
        if config["signal"] not in {"pressure", "speed"}:
            raise ValueError("signal must be pressure or speed")
        if method == "ca" and config["signal"] != "pressure":
            raise ValueError("current CA head is trained for pressure only")
        if config["cd_weight"] and config["signal"] != "pressure":
            raise ValueError("Cd geometry weighting requires pressure signal")
        if config["feature"].startswith("drag_") and not config["cd_weight"]:
            raise ValueError("drag features require Cd geometry weighting")
    return config
