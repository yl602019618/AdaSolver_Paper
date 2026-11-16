"""One resolved configuration for Airfoil calibration and deployment."""
from __future__ import annotations
import json
from pathlib import Path
from .airfoil2d_contract import coarse_count, BUDGETS

METHODS = ("uniform", "pi", "pi_voronoi", "cpg", "cpg_force_ratio",
           "cpg_force_ratio_lift_support", "cpg_single_feature", "ca")
DEFAULTS = dict(coarse_fraction=.10, feature="gradient_lsq", feature_k=16,
                ell=.1, normalization="q95", uniform_floor=.1, score_gamma=.5,
                beta=1., sampler="weighted", patch_count=96, placement="random", kappa=8.,
                operator_stage1_cache=False, pressure_offset=0., geometry_weight=.5,
                ratio_weight=1., lift_support_weight=1., lift_balance_weight=2.,
                lift_reliability_scale=.02, per_budget={}, target="pressure")
EXTRA_KEYS = {"name", "method", "head_checkpoint", "head_manifest", "amp",
              "status", "selection", "benchmark", "version"}


def read_config(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


def resolve_config(method, config=None):
    values = read_config(config) if isinstance(config, (str, Path)) else dict(config or {})
    if method not in METHODS or values.get("method", method) != method:
        raise ValueError("method and config method must agree")
    unknown = set(values) - set(DEFAULTS) - EXTRA_KEYS
    if unknown:
        raise ValueError(f"unsupported Airfoil config keys: {sorted(unknown)}")
    result = dict(DEFAULTS, method=method, name=method)
    if method == "ca":
        result.update(sampler="patch", feature="gradient_head")
    if method == "pi":
        result.update(amp=8., ell=.35, beta=.5)
    if method == "pi_voronoi":
        result.update(amp=20., ell=.3, beta=1.)
    result.update(values)
    if method.startswith("cpg") or method == "ca":
        coarse_count(11271, result["coarse_fraction"], 121)
    if result["sampler"] not in ("weighted", "weighted_nested", "patch"):
        raise ValueError("unsupported final sampler")
    if method == "ca" and result["sampler"] == "weighted":
        raise ValueError("CA requires a nested final set")
    if result["operator_stage1_cache"] and method != "ca":
        raise ValueError("operator stage1 cache is exposed through CA only")
    if result["feature"] not in ("gradient_head", "gradient_lsq", "gradient_energy", "curvature", "lift_contribution", "ratio_sensitivity"):
        raise ValueError("unsupported Airfoil feature")
    if result["beta"] < 0 or result["kappa"] < 1 or result["ell"] <= 0 or result["uniform_floor"] <= 0 or result["score_gamma"] <= 0:
        raise ValueError("invalid Airfoil sampling parameters")
    if result["patch_count"] < 1 or result["placement"] not in ("random", "fps"):
        raise ValueError("invalid patch configuration")
    if result["target"] not in ("pressure", "cdcl"):
        raise ValueError("target must be pressure or cdcl")
    for key, overrides in result["per_budget"].items():
        if float(key) not in BUDGETS or set(overrides) - {"kappa", "beta"}:
            raise ValueError("per_budget accepts only kappa/beta for official budgets")
        if overrides.get("kappa", 1.) < 1 or overrides.get("beta", 0.) < 0:
            raise ValueError("invalid per-budget kappa/beta")
    return result


def at_budget(config, budget):
    result = dict(config)
    for key, value in config["per_budget"].items():
        if float(key) == budget:
            result.update(value)
    return result
