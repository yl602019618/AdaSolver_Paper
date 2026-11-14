"""Use the existing compiled FPS loop after checking complete point-set equality."""
import os
from common import ROOT, read, write
import numpy as np
from scipy.spatial import cKDTree
from adasolver.inference import adaptive, adaptive_fast, elasticity_density
from adasolver.inference.native_fps import native_fps

ORIGINAL = adaptive.fps_within
INSTALLED = False


def sequential(candidates, coords, already, take, seed=0, distance_backend="numpy"):
    return ORIGINAL(candidates, coords, already, take, seed, distance_backend="native")


def independent(candidates, coords, anchors, take, rng):
    if take <= 0 or len(candidates) == 0:
        return np.empty(0, np.int64)
    points = np.asarray(coords[candidates], dtype=np.float64)
    nearest = cKDTree(coords[anchors]).query(points)[0] if len(anchors) else np.full(len(points), np.inf)
    first = 0 if len(anchors) else int(rng.integers(len(points)))
    selected = native_fps(points, nearest, min(take, len(points)), first_if_infinite=first)
    return candidates[selected]


def install(check_status=True):
    global INSTALLED
    if INSTALLED:
        return
    config = read(ROOT / "config/sampler_backend.json")
    if config["backend"] != "native_fps":
        return
    if check_status and config.get("status") != "verified":
        raise RuntimeError("The native sampler has not passed its reference comparison")
    os.environ["ADASOLVER_NATIVE_CACHE"] = str(ROOT / "runtime/native_fps")
    adaptive.fps_within = sequential
    elasticity_density.fps_within = sequential
    adaptive_fast._fps_fast = independent
    INSTALLED = True


def verify():
    policies = read(ROOT / "config/frozen_methods.json")
    calls = []
    for benchmark in ("elasticity", "pipe", "darcy"):
        if benchmark == "elasticity":
            coords = np.load(ROOT / "data/elasticity/test_pos.npy", mmap_mode="r")[0].astype(np.float64)
        else:
            coords = np.load(ROOT / "data" / benchmark / "reference.npy").astype(np.float64)
        count = round(len(coords) * .375)
        coarse = np.random.default_rng(4).choice(len(coords), round(len(coords) * .1), replace=False)
        scores = np.random.default_rng(5).lognormal(size=len(coarse))
        if benchmark == "elasticity":
            geometry = elasticity_density.ElasticityDensity(coords)
            for method in ("cpg", "ca"):
                cfg = elasticity_density.DensityConfig(**policies[benchmark][method]["density"])
                def call(geometry=geometry, cfg=cfg, coarse=coarse, scores=scores, count=count):
                    return geometry.refine(coarse, scores, count, cfg, np.random.default_rng(6)).indices
                calls.append((benchmark + "_" + method, call, call()))
        elif benchmark == "pipe":
            labels, mass = adaptive.build_patches(coords, 24, seed=0)
            cache = adaptive_fast.build_patch_cache(coords, labels, 24)
            def call(coords=coords, coarse=coarse, scores=scores, labels=labels, mass=mass, cache=cache, count=count):
                return adaptive_fast.refine_once(coarse, scores, coords, labels, mass, count,
                    kappa=4., seed=6, patch_cache=cache, placement="fps")[0]
            calls.append((benchmark, call, call()))
        else:
            labels, mass = adaptive.build_patches(coords, 32, seed=0)
            def call(coords=coords, coarse=coarse, scores=scores, labels=labels, mass=mass, count=count):
                return adaptive.refine_once(coarse, scores, coords, labels, mass, count, kappa=2., seed=6)[0]
            calls.append((benchmark, call, call()))
    write(ROOT / "config/sampler_backend.json", {"backend": "native_fps", "status": "checking"})
    install(check_status=False)
    checks = []
    for name, call, expected in calls:
        observed = call()
        assert np.array_equal(expected, observed), name
        checks.append({"sampler": name, "selected_points": len(observed), "indices_equal": True})
    write(ROOT / "internal/sampler_equivalence.json", {
        "purpose": "implementation check only; not accuracy evidence", "checks": checks,
        "status": "passed"})
    write(ROOT / "config/sampler_backend.json", {"backend": "native_fps", "status": "verified",
        "verification": "internal/sampler_equivalence.json"})
    print(checks)


if __name__ == "__main__":
    verify()
