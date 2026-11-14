"""Observe one frozen Airfoil score graph on CPU without opening CFD truth.

The actual scientific functions are imported from the verified task snapshot.
Python return profiling and torch forward hooks only copy their real outputs.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import os
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                  MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
HERE = Path(__file__).resolve().parent
CURRENT = HERE.parents[3]
SOURCE = CURRENT / "linearno_followup_20261114/standard_benchmarks/airfoil"
sys.path.insert(0, str(SOURCE / "code"))
import numpy as np
import torch
import common
import runtime as original_runtime
from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.inference.cpg.airfoil2d_single_feature import normalize_single_feature, sample_airfoil_cpg_single_feature
from adasolver.inference.cpg.airfoil2d_features import pressure_acquisition, pressure_gradient
from adasolver.inference.pi.airfoil2d import sample_airfoil_pi_voronoi
from adasolver.inference.uniform.airfoil2d import sample_airfoil_uniform
from adasolver.utils.interpolation import index_grid

CASE, BUDGET, SAMPLING_SEED = 1001, .5, 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return value


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    if (HERE / "score_manifest.json").exists():
        raise FileExistsError("Preserve the completed capture; use a new explicit run directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    original_load = np.load
    phase = {"name": "score_capture"}
    file_accesses, denied = [], []

    def guarded_load(path, *args, **kwargs):
        path = Path(path)
        if phase["name"] != "error_only" and path.name == "NACA_Cylinder_Q.npy":
            denied.append({"phase": phase["name"], "kind": "truth_file", "path": path.name})
            raise RuntimeError("CFD truth is unavailable to the score graph")
        file_accesses.append({"phase": phase["name"], "file": path.name})
        return original_load(path, *args, **kwargs)

    class GeometryOnlyData(AirfoilRuntimeData):
        def __init__(self, data_path, operator_manifest):
            self.x = np.load(Path(data_path) / "NACA_Cylinder_X.npy", mmap_mode="r")
            self.y = np.load(Path(data_path) / "NACA_Cylinder_Y.npy", mmap_mode="r")
            metadata = json.loads(Path(operator_manifest).read_text())
            normalizer_path = Path(metadata["normalizer"])
            with np.load(normalizer_path) as z:
                self.mean, self.std = float(z["mean"].item()), float(z["std"].item())
            self.normalizer_path = str(normalizer_path)
            self.normalizer_sha256 = sha(normalizer_path)
            self.mandatory = np.arange(50, 171, dtype=np.int64) * 51

        def target(self, sample_id):
            denied.append({"phase": phase["name"], "kind": "target_method", "case_id": int(sample_id)})
            raise RuntimeError("target() is unavailable to the score graph")

        @property
        def q(self):
            denied.append({"phase": phase["name"], "kind": "q_attribute"})
            raise RuntimeError("q is unavailable to the score graph")

    np.load = guarded_load
    original_runtime.AirfoilRuntimeData = GeometryOnlyData
    rt = original_runtime.Runtime("t3", "cpu")
    assert Path(common.__file__).resolve() == SOURCE / "code/common.py"
    cfgs = {c["name"]: c for c in original_runtime.configurations("t3")}
    coords = rt.data.coords(CASE)
    mandatory = rt.data.mandatory
    n = len(coords)
    assert n == 11271
    phase["name"] = "negative_control"
    for operation in [lambda: rt.data.target(CASE), lambda: rt.data.q,
                      lambda: np.load(common.DATA / "NACA_Cylinder_Q.npy", mmap_mode="r")]:
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Truth isolation negative control failed")
    phase["name"] = "score_capture"
    targets = {
        pressure_gradient.__code__: ("pressure_gradient", ["coords", "pressure", "k"]),
        pressure_acquisition.__code__: ("pressure_acquisition", ["indices", "pressure", "feature", "k"]),
        normalize_single_feature.__code__: ("normalize_single_feature", ["score", "mode", "values"]),
        sample_airfoil_cpg_single_feature.__code__: ("sample_airfoil_cpg_single_feature", ["score", "values", "available", "probability", "base", "mass", "realized_density", "uniform_floor", "score_gamma"]),
        sample_airfoil_pi_voronoi.__code__: ("sample_airfoil_pi_voronoi", ["distance", "target_density", "available", "probabilities", "mass", "realized_density"]),
        sample_airfoil_uniform.__code__: ("sample_airfoil_uniform", ["full_count", "count", "available", "mandatory"]),
    }
    functions = [pressure_gradient, pressure_acquisition, normalize_single_feature,
                 sample_airfoil_cpg_single_feature, sample_airfoil_pi_voronoi, sample_airfoil_uniform]
    source_functions = {f.__name__: {"path": str(Path(f.__code__.co_filename).relative_to(CURRENT)),
                                   "file_sha256": sha(f.__code__.co_filename), "line": f.__code__.co_firstlineno}
                        for f in functions}
    methods, trace_manifest, samples = [], [], {}
    for method in ["uniform", "pi", "cpg", "ca"]:
        rt.clear_dynamic()
        events = []

        def profiler(frame, event, returned):
            if event == "return" and frame.f_code in targets:
                name, keys = targets[frame.f_code]
                values = {key: copy_value(frame.f_locals[key]) for key in keys if key in frame.f_locals}
                if isinstance(returned, (np.ndarray, torch.Tensor)):
                    values["return"] = copy_value(returned)
                events.append((name, values))

        head_arrays, hooks = {}, []
        if method == "ca":
            head = rt.head(cfgs[method])[0]
            hooks.append(head.head.register_forward_pre_hook(lambda module, args: head_arrays.update(ca_features=args[0].detach().cpu().numpy().copy())))
            hooks.append(head.head.layers[0].register_forward_hook(lambda module, args, result: head_arrays.update(ca_bottleneck=result.detach().cpu().numpy().copy())))
            hooks.append(head.head.register_forward_hook(lambda module, args, result: head_arrays.update(ca_standardized_output=result.detach().cpu().numpy().copy())))
        started = time.perf_counter()
        sys.setprofile(profiler)
        try:
            sampling, coarse, _ = rt.sampling(CASE, SAMPLING_SEED, BUDGET, cfgs[method])
        finally:
            sys.setprofile(None)
            for hook in hooks:
                hook.remove()
        elapsed = time.perf_counter()-started
        samples[method] = sampling
        by_name = {name: values for name, values in events}
        raw_score = np.full(n, np.nan)
        normalized = np.full(n, np.nan)
        geometry_prior = np.full(n, np.nan)
        proposal = np.zeros(n)
        sampling_weight = np.full(n, np.nan)
        eligible = np.zeros(n, dtype=bool)
        realized = np.full(n, np.nan)
        represented_mass = np.full(n, np.nan)
        extra_arrays = {}
        if method in ["cpg", "ca"]:
            values = by_name["sample_airfoil_cpg_single_feature"]
            raw_score = values["score"]
            normalized = values["values"]
            available = values["available"]
            proposal[available] = values["probability"]
            sampling_weight[available] = np.power(values["uniform_floor"]+normalized[available], values["score_gamma"])
            realized[sampling.indices] = values["realized_density"]
            represented_mass[sampling.indices] = values["mass"]
            extra_arrays["coarse_raw_gradient_score"] = by_name["pressure_gradient"]["return"]
            extra_arrays["coarse_pressure"] = next(iter(rt.coarse_cache.values()))[0]
            extra_arrays.update(head_arrays)
        elif method == "pi":
            values = by_name["sample_airfoil_pi_voronoi"]
            geometry_prior = values["target_density"]
            available = values["available"]
            proposal[available] = values["probabilities"]
            sampling_weight[available] = geometry_prior[available]
            realized[sampling.indices] = values["realized_density"]
            represented_mass[sampling.indices] = values["mass"]
            extra_arrays["surface_distance"] = values["distance"]
        else:
            values = by_name["sample_airfoil_uniform"]
            assert values["count"] == 5636
            available = values["available"]
            proposal[available] = 1./len(available)
            sampling_weight[available] = 1.
        eligible[available] = True
        selected_mask = np.zeros(n, bool)
        selected_mask[sampling.indices] = True
        coarse_mask = np.zeros(n, bool)
        if coarse is not None:
            coarse_mask[coarse] = True
        mandatory_mask = np.zeros(n, bool)
        mandatory_mask[mandatory] = True
        reported_density = np.full(n, np.nan)
        compensation = np.full(n, np.nan)
        reported_density[sampling.indices] = sampling.sampling_density
        compensation[sampling.indices] = sampling.compensation
        stage = "test_ca" if method == "ca" else "test_preliminary"
        native = SOURCE / "records/t3" / stage / f"case{CASE}_s0_b0.5_{method}.npz"
        with np.load(native) as z:
            native_indices = z["indices"].copy()
            native_coarse = z["coarse_indices"].astype(np.int64)
            native_compensation = z["compensation"].copy()
            native_prediction = z["prediction"].copy()
        indices_match = np.array_equal(sampling.indices, native_indices)
        coarse_match = np.array_equal(np.asarray([] if coarse is None else coarse, np.int64), native_coarse)
        compensation_match = indices_match and np.array_equal(sampling.compensation, native_compensation)
        np.savez_compressed(HERE / f"{method}_score_arrays.npz", coords=coords,
                            indices=sampling.indices, coarse_indices=np.asarray([] if coarse is None else coarse, np.int64),
                            native_indices=native_indices, native_coarse_indices=native_coarse,
                            native_compensation=native_compensation, native_selected_prediction=native_prediction,
                            raw_score=raw_score, normalized_score=normalized, geometry_prior=geometry_prior,
                            proposal_probability=proposal, sampling_weight=sampling_weight,
                            reported_sampling_density=reported_density, realized_density=realized,
                            represented_mass=represented_mass, compensation=compensation,
                            selected=selected_mask, coarse=coarse_mask, mandatory=mandatory_mask, eligible=eligible,
                            **extra_arrays)
        fields = ["point_index", "x", "y", "selected", "coarse", "mandatory", "eligible",
                  "raw_score", "normalized_score", "geometry_prior", "sampling_weight", "proposal_probability",
                  "reported_sampling_density", "realized_density", "represented_mass", "compensation"]
        csv_path = HERE / f"{method}_points.csv"
        with csv_path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for i in range(n):
                values = [i, *coords[i], int(selected_mask[i]), int(coarse_mask[i]), int(mandatory_mask[i]), int(eligible[i]),
                          raw_score[i], normalized[i], geometry_prior[i], sampling_weight[i], proposal[i],
                          reported_density[i], realized[i], represented_mass[i], compensation[i]]
                writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in values])
        methods.append({"method": method, "csv": csv_path.name, "arrays": f"{method}_score_arrays.npz",
                        "columns": {"index": "point_index", "xyz": ["x", "y"], "selected": "selected", "coarse": "coarse",
                                    "raw_score": "raw_score", "normalized_score": "normalized_score", "density": "proposal_probability"},
                        "config": cfgs[method], "selected_count": len(sampling.indices), "coarse_count": int(coarse_mask.sum()),
                        "actual_selected_indices_sha256": array_sha(sampling.indices.astype(np.int64)),
                        "actual_coarse_indices_sha256": array_sha(np.asarray([] if coarse is None else coarse, np.int64)),
                        "native_npz": str(native.relative_to(CURRENT)), "native_npz_sha256": sha(native),
                        "native_selected_indices_sha256": array_sha(native_indices.astype(np.int64)),
                        "native_coarse_indices_sha256": array_sha(native_coarse),
                        "native_indices_exact_match": indices_match, "native_coarse_exact_match": coarse_match,
                        "native_compensation_exact_match": bool(compensation_match),
                        "raw_score_applicable": method in ["cpg", "ca"], "score_capture_wall_seconds": elapsed,
                        "online_generation_cost": None, "cost_status": "PARTIAL: instrumented CPU capture, not native GPU online timing",
                        "sampling_metadata": sampling.metadata, "truth_access": False})
        trace_manifest.append({"method": method, "functions_observed": [name for name, _ in events],
                               "extra_array_shapes": {k: list(np.asarray(v).shape) for k, v in extra_arrays.items()}})
        print(method, "captured; native indices match", indices_match, "coarse", coarse_match, flush=True)
    assert not any(x["phase"] == "score_capture" for x in denied)
    assert not any(x["phase"] == "score_capture" and x["file"] == "NACA_Cylinder_Q.npy" for x in file_accesses)
    write(HERE / "truth_isolation.json", {"score_graph_truth_access": False,
          "negative_control_rejections": denied, "actual_capture_truth_access_attempts": 0,
          "score_capture_file_accesses": file_accesses,
          "separation": "All score CSV/NPZ captures were persisted before the error phase; GeometryOnlyData permanently rejects target() and q"})
    write(HERE / "capture_trace.json", trace_manifest)
    normalizer_path = Path(rt.metadata["normalizer"])
    manifest = {"status": "VERIFIED_COMPLETE" if all(m["native_indices_exact_match"] and m["native_coarse_exact_match"] and m["native_compensation_exact_match"] for m in methods) else "PARTIAL",
                "scope": "one CPU score-graph capture; native selected-set replay QA; no new training or benchmark aggregate",
                "dataset": "airfoil", "backbone": "Transolver3", "training_seed": 0, "head_seed": 0,
                "case_id": CASE, "budget": BUDGET, "sampling_seed": SAMPLING_SEED, "shape": [221, 51],
                "checkpoint_sha256": rt.operator_hash, "normalizer_sha256": rt.data.normalizer_sha256,
                "head_sha256": rt.head(cfgs["ca"])[1], "device": "cpu", "torch": str(torch.__version__),
                "source_functions": source_functions, "source_code": {str(p.relative_to(CURRENT)): sha(p) for p in [SOURCE/"code/common.py", SOURCE/"code/runtime.py", Path(original_runtime.AirfoilEvaluator.sampling.__wrapped__.__code__.co_filename)]},
                "capture_script_sha256": sha(__file__), "truth_access": False, "methods": methods,
                "field_semantics": {"raw_score": {"support": "full", "applicable": ["cpg", "ca"], "normalization": "none", "meaning": "LSQ coarse-pressure gradient norm lifted to all native points by the original nearest-coarse assignment"},
                    "normalized_score": {"support": "full", "applicable": ["cpg", "ca"], "normalization": "q95", "meaning": "max(raw_score,0)/max(q95,1e-12); not clipped"},
                    "geometry_prior": {"support": "full", "applicable": ["pi"], "normalization": "none", "meaning": "1 + amp*exp(-physical surface distance/ell)"},
                    "sampling_weight": {"support": "eligible", "mask": "eligible", "normalization": "none", "meaning": "actual pre-normalization proposal weight; CPG/CA=(floor+normalized_score)^gamma; PI=geometry prior; Uniform=implicit equal weight1"},
                    "proposal_probability": {"support": "eligible", "mask": "eligible", "normalization": "sum1", "meaning": "actual initial rng.choice probability on remaining eligible points; zero on already-forced base; not marginal inclusion probability"},
                    "reported_sampling_density": {"support": "selected", "mask": "selected", "normalization": "none", "meaning": "unmodified SamplingResult.sampling_density; this differs from gamma-powered proposal probability"},
                    "realized_density": {"support": "selected", "mask": "selected", "normalization": "none", "meaning": "mean represented Voronoi mass / represented mass; not a mean-one density over selected samples; Uniform not applicable"},
                    "compensation": {"support": "selected", "mask": "selected", "normalization": "none", "meaning": "actual retained-point measure weights passed to final inference; Uniform ones"}},
                "not_applicable_encoding": "empty CSV values / NaN NPZ values; no fabricated Uniform/PI refinement score",
                "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(HERE / "score_manifest.json", manifest)
    # Ground truth is opened only now, after the truth-free score bundle exists.
    phase["name"] = "error_only"
    truth = np.asarray(np.load(common.DATA / "NACA_Cylinder_Q.npy", mmap_mode="r")[CASE, 3], dtype=np.float32).reshape(-1, 1)
    reconstruction_gate = json.loads((SOURCE / "config/evaluation_gate.json").read_text())
    sys.path.insert(0, str(Path(reconstruction_gate["module"]).parent))
    import reconstruct_api
    assert reconstruct_api.IMPLEMENTATION_SOURCE_SHA256 == reconstruction_gate["module_sha256"]
    error_rows = []
    for item in methods:
        method = item["method"]
        sampling = samples[method]
        if item["native_indices_exact_match"] and item["native_compensation_exact_match"]:
            with np.load(HERE / item["arrays"]) as z:
                prediction = z["native_selected_prediction"].copy()
            prediction_source = "reused verified native GPU selected predictions, exact selected/weight match"
        else:
            pos = torch.as_tensor(coords[sampling.indices])[None]
            rt.context.set(None if method == "uniform" else torch.as_tensor(sampling.compensation, dtype=torch.float32))
            with torch.no_grad():
                prediction = rt.data.decode(rt.model(pos, pos))[0].numpy()
            rt.context.clear()
            prediction_source = "CPU final prediction on captured actual selected set; native mismatch retained"
        complete = reconstruction_gate["old_id"]
        field = reconstruct_api.native_reconstruct("airfoil", sampling.indices, prediction, index_grid((221, 51)))
        error = np.abs(field-truth)
        np.savez_compressed(HERE / f"{method}_error_only.npz", prediction=field, truth=truth, absolute_error=error)
        error_rows.append({"method": method, "field_relative_l2": float(np.linalg.norm(field-truth)/np.linalg.norm(truth)),
                           "prediction_source": prediction_source, "reconstruction_id": complete, "truth_access": True,
                           "artifact": f"{method}_error_only.npz"})
    write(HERE / "error_manifest.json", {"scope": "post-hoc error only; never supplied to the score graph", "truth_access": True,
          "truth_case": CASE, "truth_channel": 3, "methods": error_rows})
    np.load = original_load
    print("Complete score capture and separate error artifacts", HERE, flush=True)


if __name__ == "__main__":
    main()
