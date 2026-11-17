"""Old/new complete-field requests for each independently trained Car LinearNO.

Uses the existing reconstruction CarRequest without changing sampling, the
operator, interpolation, or request boundaries. Seed0 fills PI/CA only.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import asdict

from common import ROOT, E3, WORKING_ROOT, activate_sources, sha256, verify_lock, write_json

RECONSTRUCTION = WORKING_ROOT / "linearno_followup_20261114/reconstruction"


def activate_request_sources():
    activate_sources()
    # inference_speed lives in the active source tree, but its adasolver
    # package must never shadow the frozen E3 scientific implementation.
    prefix = [str(RECONSTRUCTION / "code"), str(E3 / "code"), str(E3 / "code/vendor"),
              str(WORKING_ROOT / "AdaSolver_official")]
    sys.path[:] = prefix + [path for path in sys.path if path not in prefix]


def variant(config):
    method = config["method"]
    return "full" if method == "fullmesh" else (f"cpg_beta{int(config['beta'])}" if method == "cpg" else method)


def configuration_bank(seed):
    name = "seed0_missing_methods.json" if seed == 0 else "evaluation_methods.json"
    rows = json.loads((ROOT / name).read_text())
    bank = {variant(row): row for row in rows}
    if seed:
        bank["full"] = {"method": "fullmesh", "selection_target": "field"}
    return bank


def source_identity():
    protocol = json.loads((ROOT / "timing_protocol.json").read_text())
    for item in protocol["source_files"]:
        if sha256(WORKING_ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Reconstruction timing dependency changed: " + item["path"])
    return protocol


def native_references(seed, runtime, bank, test_ids, split_hash, head_hash):
    """Require every native accuracy row before attaching any timing to it."""
    from adasolver.inference.shapenet_car_contract import method_config
    entries = {}
    files = []
    roots = [ROOT / "records" / f"seed{seed}" / "adaptive"]
    if seed:
        roots.append(ROOT / "records" / f"seed{seed}" / "full")
    for root in roots:
        for manifest in sorted(root.glob("*/run_manifest.json")):
            metadata = json.loads(manifest.read_text())
            config = metadata["config"]
            label = variant(config)
            if label not in bank:
                continue
            if config.get("selection_target") != "field":
                raise ValueError("Do not mix Cd-optimized and field-policy rows")
            expected_config = method_config(bank[label]["method"], **{k: v for k, v in bank[label].items() if k != "method"})
            if config != expected_config or metadata["split_hash"] != split_hash:
                raise ValueError("Native accuracy protocol or split differs from the request lock")
            if metadata["operator_sha256"] != runtime.metadata["checkpoint_sha256"] or metadata["normalizer_sha256"] != runtime.metadata["normalizer_sha256"]:
                raise ValueError("Accuracy reference belongs to another checkpoint or normalizer")
            if label == "ca" and metadata["head_sha256"] != head_hash:
                raise ValueError("CA accuracy was produced by another trained head")
            raw = manifest.parent / "raw.jsonl"
            for line in raw.open():
                row = json.loads(line)
                if row["status"] != "ok" or row["sample_id"] not in test_ids:
                    raise ValueError("Invalid native accuracy row")
                if row["operator_sha256"] != runtime.metadata["checkpoint_sha256"] or row["split_hash"] != split_hash:
                    raise ValueError("Native accuracy row identity differs from its manifest")
                key = (row["sample_id"], label, None if label == "full" else int(row["seed"]), float(row["budget_fraction"]))
                if key in entries:
                    raise ValueError("Duplicate native accuracy identity")
                entries[key] = row
            files.extend([{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in (manifest, raw)])
    expected = {(case, name, sample_seed, budget) for case in test_ids for name in bank
                for sample_seed in ([None] if name == "full" else [0, 1, 2])
                for budget in ([1.] if name == "full" else [.375, .5, .75])}
    if set(entries) != expected:
        raise ValueError(f"Native accuracy matrix incomplete: found {len(entries)}, expected {len(expected)}")
    return entries, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--scope", choices=("A", "B", "C"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    verify_lock()
    protocol = source_identity()
    if not json.loads((ROOT / "timing_preflight.json").read_text())["passed"]:
        raise ValueError("Timing interface preflight has not passed")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    activate_request_sources()
    import numpy as np
    import torch
    from evaluate_linearno import LinearNORuntime
    from linearno_adapter import install_linear_compensation
    from linearno_ca import LinearNOCA
    from car_request import CarRequest
    from reconstruct_api import IMPLEMENTATION_VERSION, IMPLEMENTATION_SOURCE_SHA256
    from timing_helper import measure_paired
    from adasolver.metrics.shapenet_car import dual_field_metrics
    from adasolver.metrics.car_cd import CarCdGeometry
    from inference_speed.gpu_monitor import ExclusiveGPU

    destination = ROOT / "records" / f"seed{args.seed}" / "formal_timing" / args.scope
    destination.mkdir(parents=True, exist_ok=True)
    task_lock = open(destination / "task.lock", "a")
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (destination / "run_manifest.json").exists():
        raise FileExistsError("Retain the existing timing attempt; review before resuming")
    name = f"shapenet_car_linearno_ddp2_e400_trainingseed{args.seed}"
    manifest = (E3 if args.seed == 0 else ROOT) / "checkpoints" / f"seed{args.seed}" / f"{name}_manifest.json"
    runtime = LinearNORuntime(manifest)
    if runtime.metadata["seed"] != args.seed:
        raise ValueError("Timing checkpoint training seed differs")
    splits = runtime.splits(ROOT / "splits" / f"seed{args.seed}.json")
    bank = configuration_bank(args.seed)
    head_manifest = ROOT / "checkpoints" / f"seed{args.seed}" / "head" / f"ca_pressure_linearno_trainingseed{args.seed}_headseed{args.seed}_manifest.json"
    head_meta = json.loads(head_manifest.read_text())
    if head_meta["status"] != "trained_head" or head_meta["training_seed"] != args.seed or head_meta["head_seed"] != args.seed or head_meta["epochs_completed"] != 200:
        raise ValueError("Formal timing requires the completed matching trained CA head")
    references, reference_sources = native_references(args.seed, runtime, bank, set(splits["test"]),
                                                       splits["split_hash"], head_meta["head_sha256"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_hashes = {p.name: sha256(p) for p in ROOT.glob("*.py")}
    state = {"status": "RUNNING", "evidence_status": "pending", "training_seed": args.seed,
             "head_seed": args.seed, "physical_gpu": args.gpu, "scope": args.scope,
             "checkpoint_sha256": runtime.metadata["checkpoint_sha256"],
             "normalizer_sha256": runtime.metadata["normalizer_sha256"],
             "head_sha256": head_meta["head_sha256"], "policy_target": "field",
             "protocol_sha256": sha256(ROOT / "timing_protocol.json"), "source_hashes": source_hashes,
             "reconstruction_version": IMPLEMENTATION_VERSION, "reconstruction_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
             "native_accuracy_sources": reference_sources, "completed_groups": [], "conditions": 0,
             "sampling_seeds": [0, 1, 2], "budgets": [.375, .5, .75], "warmups": 20, "formal_repeats": 100,
             "pid": os.getpid(), "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(destination / "run_manifest.json", state)

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    try:
        # This script owns GPU admission directly. The root queue holds only
        # its queue lock; no parent process holds this profile lock.
        with ExclusiveGPU(args.gpu, destination / "gpu_telemetry.json", interval=1.) as monitor:
            startup = time.perf_counter()
            model = runtime.load_operator("cuda:0")
            context, _ = install_linear_compensation(model)
            head = LinearNOCA(model, head_manifest, runtime.binding(), "cuda:0")
            torch.cuda.synchronize()
            write_json(destination / "model_preparation.json", {"model_head_load_ms": (time.perf_counter()-startup)*1000,
                "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "fp32": True, "allow_tf32": False, "cpu_affinity": sorted(os.sched_getaffinity(0))})
            for case_index, case in enumerate(protocol["timing_cases"]):
                if case not in splits["test"]:
                    raise ValueError("Timing case is outside the locked test population")
                item = runtime.sample(case)
                truth, mask = item.target_phys.numpy(), item.surface_mask.numpy()
                geometry = CarCdGeometry.load(runtime.data_path / "training_data" / case / "quadpress_smpl.vtk",
                                              ROOT / "records/geometry_cache")
                gt_cd = float(geometry.compute(truth[mask, 3], truth[mask, :3]))
                groups = ([('full', None, 1.)] if "full" in bank else []) + [
                    (f"s{s}_b{b}", s, b) for s in (0, 1, 2) for b in (.375, .5, .75)]
                for tag, sample_seed, budget in groups:
                    key = f"case{case_index}_{tag}"
                    directory = destination / key
                    directory.mkdir()
                    names = ["full"] if tag == "full" else [name for name in bank if name != "full"]
                    calls, labels, preparation = {}, {}, []
                    for label in names:
                        request = CarRequest(model, context, item, runtime.output_normalizer, bank[label], budget,
                                             sample_seed, ca=head, device="cuda:0")
                        resident_ms = request.prepare_scope(args.scope)
                        preparation.append({"variant": label, "resident_input_h2d_ms": resident_ms,
                                            "sampling_geometry": "prepared inside each timed request"})
                        for implementation in (["shared"] if label == "full" else ["old", "new"]):
                            condition = label + ":" + implementation
                            calls[condition] = lambda r=request, impl=implementation: r("old" if impl == "shared" else impl, args.scope)
                            labels[condition] = {"variant": label, "implementation": implementation, "request": request}
                    original, original_metrics, identities, checks = {}, {}, {}, []

                    def check_output(condition, output):
                        entry = labels[condition]
                        request = entry["request"]
                        array = output.cpu().numpy() if isinstance(output, torch.Tensor) else output
                        if array.shape != (len(item), 4) or array.dtype != np.float64 or not np.isfinite(array).all():
                            raise ValueError("Complete field shape/dtype/finite contract failed")
                        if not np.all(array[~mask, 3] == 0):
                            raise ValueError("Pressure container must be zero outside the declared surface")
                        selected = digest(np.asarray(request.last_indices, np.int64))
                        coarse = None if request.last_coarse_values is None else digest(request.last_coarse_values)
                        reference = references[(case, entry["variant"], sample_seed, budget)]
                        if selected != reference["selected_indices_sha256"]:
                            raise ValueError("Timed request selected indices differ from native accuracy")
                        values = asdict(dual_field_metrics(array, truth, mask))
                        cd = float(geometry.compute(array[mask, 3], array[mask, :3]))
                        values["cd_relative_error"] = abs(cd-gt_cd)/max(abs(gt_cd), 1e-12)
                        deltas = {k: abs(values[k]-reference[k]) for k in ("field_error", "velocity_rel_l2", "pressure_rel_l2", "cd_relative_error")}
                        if max(deltas.values()) > 2e-6:
                            raise ValueError("Native accuracy/request metric alignment failed: " + str(deltas))
                        parity = None
                        if entry["implementation"] == "new":
                            old = original[entry["variant"]]
                            error = np.abs(array-old)
                            passed = bool(np.all(error <= 1e-6 + 1e-5*np.abs(old)))
                            parity = {"pass": passed, "atol": 1e-6, "rtol": 1e-5,
                                      "max_absolute_difference": float(error.max()),
                                      "relative_norm_difference": float(np.linalg.norm(array-old)/max(np.linalg.norm(old), 1e-30)),
                                      "downstream_metric_differences": {k: abs(values[k]-original_metrics[entry["variant"]][k]) for k in values}}
                            if not passed:
                                write_json(directory / "parity_failure.json", {"condition": condition, **parity})
                                raise ValueError("Old/new elementwise parity failed at the fixed tolerance")
                            if identities[entry["variant"]] != (selected, coarse):
                                raise ValueError("Old/new requests changed coarse prediction or selected indices")
                        else:
                            original[entry["variant"]] = array.copy()
                            original_metrics[entry["variant"]] = values
                            identities[entry["variant"]] = selected, coarse
                        row = {"condition": condition, "selected_indices_sha256": selected,
                               "coarse_prediction_sha256": coarse, "metrics": values,
                               "native_accuracy_metric_differences": deltas, "old_new_parity": parity}
                        checks.append(row)
                        return row

                    def progress(kind, done, total):
                        write_json(destination / "heartbeat.json", {"group": key, "kind": kind, "done": done,
                                                                   "total": total, "pid": os.getpid()})

                    torch.cuda.reset_peak_memory_stats()
                    measured = measure_paired(calls, scope=args.scope, warmups=20, repeats=100,
                                              order_seed=20261114+args.seed*10000+case_index*1000+int(budget*1000)+(sample_seed or 0),
                                              check_output=check_output, guard=monitor.assert_exclusive, progress=progress)
                    for label in names:
                        request = labels[next(c for c in labels if labels[c]["variant"] == label)]["request"]
                        final_identity = (digest(np.asarray(request.last_indices, np.int64)),
                                          None if request.last_coarse_values is None else digest(request.last_coarse_values))
                        if final_identity != identities[label]:
                            raise ValueError("Repeated request selected set or coarse prediction drifted")
                    if "cpg_beta0" in names:
                        if identities["cpg_beta0"] != identities["cpg_beta1"]:
                            raise ValueError("CPG beta pair changed selected indices or coarse prediction")
                    for filename, records in (("timing_repeats.csv", measured["raw"]), ("summary.csv", measured["summary"])):
                        with (directory / filename).open("x", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                            writer.writeheader()
                            writer.writerows(records)
                    write_json(directory / "alignment.json", {"passed": True, "checks": checks})
                    write_json(directory / "first_requests.json", measured["first_requests"])
                    write_json(directory / "order.json", measured["order"])
                    write_json(directory / "preparation.json", preparation)
                    write_json(directory / "group_memory.json", {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "scope": "group process peak with this backbone, matching CA head, and all compared requests/maps resident; not an isolated method memory estimate"})
                    profiles, cache = [], []
                    for condition, info in labels.items():
                        request = info["request"]
                        impl = "old" if info["implementation"] == "shared" else info["implementation"]
                        if args.scope == "C":
                            request.clear_caches()
                        for repeat in range(3):
                            _, stages, map_info = request.profile(impl, args.scope)
                            profiles.extend({"condition": condition, "profile_repeat": repeat, **row} for row in stages)
                            if map_info:
                                cache.append({"condition": condition, "profile_repeat": repeat, "cache_hit": request.last_cache_hit, **map_info})
                    write_json(directory / "profiles.json", profiles)
                    write_json(directory / "cache.json", cache)
                    write_json(directory / "complete.json", {"execution_status": "completed", "evidence_status": "pending_independent_qa",
                        "training_seed": args.seed, "case_id": case, "sampling_seed": sample_seed, "budget": budget,
                        "scope": args.scope, "conditions": len(calls), "warmups": 20, "formal_repeats": 100})
                    state["completed_groups"].append(key)
                    state["conditions"] += len(calls)
                    write_json(destination / "run_manifest.json", state)
                    print(args.seed, args.scope, key, "completed", flush=True)
            expected_groups = 27 if args.seed == 0 else 30
            expected_conditions = 108 if args.seed == 0 else 273
            if len(state["completed_groups"]) != expected_groups or state["conditions"] != expected_conditions:
                raise ValueError("Formal timing coverage mismatch")
        state.update(status="COMPLETED_EXECUTION_PENDING_QA", evidence_status="pending_independent_qa",
                     finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(destination / "run_manifest.json", state)
    except Exception as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        write_json(destination / "run_manifest.json", state)
        raise


if __name__ == "__main__":
    main()
