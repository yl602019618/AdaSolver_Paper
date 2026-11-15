"""Read-only source audit and public scalar export of completed Darcy seed0 runs."""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import re
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
SOURCE = PROJECT / "linearno_followup_20261114/standard_benchmarks/darcy"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        value = value.replace(str(SOURCE), "local:darcy").replace(str(PROJECT), "local:workspace")
        if re.search(r"/(mnt|Users|root)/", value):
            return "external:" + Path(value).name
    return value


def json_write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-10):
        raise AssertionError((label, actual, expected))


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    sources = {}
    checks = []

    def register(path, kind="source", expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            assert digest == expected, (path, digest, expected)
        key = clean(str(path))
        sources[key] = {"path": key, "bytes": path.stat().st_size, "sha256": digest, "kind": kind}
        return digest

    def read(path):
        register(path)
        return json.loads(Path(path).read_text())

    lock = read(SOURCE / "config/evaluation_lock.json")
    protocol = read(SOURCE / "config/protocol_lock.json")
    accuracy_receipt = read(SOURCE / "results/accuracy_completion.json")
    latency_receipt = read(SOURCE / "results/latency_completion.json")
    assert accuracy_receipt["execution_status"] == latency_receipt["execution_status"] == "completed"
    assert len(lock["test_case_ids"]) == 200 and len(lock["timing_case_ids"]) == 3
    test_ids = set(lock["test_case_ids"])
    reconstruction_ids = [lock["old_reconstruction_id"], lock["new_reconstruction_id"]]
    models, records, histories = [], [], []
    heads = {}
    label_by_key = {"transolver3": "Transolver3", "linearno": "LinearNO_point_temp", "linearno_conv": "LinearNO_conv_temp"}
    for key, label in label_by_key.items():
        manifest_path = SOURCE / "checkpoints" / key / "manifest.json"
        meta = read(manifest_path)
        assert meta["execution_status"] == "completed" and meta["epochs_completed"] == 1000
        assert meta["total_optimization_steps"] == 225000 and meta["seed"] == 0
        assert len(meta["train_ids"]) == 900 and len(meta["validation_ids"]) == 100
        assert not set(meta["train_ids"]) & set(meta["validation_ids"])
        assert not (set(meta["train_ids"]) | set(meta["validation_ids"])) & test_ids
        assert meta["test_used_for_selection"] is False
        register(meta["checkpoint"], "external backbone weights", meta["checkpoint_sha256"])
        register(meta["normalizer"], "external frozen normalizer", meta["normalizer_sha256"])
        assert meta["normalizer_sha256"] == "5a2b073849d5b7d2e968f05f1492737a7646ccbd78baaacc3d662db6aca36251"
        completion = read(SOURCE / "records" / key / "complete_completion.json")
        run = read(SOURCE / "records" / key / "run_manifest.json")
        assert run["execution_status"] == completion["execution_status"] == "completed"
        assert completion["model_specific_validation_accepted"] is True
        rows_path = SOURCE / "records" / key / "metrics_complete.csv"
        register(rows_path, "raw scalar accuracy records", completion["metrics_csv_sha256"])
        rows = csv_rows(rows_path)
        assert len(rows) == (400 if key == "linearno_conv" else 18400)
        actual = set()
        for row in rows:
            identity = (int(row["case_id"]), row["method"], float(row["budget_fraction"]), row["sampling_seed"], row["reconstruction_id"])
            assert identity not in actual
            actual.add(identity)
            assert row["status"] == "ok" and math.isfinite(float(row["value"]))
            assert row["training_seed"] == "0" and row["checkpoint_id"] == meta["checkpoint_sha256"]
            assert row["normalizer_id"] == meta["normalizer_sha256"]
            assert row["test_population"] == "historical_smooth2_200"
        expected = {(case, "full", 1., "", rid) for case in test_ids for rid in reconstruction_ids}
        if key != "linearno_conv":
            expected |= {(case, method, budget, str(seed), rid) for case in test_ids
                         for method in ["uniform", "pi", "cpg_beta0", "cpg", "ca"]
                         for budget in [.375, .5, .75] for seed in [0, 1, 2] for rid in reconstruction_ids}
        assert actual == expected
        row = {"dataset": "darcy", "backbone": label, "model_class": meta["model_class"],
               "model_variant": meta["variant"], "training_seed": 0, "training_cases": 900,
               "validation_cases": 100, "test_cases": 200, "epochs": 1000, "updates": 225000,
               "nmin_fraction": meta["nmin_fraction"], "parameters": meta["parameter_count"],
               "checkpoint_id": meta["checkpoint_sha256"], "normalizer_id": meta["normalizer_sha256"],
               "best_validation_relative_l2": meta["best_validation_relative_l2"], "best_epoch": meta["best_epoch"],
               "training_process_wall_seconds": meta["training_wall_seconds"], "gpu_busy_seconds": None,
               "cost_scope": "recorded training wall time; actual GPU busy time not measured",
               "official_source_commit": meta["official_source_commit"], "normalizer_fit": "900 training IDs only",
               "checkpoint_selection": "minimum validation physical field relative L2",
               "test_history": "historical smooth2 test visibility retained; current epoch selection uses validation only",
               "backbone_status": "VERIFIED_COMPLETE", "head_applicable": key != "linearno_conv",
               "accuracy_status": "VERIFIED_COMPLETE", "timing_status": "VERIFIED_COMPLETE",
               "independent_training_count": 1, "checkpoint_external_id": clean(meta["checkpoint"])}
        if key != "linearno_conv":
            head_path = SOURCE / "checkpoints" / key / "head/manifest.json"
            head = read(head_path)
            assert head["execution_status"] == "completed" and head["epochs_completed"] == 1000
            assert head["operator_sha256"] == meta["checkpoint_sha256"] and head["normalizer_sha256"] == meta["normalizer_sha256"]
            assert head["head_seed"] == 0 and head["test_used_for_selection"] is False
            assert set(head["training_ids"]) == set(meta["train_ids"])
            assert set(head["validation_ids"]) == set(meta["validation_ids"])
            register(head["checkpoint"], "external matching head weights", head["checkpoint_sha256"])
            register(head["bank_manifest"], "head-bank provenance", head["bank_manifest_sha256"])
            row.update(head_seed=0, head_class=head["head_architecture"], head_epochs=1000,
                       head_parameters=head["parameter_count"], head_checkpoint_id=head["checkpoint_sha256"],
                       head_best_epoch=head["best_epoch"], head_best_validation=head["best_validation"],
                       head_training_wall_seconds=head["training_seconds"], head_selection=head["selection"],
                       head_status="VERIFIED_COMPLETE")
            heads[label] = head["checkpoint_sha256"]
            pairs = read(SOURCE / "records" / key / "cpg_pair_validation.json")
            assert len(pairs) == 1800 and all(p["identical_coarse"] and p["identical_final"] for p in pairs)
        else:
            row.update(head_status="not_applicable_full_only", head_class="", head_checkpoint_id="",
                       adaptive_scope="native convolution rejects sparse input; Full-only supplement")
        models.append(row)
        records.extend(rows)
        for stage in ["backbone"] + ([] if key == "linearno_conv" else ["head"]):
            history_path = SOURCE / "logs" / (key + ("_head" if stage == "head" else "") + "_history.jsonl")
            register(history_path, "scalar training history")
            history = [json.loads(line) for line in history_path.open()]
            assert len(history) == 1000 and [h["epoch"] for h in history] == list(range(1, 1001))
            histories.extend({"dataset": "darcy", "backbone": label, "training_seed": 0,
                              "stage": stage, **h} for h in history)
        json_write(HERE / f"{key}_training_metadata.json", {"backbone": meta,
                   "head": read(SOURCE / "checkpoints" / key / "head/manifest.json") if key != "linearno_conv" else None})
    assert len(records) == 37200
    checks.append({"check": "model_head_hashes_splits_and_history", "passed": True,
                   "backbones": 3, "matching_heads": 2, "history_rows": len(histories)})

    # Recompute every budget summary and seed-specific trapezoidal ratio.
    grouped = collections.defaultdict(list)
    physical = collections.defaultdict(list)
    for row in records:
        grouped[(row["backbone"], row["reconstruction_id"], row["method"], float(row["budget_fraction"]))].append(row)
        physical[(row["backbone"], row["case_id"], row["sampling_seed"], row["method"], row["beta"], row["budget_fraction"])].append(row)
    assert len(physical) == 18600
    budget_source = csv_rows(SOURCE / "results/backbone_absolute_errors.csv")
    register(SOURCE / "results/backbone_absolute_errors.csv", "owner budget summary")
    max_budget_delta = 0.
    sampling_rows = []
    for row in budget_source:
        key = (row["backbone"], row["reconstruction_id"], row["method"], float(row["budget_fraction"]))
        source_rows = grouped[key]
        value = float(np.mean([float(r["value"]) for r in source_rows]))
        close(value, row["mean_error"], key)
        max_budget_delta = max(max_budget_delta, abs(value-float(row["mean_error"])))
        seeds = sorted({r["sampling_seed"] for r in source_rows if r["sampling_seed"] != ""})
        means = []
        for seed in seeds:
            errors = [float(r["value"]) for r in source_rows if r["sampling_seed"] == seed]
            assert len(errors) == 200
            means.append(float(np.mean(errors)))
            sampling_rows.append({"dataset": "darcy", "backbone": key[0], "reconstruction_id": key[1],
                "method": key[2], "budget_fraction": key[3], "beta": row["beta"], "training_seed": 0,
                "sampling_seed": int(seed), "case_count": 200, "mean_error": means[-1],
                "checkpoint_id": row["checkpoint_id"], "normalizer_id": row["normalizer_id"]})
        if means:
            close(np.std(means, ddof=1), row["sampling_sd"], (key, "sampling SD"))
        full = grouped[(key[0], key[1], "full", 1.)]
        close(np.mean([float(r["value"]) for r in full]), row["same_checkpoint_full_error"], (key, "Full"))
    ratios = csv_rows(SOURCE / "results/backbone_rauec.csv")
    register(SOURCE / "results/backbone_rauec.csv", "owner rAUEC summary")
    ratio_details = []
    max_ratio_delta = 0.
    grid = np.array([.375, .5, .75])
    for row in ratios:
        seed_ratios = []
        for seed in ["0", "1", "2"]:
            curves = []
            for method in [row["method"], "uniform"]:
                curves.append([np.mean([float(r["value"]) for r in grouped[(row["backbone"], row["reconstruction_id"], method, float(b))]
                                        if r["sampling_seed"] == seed]) for b in grid])
            area, denominator = np.trapezoid(curves[0], grid), np.trapezoid(curves[1], grid)
            assert denominator > 0
            ratio = float(area/denominator)
            seed_ratios.append(ratio)
            close(ratio, row["seed"+seed], "seed rAUEC")
            ratio_details.append({"dataset": "darcy", "backbone": row["backbone"], "method": row["method"],
                 "beta": row["beta"], "reconstruction_id": row["reconstruction_id"], "training_seed": 0,
                 "sampling_seed": seed, "checkpoint_id": row["checkpoint_id"], "method_area": float(area),
                 "matched_uniform_area": float(denominator), "rAUEC": ratio, "budget_grid": "0.375;0.5;0.75"})
        close(np.mean(seed_ratios), row["mean"], "mean rAUEC")
        close(np.std(seed_ratios, ddof=1), row["sampling_sd"], "rAUEC sampling SD")
        max_ratio_delta = max(max_ratio_delta, abs(float(np.mean(seed_ratios))-float(row["mean"])))
    parity = []
    cpg_pairs = 0
    physical_lookup = {}
    for key, pair in physical.items():
        assert len(pair) == 2
        rows = {r["reconstruction_id"]: r for r in pair}
        old, new = rows[reconstruction_ids[0]], rows[reconstruction_ids[1]]
        for field in ["selected_indices_hash", "coarse_indices_hash", "coarse_prediction_hash", "compensation_weights_hash"]:
            assert old[field] == new[field]
        metric_delta = abs(float(old["value"])-float(new["value"]))
        assert metric_delta <= 1e-7 + 1e-5*abs(float(old["value"]))
        assert float(old["old_new_max_abs"]) <= 1e-6
        parity.append({"dataset": "darcy", "backbone": key[0], "case_id": key[1], "training_seed": 0,
            "sampling_seed": key[2], "method": key[3], "beta": key[4], "budget_fraction": key[5],
            "checkpoint_id": old["checkpoint_id"], "old_reconstruction_id": reconstruction_ids[0],
            "new_reconstruction_id": reconstruction_ids[1], "selected_indices_hash": old["selected_indices_hash"],
            "coarse_indices_hash": old["coarse_indices_hash"], "coarse_prediction_hash": old["coarse_prediction_hash"],
            "max_absolute_difference": old["old_new_max_abs"], "relative_difference": old["old_new_relative_difference"],
            "metric_difference": metric_delta, "atol": 1e-6, "rtol": 1e-5,
            "source_scope": "recorded evaluator parity, checkpoint/hash-bound raw scalar replay", "pass": True})
        physical_lookup[(key[0], key[1], key[2], key[3], key[5])] = old
    for key, row in physical_lookup.items():
        if key[3] != "cpg":
            continue
        zero = physical_lookup[(key[0], key[1], key[2], "cpg_beta0", key[4])]
        assert all(row[k] == zero[k] for k in ["selected_indices_hash", "coarse_indices_hash", "coarse_prediction_hash"])
        cpg_pairs += 1
    assert cpg_pairs == 3600
    checks.append({"check": "accuracy_complete_matrix_and_independent_numeric_replay", "passed": True,
                   "raw_metric_rows": 37200, "physical_requests": 18600, "budget_rows": len(budget_source),
                   "rAUEC_rows": len(ratios), "CPG_beta_pairs": cpg_pairs,
                   "max_budget_mean_delta": max_budget_delta, "max_rAUEC_mean_delta": max_ratio_delta})

    # Read the original per-group timing records, including all warmups, and
    # compare every scalar summary rather than accepting a controller flag.
    owner_conditions = csv_rows(SOURCE / "results/backbone_latency_summary.csv")
    register(SOURCE / "results/backbone_latency_summary.csv", "owner condition summary")
    owner_lookup = {(r["group_id"], r["condition"]): r for r in owner_conditions}
    assert len(owner_lookup) == 1323
    repetition_rows, profile_rows, first_requests, condition_rows = [], [], [], []
    maximum_timing_delta = 0.
    group_inputs = latency_receipt["group_inputs"]
    assert len(group_inputs) == 333
    for path_string, expected_hash in group_inputs.items():
        path = Path(path_string)
        register(path, "original timing group JSON", expected_hash)
        group = json.loads(path.read_text())
        assert group["execution_status"] == group["profile_execution_status"] == "completed"
        by_condition = collections.defaultdict(list)
        for row in group["raw"]:
            assert math.isfinite(row["total_ms"]) and row["total_ms"] > 0
            by_condition[row["condition"]].append(row)
            repetition_rows.append({"group_id": group["group_id"], **row})
        first_lookup = {r["condition"]: r for r in group["first_requests"]}
        identities = collections.defaultdict(set)
        for row in group["identities"]:
            identities[row["condition"]].add((row["selected_indices_hash"], row["coarse_indices_hash"], row["compensation_weights_hash"]))
        for summary in group["summary"]:
            name = summary["condition"]
            metadata = group["metadata"][name]
            raw = by_condition[name]
            formal = [r for r in raw if r["kind"] == "formal"]
            warmup = [r for r in raw if r["kind"] == "warmup"]
            assert len(formal) == 100 and len(warmup) == 20
            assert {r["repeat_id"] for r in formal} == set(range(100))
            assert {r["repeat_id"] for r in warmup} == set(range(20))
            assert len(identities[name]) == 1
            first = first_lookup[name]
            accepted = first["output_check"]
            assert identities[name] == {(accepted["selected_indices_hash"], accepted["coarse_indices_hash"], accepted["compensation_weights_hash"])}
            assert accepted["full_field_max_absolute_difference"] <= 1e-6
            close(accepted["field_error"], accepted["reference_field_error"], "timed reference field error")
            assert metadata["training_seed"] == 0 and metadata["output_dtype"] == "float64"
            if metadata["method"] == "ca":
                assert metadata["head_checkpoint_id"] == heads[metadata["backbone"]]
            owner = owner_lookup[(group["group_id"], name)]
            values = np.array([r["total_ms"] for r in formal])
            for ours, key in [(np.median(values), "direct_total_ms"), (np.percentile(values, 95), "p95_ms"),
                              (values.mean(), "mean_ms"), (values.std(ddof=1), "sample_sd_ms")]:
                close(ours, owner[key], (group["group_id"], name, key))
                maximum_timing_delta = max(maximum_timing_delta, abs(float(ours)-float(owner[key])))
            condition_rows.append(owner)
            first_requests.append({"group_id": group["group_id"], "condition": name,
                                   "first_request_ms": first["first_request_ms"], **metadata, **accepted})
        for name, rows in group["profiles"].items():
            profile_rows.extend({"group_id": group["group_id"], "condition": name, **row} for row in rows)
    assert len(condition_rows) == 1323
    assert sum(r["kind"] == "formal" for r in repetition_rows) == 132300
    assert sum(r["kind"] == "warmup" for r in repetition_rows) == 26460
    telemetry = read(SOURCE / "timing/gpu_telemetry.json")
    assert telemetry["exclusive"] is True and not telemetry["errors"] and telemetry["physical_gpu"] == 0
    memory_rows = []
    for path_string, expected_hash in latency_receipt["memory_inputs"].items():
        path = Path(path_string)
        register(path, "independent method memory process", expected_hash)
        memory = json.loads(path.read_text())
        assert memory["execution_status"] == "completed"
        memory_rows.extend(memory["rows"])
        monitor_path = path.with_name(path.stem + "_gpu.json")
        monitor = read(monitor_path)
        assert monitor["exclusive"] is True and not monitor["errors"]
    assert len(memory_rows) == 1323
    checks.append({"check": "direct_request_timing_and_memory_replay", "passed": True,
                   "group_json_hashes": 333, "conditions": 1323, "formal_repetitions": 132300,
                   "warmups": 26460, "independent_memory_processes": 11, "memory_conditions": 1323,
                   "maximum_summary_difference_ms": maximum_timing_delta, "gpu_exclusive": True})

    # Compact public raw timing data reference the condition registry instead
    # of repeating identities and private paths on every scalar observation.
    csv_write(HERE / "checkpoint_registry.csv", models)
    csv_write(HERE / "training_history.csv", histories)
    csv_write(HERE / "metrics_records.csv", records)
    csv_write(HERE / "backbone_absolute_errors.csv", budget_source)
    csv_write(HERE / "backbone_rauec.csv", ratios)
    csv_write(HERE / "sampling_variability.csv", sampling_rows)
    csv_write(HERE / "rauec_seed_details.csv", ratio_details)
    csv_write(HERE / "reconstruction_parity.csv", parity)
    csv_write(HERE / "backbone_latency_summary.csv", condition_rows)
    csv_write(HERE / "latency_raw_repeats.csv", repetition_rows)
    csv_write(HERE / "latency_profiles.csv", profile_rows)
    csv_write(HERE / "latency_first_requests.csv", first_requests)
    csv_write(HERE / "memory_conditions.csv", memory_rows)
    for name in ["reconstruction_summary.csv", "latency_reuse_costs.csv"]:
        register(SOURCE / "results" / name, "derived cache/speedup view")
        csv_write(HERE / name, csv_rows(SOURCE / "results" / name))
    timing_groups = collections.defaultdict(list)
    for row in condition_rows:
        timing_groups[(row["backbone"], row["method"], row["budget_fraction"], row["scope"], row["implementation"])].append(row)
    overview = []
    for key, rows in timing_groups.items():
        overview.append({"dataset": "darcy", "backbone": key[0], "method": key[1], "budget_fraction": key[2],
                         "scope": key[3], "implementation": key[4], "training_seed": 0,
                         "checkpoint_id": rows[0]["checkpoint_id"], "case_count": len({r["case_id"] for r in rows}),
                         "condition_count": len(rows), "formal_repeats_per_condition": 100,
                         "median_condition_median_ms": float(np.median([float(r["direct_total_ms"]) for r in rows])),
                         "median_condition_p95_ms": float(np.median([float(r["p95_ms"]) for r in rows])),
                         "aggregation_scope": "median over case/sampling condition statistics; not pooled request quantiles"})
    csv_write(HERE / "latency_overview.csv", overview)
    csv_write(HERE / "source_record_index.csv", list(sources.values()))
    audit = {"status": "VERIFIED_COMPLETE", "scope": "completed matched Darcy training-seed0 cohort only",
             "audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpu_work_run_by_auditor": False,
             "existing_source_modified": False, "checks": checks,
             "monitor_status_resolution": "status_snapshot final_accuracy flag is not the Darcy gate. Actual keys are records/*/complete_completion.json:model_specific_validation_accepted and results/latency_completion.json exact coverage. Independent aggregation is supplied here without changing the original pending strings.",
             "sampling_seed_not_training_seed": True, "independent_training_count_per_model": 1,
             "pending": ["training seeds1/2 and across-training-seed mean/sampleSD", "full requested multi-dataset follow-up"],
             "not_applicable": ["CA and adaptive sampling for native LinearNO convolution Full-only supplement"],
             "evidence_limits": ["public replay recomputes scalar statistics and checks source hashes/identities; it does not rerun networks or reconstruct every stored field", "memory is CUDA allocator peaks plus post-request NVML samples; CPU allocations and transient external-library peaks are outside this measure", "instrumented stage times are profiling only; cached-scene cost estimates are labelled and not independently timed contiguous sequences"],
             "test_history": "smooth2 200-case population has historical test visibility; not a newly sealed holdout",
             "source_accuracy_evidence_status": accuracy_receipt["evidence_status"], "source_files": list(sources.values())}
    json_write(HERE / "source_audit.json", audit)
    scope_rows = [
        {"test": "backbone training", "status": "VERIFIED_COMPLETE", "coverage": "3 model variants x1000epochs/225000updates; training seed0"},
        {"test": "matching CA heads", "status": "VERIFIED_COMPLETE", "coverage": "2 own-feature heads x1000epochs; conv head not applicable"},
        {"test": "Full and adaptive accuracy", "status": "VERIFIED_COMPLETE", "coverage": "200 cases;9200 requests per point model plus200 convFull;37200 old/new scalar rows"},
        {"test": "CPG beta0/1", "status": "VERIFIED_COMPLETE", "coverage": "3600 same-coarse/same-selected/same-coarse-prediction pairs"},
        {"test": "reconstruction parity", "status": "VERIFIED_COMPLETE", "coverage": "18600 recorded fixed-prediction/point-set old/new pairs, including identity Full aliases"},
        {"test": "A/B/C direct timing", "status": "VERIFIED_COMPLETE", "coverage": "333 groups,1323 conditions,20warmups+100formal each;3 fixed cases; GPU0 H20"},
        {"test": "independent method memory", "status": "VERIFIED_COMPLETE", "coverage": "11 separate model/method processes;1323 conditions"},
        {"test": "three training seeds", "status": "PARTIAL", "coverage": "current matched cohort contains training seed0 only"}]
    csv_write(HERE / "test_scope.csv", scope_rows)
    full = [r for r in budget_source if r["method"] == "full" and r["reconstruction_id"] == reconstruction_ids[0]]
    main_ratios = [r for r in ratios if r["reconstruction_id"] == reconstruction_ids[1]]
    readme = """# Darcy matched900/100 results, audited15 November2026

This package contains completed training-seed0 experiments, independently replayed from the original scalar records. It adds no training, GPU inference or modifications to the source experiment directory. The three model variants are Transolver3, official LinearNO point-temp, and the native convolution-temp Full-only supplement. They are not three independent training seeds.

All three backbones used900 training and100 validation cases, common900-only normalizers,1000epochs and225000optimizer updates. The two adaptive backbones each have their own1000-epoch CA head bound to that backbone's hash. The convolution model has no adaptive/head task because its native input is fixed85x85. Backbone/head selection used validation data; historical visibility of the smooth2 test population remains disclosed.

The accuracy test covers all200 smooth2 cases. Each point backbone evaluates Full once per case plus Uniform, PI, CPG beta0, CPG beta1 and CA at37.5%,50%,75% with sampling seeds0/1/2:9200 physical requests each. Conv contributes200Full requests. Old/new reconstruction views produce37200 scalar metric rows. The3600 CPG beta pairs preserve coarse indices, coarse predictions and final selected indices; only compensation changes. All unfavorable beta0 results are retained.

For each model/reconstruction and sampling seed, rAUEC integrates the case-mean error curve at the three budgets and divides by that model's matched Uniform area. Full is excluded. The reported SD is across three sampling seeds. Across-training-seed SD remains pending; independently normalized rAUEC is not an absolute cross-model accuracy ranking.

Formal timing covers333 paired groups and1323 conditions. Each condition has20warmups and100 formal direct wall-time measurements with GPU boundary synchronization. The three scenes are A CPU model-ready input to complete CPU float64 field; B GPU-resident input to complete GPU float64 field; and C explicitly reused geometry/selected-set reconstruction maps while acquisition and predictions are recomputed. Model/kernel startup and first-request costs are separate. Full has one physical series per model/case/scene, without fabricated sampling repeats or duplicate old/new timings. All timing ran under the recorded exclusive H20 GPU0 admission. Independent memory comprises11 model/method processes and1323 conditions.

`latency_raw_repeats.csv` stores the original compact warmup/formal observations; `backbone_latency_summary.csv` maps each group/condition to its checkpoint, head, population, boundary and code hashes. Every median,p95,mean and sampleSD was recomputed. `latency_profiles.csv` contains separate instrumented passes, not an additive substitute for direct latency. Reuse-cost estimates retain their explicit derived-estimate definitions. Memory covers PyTorch allocator peaks and post-request NVML samples, not CPU SciPy allocations or all transient external-library GPU peaks.

The original controller's generic final_accuracy flag did not read Darcy's actual acceptance keys. This audit checks `model_specific_validation_accepted`, complete raw matrices and exact latency/memory coverage, and supplies the previously pending independent scalar aggregation. It does not silently change the original receipt strings.

Weights, full physical fields, caches and long logs remain external. The source index records immutable hashes. Public CSVs contain scalar results, identities and training histories. Historical training Git IDs are not manufactured from the paper baseline. The official LinearNO source pin is3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269.

Full-field relative-L2 errors:

| Model | Full error |
|---|---:|
"""
    readme += "\n".join(f"| {r['backbone']} | {float(r['mean_error']):.9f} |" for r in full)
    readme += "\n\nNew-reconstruction rAUEC (mean ± sampling sampleSD):\n\n| Model | Method | rAUEC | Sampling SD |\n|---|---|---:|---:|\n"
    readme += "\n".join(f"| {r['backbone']} | {r['method']} | {float(r['mean']):.6f} | {float(r['sampling_sd']):.6f} |" for r in main_ratios)
    readme += "\n\nCompleted versus pending tests are enumerated in `test_scope.csv`; detailed evidence and limits are in `source_audit.json`. No three-training-seed claim is made for this seed0 package.\n"
    (HERE / "README.md").write_text(readme)
    (HERE / "read.me").write_text("Darcy seed0 source audit and public scalar export completed. See source_audit.json, test_scope.csv and README.md. No existing experiment changed and no GPU work started.\n")
    manifest = []
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name != "release_manifest.json":
            if path.suffix in [".csv", ".json", ".md", ".me"]:
                assert not re.search(r"/(?:mnt|Users|root)/", path.read_text()), path
            manifest.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path), "public": True})
    json_write(HERE / "release_manifest.json", {"status": "VERIFIED_COMPLETE", "scope": "matched training-seed0 cohort", "files": manifest,
               "full_followup_status": "PARTIAL", "weights_and_raw_fields_public": False})
    print(json.dumps({"checks": checks, "public_files": len(manifest), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
