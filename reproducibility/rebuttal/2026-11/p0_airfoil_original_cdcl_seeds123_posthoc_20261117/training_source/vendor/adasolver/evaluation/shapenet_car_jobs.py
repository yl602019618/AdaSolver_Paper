"""Disjoint sample shards, then an audited merge into complete Car runs."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from adasolver.data.shapenet_car_runtime import CarRuntime
from adasolver.evaluation.shapenet_car import config_name
from adasolver.evaluation.shapenet_car_compare import load_run, summarize_records, record_key
from adasolver.inference.shapenet_car_contract import ROOT, json_hash


def merge_shards(output, shard_roots, configs, expected_ids):
    output = Path(output)
    for config in configs:
        destination = output/config_name(config)
        if destination.exists():
            raise FileExistsError(destination)
        items = [load_run(Path(root)/config_name(config)) for root in shard_roots]
        first = items[0][0]
        rows, seen = [], set()
        for root, (meta, records, _) in zip(shard_roots, items):
            for key in ("config", "operator_sha256", "normalizer_sha256", "head_sha256", "split_hash",
                        "source_sha256", "split_role", "seeds", "budgets", "reconstruction"):
                if meta[key] != first[key]:
                    raise ValueError(f"shard provenance differs: {key}")
            if seen.intersection(meta["sample_ids"]):
                raise ValueError("overlapping sample shards")
            seen.update(meta["sample_ids"])
            for row in records:
                row["source_run_id"] = row["run_id"]
                row["arrays_path"] = os.path.relpath(Path(root)/config_name(config)/row["arrays_path"], destination)
                rows.append(row)
        if seen != set(expected_ids):
            raise ValueError("merged sample IDs do not equal the requested split")
        rows.sort(key=record_key)
        merged = {**first, "sample_ids": list(expected_ids),
                  "shard_manifests": [str(Path(root)/config_name(config)/"run_manifest.json") for root in shard_roots]}
        merged["run_id"] = json_hash({"shard_run_ids": [i[0]["run_id"] for i in items], "sample_ids": list(expected_ids)})
        merged["environment"] = {"shards": [item[0]["environment"] for item in items]}
        merged["records"] = len(rows)
        for row in rows:
            row["run_id"] = merged["run_id"]
        summary = summarize_records(rows, budgets=merged["budgets"])
        destination.mkdir(parents=True)
        (destination/"run_manifest.json").write_text(json.dumps(merged, indent=2)+"\n")
        (destination/"raw.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
        (destination/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


def run_sharded(output, configs, *, manifest, data_path, splits_path, split,
                seeds=(0,1,2), devices=("cuda:2","cuda:3","cuda:4","cuda:5"),
                head_manifest=None, save_prediction_ids=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = CarRuntime(manifest, data_path)
    ids = runtime.splits(splits_path)[split]
    devices = list(devices[:len(ids)])
    config_path = output/"candidate_configs.json"
    config_path.write_text(json.dumps(configs, indent=2)+"\n")
    jobs, roots = [], []
    try:
        for i, device in enumerate(devices):
            start, end = len(ids)*i//len(devices), len(ids)*(i+1)//len(devices)
            root = output/"shards"/f"part{i}"
            root.mkdir(parents=True, exist_ok=True)
            log = (root/"run.log").open("w")
            command = [sys.executable, "-m", "adasolver.evaluation.shapenet_car", "--output", str(root),
                "--configs", str(config_path), "--manifest", str(manifest), "--splits", str(splits_path),
                "--split", split, "--device", device, "--samples", str(end-start), "--sample-offset", str(start),
                "--seeds", *map(str,seeds)]
            if data_path:
                command += ["--data-path", str(data_path)]
            if head_manifest:
                command += ["--head-manifest", str(head_manifest)]
            if save_prediction_ids:
                command += ["--save-prediction-ids", *save_prediction_ids]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((process, log)); roots.append(root)
            print(f"shard={i} pid={process.pid} device={device} samples={end-start} log={root/'run.log'}", flush=True)
        codes = []
        for process, log in jobs:
            codes.append(process.wait()); log.close()
        if any(codes):
            raise RuntimeError(f"evaluation shard failures: {codes}; see per-shard logs")
        merge_shards(output, roots, configs, ids)
        print(f"Merged {len(ids)} samples, {len(configs)} configurations into {output}", flush=True)
    finally:
        for process, log in jobs:
            if process.poll() is None:
                process.terminate()
            if not log.closed:
                log.close()
