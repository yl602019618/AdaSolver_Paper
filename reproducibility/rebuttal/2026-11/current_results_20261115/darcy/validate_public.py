"""Validate the public scalar package; no model weights, CUDA or raw CFD files."""
from pathlib import Path
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import numpy as np


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); root = args.data_dir
    records = read(root / "metrics_records.csv")
    assert len(records) == 37200
    curves = collections.defaultdict(list)
    identities = set()
    for r in records:
        key = (r["backbone"], r["reconstruction_id"], r["method"], r["budget_fraction"], r["sampling_seed"])
        identity = key + (r["case_id"],)
        assert identity not in identities
        identities.add(identity)
        curves[key].append(float(r["value"]))
        assert r["training_seed"] == "0" and r["status"] == "ok"
    assert all(len(v) == 200 for v in curves.values())
    area = lambda y: .125*(y[0]+y[1])/2 + .25*(y[1]+y[2])/2
    ratios = read(root / "backbone_rauec.csv")
    for row in ratios:
        values = []
        for seed in ["0", "1", "2"]:
            ys = []
            for method in [row["method"], "uniform"]:
                ys.append([np.mean(curves[(row["backbone"], row["reconstruction_id"], method, budget, seed)]) for budget in ["0.375", "0.5", "0.75"]])
            value = area(ys[0])/area(ys[1]); values.append(value)
            near(value, row["seed"+seed])
        near(np.mean(values), row["mean"]); near(np.std(values, ddof=1), row["sampling_sd"])
    raw_path = root / "latency_raw_repeats.csv.gz"
    if not raw_path.exists(): raw_path = root / "latency_raw_repeats.csv"
    observations = read(raw_path)
    timing = collections.defaultdict(list)
    warmups = collections.Counter()
    for r in observations:
        key = (r["group_id"], r["condition"])
        if r["kind"] == "formal": timing[key].append(float(r["total_ms"]))
        else:
            assert r["kind"] == "warmup"
            warmups[key] += 1
    assert len(timing) == 1323 and sum(len(v) for v in timing.values()) == 132300
    for r in read(root / "backbone_latency_summary.csv"):
        key = (r["group_id"], r["condition"]); v = np.array(timing[key])
        assert len(v) == 100 and warmups[key] == 20
        near(np.median(v), r["direct_total_ms"]); near(np.percentile(v, 95), r["p95_ms"])
        near(v.mean(), r["mean_ms"]); near(v.std(ddof=1), r["sample_sd_ms"])
        assert r["training_seed"] == "0" and r["output_dtype"] == "float64"
    registry = read(root / "checkpoint_registry.csv")
    assert len(registry) == 3 and len(read(root / "training_history.csv")) == 5000
    assert sum(r["head_applicable"] == "True" for r in registry) == 2
    manifest = json.loads((root / "release_manifest.json").read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    from PIL import Image
    for path in (root / "figures").glob("*.png"):
        with Image.open(path) as image: image.verify()
    print(json.dumps({"passed": True, "accuracy_rows": 37200, "rAUEC_rows_recomputed": len(ratios),
                      "timing_conditions_recomputed": len(timing), "formal_repeats": 132300,
                      "manifest_files": len(manifest["files"]), "training_seeds_per_model": 1}, indent=2))


if __name__ == "__main__":
    main()
