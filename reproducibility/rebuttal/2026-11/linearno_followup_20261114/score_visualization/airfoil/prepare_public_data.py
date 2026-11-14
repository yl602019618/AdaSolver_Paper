"""Export scalar plot data; full feature tensors and CFD truth remain external."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    public, external = [], []
    for method in ["uniform", "pi", "cpg", "ca"]:
        with np.load(ROOT / f"{method}_score_arrays.npz") as z:
            n = len(z["coords"])
            selected_order, coarse_order = np.full(n, np.nan), np.full(n, np.nan)
            selected_order[z["indices"]] = np.arange(len(z["indices"]))
            coarse_order[z["coarse_indices"]] = np.arange(len(z["coarse_indices"]))
            coarse_pressure, standardized = np.full(n, np.nan), np.full(n, np.nan)
            if "coarse_pressure" in z:
                coarse_pressure[z["coarse_indices"]] = z["coarse_pressure"].reshape(-1)
            if "ca_standardized_output" in z:
                standardized[z["coarse_indices"]] = z["ca_standardized_output"].reshape(-1)
            normalized_prior = np.full(n, np.nan)
            feature_l2, feature_mean, feature_std = [np.full(n, np.nan) for _ in range(3)]
            if method == "pi":
                normalized_prior = z["geometry_prior"] / z["geometry_prior"].sum()
            elif method == "uniform":
                normalized_prior[:] = 1./n
            if "ca_features" in z:
                features = z["ca_features"][0].astype(np.float64)
                feature_l2[z["coarse_indices"]] = np.linalg.norm(features, axis=1)
                feature_mean[z["coarse_indices"]] = features.mean(axis=1)
                feature_std[z["coarse_indices"]] = features.std(axis=1, ddof=0)
            coords = z["coords"].copy()
        path = ROOT / f"{method}_points.csv"
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        added = ["selected_order", "coarse_order", "coarse_pressure", "ca_standardized_output", "normalized_prior",
                 "coarse_feature_l2", "coarse_feature_mean", "coarse_feature_std"]
        fields = list(fields) + [k for k in added if k not in fields]
        for i, row in enumerate(rows):
            for key, values in zip(added, [selected_order, coarse_order, coarse_pressure, standardized, normalized_prior,
                                           feature_l2, feature_mean, feature_std]):
                row[key] = "" if not np.isfinite(values[i]) else values[i]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        error = ROOT / f"{method}_error_points.csv"
        with np.load(ROOT / f"{method}_error_only.npz") as z:
            absolute_error = z["absolute_error"].reshape(-1)
        with error.open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["point_index", "x", "y", "absolute_error"])
            writer.writerows((i, *coords[i], absolute_error[i]) for i in range(n))
        for p in [path, error]:
            public.append({"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        for suffix, reason in [("score_arrays", "full CA128D feature tensor and intermediate arrays remain external"),
                               ("error_only", "contains CFD ground-truth field; external only")]:
            p = ROOT / f"{method}_{suffix}.npz"
            external.append({"artifact_id": f"local:airfoil-score-case1001/{p.name}", "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "public": False, "reason": reason})
    (ROOT / "public_data_manifest.json").write_text(json.dumps({"public_scalar_csvs": public,
        "external_artifacts": external, "plot_inputs": "Only the public scalar CSVs are required by plot_scores.py",
        "ground_truth_in_public_csv": False}, indent=2) + "\n")
    print("Prepared8 public scalar CSVs; full NPZ/CFD truth remain external")


if __name__ == "__main__":
    main()
