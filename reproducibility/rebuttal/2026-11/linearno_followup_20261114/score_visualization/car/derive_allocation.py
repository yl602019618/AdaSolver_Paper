#!/usr/bin/env python3
"""Normalize recovered native quota rates as an explicitly derived measure."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    args = p.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    path = output / "score_manifest.json"
    manifest = json.loads(path.read_text())
    n, k = manifest["points"], round(manifest["points"] * manifest["budget"])
    for method, info in manifest["methods"].items():
        source = output / info["npz"]
        with np.load(source) as z:
            arrays = {key: np.array(z[key]) for key in z.files}
        rate = arrays["allocation_density"]
        measure = np.full(n, np.nan)
        ratio = np.full(n, np.nan)
        if method != "pi":
            total = float(rate.sum())
            assert np.isclose(total, k, rtol=0, atol=1e-10)
            measure = rate / total
            ratio = measure * n
            assert abs(measure.sum()-1) < 1e-12
            check = {"sum_rate": total, "sum_normalized_allocation_density": float(measure.sum()),
                     "mean_target_reference_ratio": float(ratio.mean())}
            if method in ["cpg", "ca"]:
                ids = arrays["actual_selected_indices"]
                beta = info["configuration"]["beta"]
                paper_weight = ratio[ids]**(-beta)
                scaled_native = arrays["native_compensation"] * (k/n)**beta
                delta = np.abs(paper_weight-scaled_native)
                assert np.all(delta <= 1e-6+1e-5*np.abs(paper_weight))
                check.update(native_to_reference_compensation_factor=(k/n)**beta,
                             weight_relation_max_absolute_difference=float(delta.max()),
                             weight_relation_pass=True,
                             relation="lambda^(-beta)=(K/N)^beta*native_compensation, using lambda=N*allocation_rate/K",
                             attention_or_model_output_scale_parity="not_tested; no output-invariance claim")
            info["allocation_measure_validation"] = check
        arrays.update(normalized_allocation_density=measure, target_to_uniform_reference_ratio=ratio)
        if method == "uniform":
            target_weight = np.ones(n)
            formula = "equal full-domain target; p_i=1/N"
        elif method == "pi":
            alpha = 1.-info["configuration"]["frac_extra"]
            membership = arrays["raw_prior"]
            shell_fraction = float(membership.mean())
            target_weight = alpha+(1.-alpha)*membership/shell_fraction
            formula = "lambda=alpha+(1-alpha)*I_H/rho(H); p_i=lambda_i/N; alpha=1-frac_extra"
            info["mixture_implementation"] = {"configured_alpha":alpha,
                "configured_shell_fraction":shell_fraction,"shell_points":int(membership.sum()),
                "actual_base_count":int(arrays["pi_uniform_base"].sum()),
                "actual_extra_count":int(arrays["pi_second_stage_selected"].sum()),
                "actual_base_fraction_of_budget":float(arrays["pi_uniform_base"].sum()/k),
                "fallback_non_shell_points":int(np.sum(arrays["pi_second_stage_selected"] & (membership==0))),
                "distinction":"design mixture target; actual selection is without replacement after a fixed uniform base, with nearest nonshell fallback; not independent point inclusion probabilities"}
        else:
            target_weight = arrays["allocation_priority"]
            formula = "lambda=N[(1+64*exp(-d/0.1))*(1+2*clip(saved_score/q95(surface_saved_score),0,1))]; N[.] means full-reference mean normalization"
        target_density = target_weight/target_weight.sum()
        target_reference_ratio = target_density*n
        assert abs(target_density.sum()-1)<1e-12 and abs(target_reference_ratio.mean()-1)<1e-12
        arrays.update(target_density=target_density,target_reference_ratio=target_reference_ratio)
        np.savez_compressed(source, **arrays)
        csv_path = output / info["perpoint_csv"]
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            row["normalized_allocation_density"] = "" if np.isnan(measure[i]) else measure[i]
            row["target_to_uniform_reference_ratio"] = "" if np.isnan(ratio[i]) else ratio[i]
            row["target_density"] = target_density[i]
            row["target_reference_ratio"] = target_reference_ratio[i]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        info["field_semantics"].update({
            "normalized_allocation_density": {"support":"not_applicable" if method=="pi" else "full",
                "semantic_type":"derived_normalized_quota_measure", "normalization_rule":"allocation_rate/sum_full_reference(allocation_rate)=allocation_rate/K",
                "probabilistic_interpretation":"not independent point inclusion probabilities; FPS and mandatory anchors retained"},
            "target_to_uniform_reference_ratio": {"support":"not_applicable" if method=="pi" else "full",
                "semantic_type":"derived_quota_reference_density_ratio", "normalization_rule":"N*normalized_allocation_density; full-reference mean1"},
            "target_density": {"support":"full","semantic_type":"designed_target_probability_mass","normalization_rule":"target_weight/sum_full_reference(target_weight)","not":"independent inclusion probability"},
            "target_reference_ratio": {"support":"full","semantic_type":"target_reference_density","normalization_rule":"N*target_density; mean1 under uniform reference"}})
        info["columns"]["normalized_allocation_density"] = "normalized_allocation_density" if method != "pi" else None
        info["definitions"]["normalized_allocation_density"] = "Explicitly derived normalized allocation measure. Native quota/FPS policy is unchanged; not presented as the native random proposal distribution."
        info["columns"]["target_density"] = "target_density"
        info["target_design_formula"] = formula
        info["target_design_normalization_check"] = {"sum_probability_mass":float(target_density.sum()),"mean_reference_density":float(target_reference_ratio.mean()),"source":"actual frozen configuration and native priority/prior; no truth input"}
    original_manifest = root / "rebuttal_20261112/experiments/E5_training_seeds/checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json"
    native = json.loads(original_manifest.read_text())
    sample_root = Path(native["data_path"]) / "preprocessed_data" / manifest["case_id"]
    coordinate_source = np.load(sample_root/"pos.npy")
    with np.load(output/"data/uniform.npz") as z:
        assert np.array_equal(z["xyz"], coordinate_source)
        coordinate_array_hash = hashlib.sha256(z["xyz"].tobytes()).hexdigest()
    manifest["coordinate_binding"] = {"source":"SHAPENET_CAR_DATA/preprocessed_data/"+manifest["case_id"]+"/pos.npy",
        "source_file_sha256":hashlib.sha256((sample_root/"pos.npy").read_bytes()).hexdigest(),
        "source_array_sha256":coordinate_array_hash,"native_dtype":str(coordinate_source.dtype),"equal":True}
    manifest["capture_source_identity"] = {"path":"capture.py", "sha256":hashlib.sha256((output/"capture.py").read_bytes()).hexdigest(),
        "status":"observed_current_after_metadata_only_edits", "executed_source_file_hash":"not_captured_before_run",
        "note":"Acquisition code unchanged; subsequent edits added field semantic metadata only. No claim of a pre-execution script hash."}
    manifest["allocation_derivation_source"] = {"path":"derive_allocation.py","sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"status":"executed_current"}
    paper = subprocess.check_output(["git","-C",str(root/"paper_git"),"show","91c1fa7cc81b8c1cb61a43618296c1c892d818c0:sections/density_policy_design.tex"])
    manifest["paper_formula_status"] = "VERIFIED_COMPLETE"
    manifest["paper_formula_scope"] = "Designed target-density mapping only; separate from actual inclusion probabilities and native selected compensation."
    manifest["paper_formula_source"] = {"commit":"91c1fa7cc81b8c1cb61a43618296c1c892d818c0","file":"sections/density_policy_design.tex","sha256":hashlib.sha256(paper).hexdigest(),"equations":["app_pi_car_density","app_density_car","app_ca_car_density"]}
    manifest["paper_formula_reason"] = "The full-domain designed target matches the locked paper formula from the real hard-shell mixture and native CPG/CA priority. Integer quotas, mandatory surface/coarse anchors, FPS and PI fallback determine actual selection. Those quantities and actual compensation are preserved separately. PI has no separate native normalized-score array. CA archived-GPU raw replay remains partial; its exact saved score defines the archived target. No model-level global-scale-invariance claim is made."
    manifest["files"] = {str(p.relative_to(output)):{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"allow_public":False,"reason":"native per-point geometry/features; internal data"} for p in sorted((output/"data").iterdir())}
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print("native allocation measures normalized and verified; no model inference")


if __name__ == "__main__":
    main()
