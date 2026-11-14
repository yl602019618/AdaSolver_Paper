#!/usr/bin/env python3
"""Add ordered selection and coarse signal columns for CSV-only reproduction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args();out=args.output
    manifest_path=out/"score_manifest.json";manifest=json.loads(manifest_path.read_text())
    for method,info in manifest["methods"].items():
        with np.load(out/info["npz"]) as z:
            indices=z["actual_selected_indices"];coarse=z["coarse_indices"] if "coarse_indices" in z else np.array([],dtype=np.int64)
            n=len(z["xyz"]);selected=z["selected"];nretained=int(selected[coarse].sum())
        order=np.full(n,-1);order[indices]=np.arange(len(indices));coarse_order=np.full(n,-1);coarse_order[coarse]=np.arange(len(coarse))
        path=out/info["perpoint_csv"]
        with path.open() as f:rows=list(csv.DictReader(f))
        for i,r in enumerate(rows):r["selected_order"]="" if order[i]<0 else int(order[i]);r["coarse_order"]="" if coarse_order[i]<0 else int(coarse_order[i])
        with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        info["columns"].update(selected_order="selected_order",coarse_order="coarse_order")
        info["coarse_retention"]={"coarse_total":len(coarse),"coarse_selected_intersection":nretained,"all_coarse_retained":bool(nretained==len(coarse)) if len(coarse) else None,"scope":"actual saved sets, not an assumed policy"}
        if method=="pi":
            info["field_semantics"]["target_density"]["provenance"]="derived from frozen frac_extra/r and locked paper mixture formula; this full array is not computed by the native PI selector"
    for name,info in manifest["files"].items():
        path=out/name;info.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),allow_public=path.suffix==".csv",reason="user-authorized per-point figure source" if path.suffix==".csv" else "large feature/field cache remains internal")
    manifest["public_source_preparation_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    cpu=out/"ca_cpu_diagnostic";path=cpu/"run_manifest.json";m=json.loads(path.read_text())
    with np.load(cpu/"trace.npz") as z:
        xyz=z["xyz"];surface=z["surface"];score=z["normalized_score"];indices=z["actual_selected_indices"];coarse=z["coarse_indices"];selected=z["selected"]
        rawhead=z["ca_head_raw"];pressure=z["coarse_pressure_proxy"];features=z["ca_block0_features"];field=z["complete_prediction_physical"];n=len(xyz)
    rawfull=np.full(n,np.nan);rawfull[coarse]=rawhead;pressurefull=np.full(n,np.nan);pressurefull[coarse]=pressure
    feature_l2=np.full(n,np.nan);feature_l2[coarse]=np.linalg.norm(features.astype(np.float64),axis=1)
    feature_mean=np.full(n,np.nan);feature_mean[coarse]=features.astype(np.float64).mean(axis=1)
    feature_std=np.full(n,np.nan);feature_std[coarse]=features.astype(np.float64).std(axis=1,ddof=0)
    predicted_pressure=np.full(n,np.nan);predicted_pressure[surface]=field[surface,3]
    order=np.full(n,-1);order[indices]=np.arange(len(indices));coarse_order=np.full(n,-1);coarse_order[coarse]=np.arange(len(coarse))
    distance=cKDTree(xyz[surface].astype(np.float64)).query(xyz.astype(np.float64),k=1)[0]
    scale=max(float(np.quantile(score[surface],.95)),1e-12)
    priority=(1+64*np.exp(-distance/.1))*(1+2*np.clip(score/scale,0,1));target=priority/priority.sum()
    csvpath=cpu/"perpoint.csv"
    with csvpath.open() as f:rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):
        r.update(selected_order="" if order[i]<0 else int(order[i]),coarse_order="" if coarse_order[i]<0 else int(coarse_order[i]),coarse_head_output_raw="" if np.isnan(rawfull[i]) else rawfull[i],coarse_pressure_proxy="" if np.isnan(pressurefull[i]) else pressurefull[i],target_density=target[i],target_reference_ratio=target[i]*n,coarse_feature_l2="" if np.isnan(feature_l2[i]) else feature_l2[i],coarse_feature_mean="" if np.isnan(feature_mean[i]) else feature_mean[i],coarse_feature_std="" if np.isnan(feature_std[i]) else feature_std[i],predicted_surface_pressure="" if np.isnan(predicted_pressure[i]) else predicted_pressure[i])
    with csvpath.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    m["columns"].update(selected_order="selected_order",coarse_order="coarse_order",coarse_head_output_raw="coarse_head_output_raw",coarse_pressure_proxy="coarse_pressure_proxy",target_density="target_density",coarse_feature_l2="coarse_feature_l2",coarse_feature_mean="coarse_feature_mean",coarse_feature_std="coarse_feature_std",predicted_surface_pressure="predicted_surface_pressure")
    m["coarse_retention"]={"coarse_total":len(coarse),"coarse_selected_intersection":int(selected[coarse].sum()),"all_coarse_retained":bool(selected[coarse].all())}
    m["coarse_signal_scope"]="raw standardized MLP output and physical pressure proxy at the actual ordered coarse indices only; all other per-point rows empty"
    m["feature_summary"]={"source":"trace.npz/ca_block0_features","dimensions":features.shape[1],"support":"coarse indices only","l2":"Euclidean norm across256 channels computed in float64","mean":"arithmetic mean across256 channels","std":"population SD across256 channels; ddof0; not sampling or training variability"}
    m["target_density_definition"]="own CPU score defines normalized native priority (1+64exp(-d/.1))*(1+2clip(score/q95(surface_score),0,1)); all target/score/error fields belong to this CPU run"
    m["public_source_preparation_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,info in m["files"].items():
        q=cpu/name;info.update(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest(),allow_public=q.suffix==".csv")
    path.write_text(json.dumps(m,indent=2)+"\n")
    print("CSV sources ready; exact ordered indices retained; coarse signals on native coarse support")


if __name__=="__main__":main()
