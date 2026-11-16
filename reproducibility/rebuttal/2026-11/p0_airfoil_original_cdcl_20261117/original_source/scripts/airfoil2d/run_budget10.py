"""Run frozen/configured Airfoil candidates through the same evaluator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from adasolver.evaluation.airfoil2d import AirfoilEvaluator, summarize_airfoil_records, airfoil_rauec
from adasolver.inference.airfoil2d_contract import BUDGETS
from adasolver.inference.airfoil2d_config import resolve_config, read_config
from adasolver.train.airfoil2d_ca_dataset import DEFAULT_OPERATOR, DEFAULT_MANIFEST, DEFAULT_DATA


def run(configs, ids, output, *, device="cuda:0", seeds=(0,1,2), protocol="positive_lift",
        checkpoint=DEFAULT_OPERATOR, manifest=DEFAULT_MANIFEST, data_path=DEFAULT_DATA,
        role="calibration", min_true_cl=.01):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"run output not empty: {output}")
    if role in ("calibration", "confirmation") and any(1000 <= i < 1200 for i in ids):
        raise ValueError("calibration/confirmation cannot include official test IDs")
    configs = [resolve_config(c["method"],c) for c in configs]
    if len({c["name"] for c in configs}) != len(configs):
        raise ValueError("candidate names must be unique")
    if not any(c["name"] == "uniform" and c["method"] == "uniform" for c in configs):
        configs.insert(0,resolve_config("uniform"))
    runner = AirfoilEvaluator(checkpoint,manifest,data_path,device)
    selected = runner.selected_ids(ids,protocol,min_true_cl=min_true_cl)
    if not selected:
        raise ValueError("no samples pass the protocol")
    output.mkdir(parents=True,exist_ok=True)
    (output/"run_manifest.json").write_text(json.dumps(dict(role=role,protocol=protocol,min_true_cl=min_true_cl,
        source_ids=list(ids),sample_ids=selected,seeds=list(seeds),configs=configs,
        operator_sha256=runner.operator_hash,normalizer_sha256=runner.data.normalizer_sha256,
        status="running"),indent=2)+"\n")
    rows={c["name"]:[] for c in configs}
    streams={c["name"]:(output/(c["name"]+"_raw.jsonl")).open("x") for c in configs}
    # Warm up using a training geometry, never a test output used for selection.
    for config in configs:
        runner.record(0,0,.375,config)
    try:
        for index,sid in enumerate(selected):
            runner.coarse_cache.clear();runner.feature_cache.clear();runner.patch_cache.clear()
            for seed in seeds:
                for config in configs:
                    for budget in BUDGETS:
                        row=runner.record(sid,seed,budget,config)
                        row.update(evaluation_protocol=protocol,run_role=role,min_true_cl=min_true_cl)
                        rows[config["name"]].append(row)
                        streams[config["name"]].write(json.dumps(row,allow_nan=False)+"\n")
                for stream in streams.values():stream.flush()
            print(f"{role}: {index+1}/{len(selected)} samples, {len(configs)} methods",flush=True)
    finally:
        for stream in streams.values():stream.close()
    summaries={}
    for config in configs:
        name=config["name"]
        summary=summarize_airfoil_records(rows[name],name)
        summary.update(sample_ids=selected,seeds=list(seeds),protocol=protocol,role=role,min_true_cl=min_true_cl,
                       operator_sha256=runner.operator_hash,normalizer_sha256=runner.data.normalizer_sha256,
                       config=config,records=len(rows[name]))
        summaries[name]=summary
    for name,summary in summaries.items():
        summary["rAUEC_pressure"]=airfoil_rauec(summary,summaries["uniform"],"mean_pressure_field_relative_l2")
        summary["rAUEC_Cd_over_Cl"]=airfoil_rauec(summary,summaries["uniform"],"mean_ratio_abs_error")
        (output/(name+".json")).write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    comparison={name:{key:summary[key] for key in ("rAUEC_pressure","rAUEC_Cd_over_Cl","records","config")}
                for name,summary in summaries.items()}
    (output/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    run_manifest=json.loads((output/"run_manifest.json").read_text());run_manifest["status"]="complete"
    (output/"run_manifest.json").write_text(json.dumps(run_manifest,indent=2)+"\n")
    print(json.dumps(comparison,indent=2),flush=True)
    return comparison


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--configs",type=Path,nargs="+",required=True)
    p.add_argument("--ids",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--role",choices=("smoke","calibration","confirmation","official","sensitivity"),default="calibration")
    p.add_argument("--protocol",choices=("positive_lift","all"),default="positive_lift")
    p.add_argument("--min-true-cl",type=float,default=.01)
    a=p.parse_args();configs=[]
    for path in a.configs:
        value=read_config(path);configs.extend(value if isinstance(value,list) else [value])
    ids=json.loads(a.ids.read_text())
    if isinstance(ids,dict):ids=ids["sample_ids"]
    run(configs,ids,a.output,device=a.device,seeds=a.seeds,role=a.role,protocol=a.protocol,min_true_cl=a.min_true_cl)


if __name__=="__main__":main()
