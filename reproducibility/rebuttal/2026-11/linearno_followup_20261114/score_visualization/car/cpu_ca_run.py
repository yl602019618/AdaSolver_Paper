#!/usr/bin/env python3
"""Independent CPU CA diagnostic: own raw head -> score -> sampler -> field.

The archived GPU arrays are used only for a separately labeled comparison.
They never select points or supply errors to this run.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    if (output/"run_manifest.json").exists():
        raise FileExistsError("A completed diagnostic is never overwritten")
    output.mkdir(parents=True,exist_ok=True)
    os.environ["ADASOLVER_NATIVE_CACHE"] = str(output/"native_fps_cache")
    e5 = root/"rebuttal_20261112/experiments/E5_training_seeds"
    sys.path.insert(0,str(e5/"code/vendor"))
    import torch
    from adasolver.data.shapenet_car_runtime import CarRuntime
    from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
    from adasolver.inference.cpg.shapenet_car import surface_scalar_score,normalize_cpg_score
    from adasolver.inference.cpg.shapenet_car_stratified import ShapeNetCarStratified
    from adasolver.inference.shapenet_car_contract import coarse_count,rng_for
    from adasolver.layers import install_compensation
    from adasolver.metrics.shapenet_car import reconstruct_car_fields,dual_field_metrics
    torch.set_num_threads(4)
    sid,budget,seed="param0/100715345ee54d7ae38b52b4ee9d36a3",.5,0
    source=e5/"records/seed0/adaptive/ca_field_99d5b859e461"
    cfg=json.loads((source/"run_manifest.json").read_text())["config"]
    model=e5/"checkpoints/seed0/shapenet_car_t3_ddp2_e400_trainingseed0_manifest.json"
    head=e5/"checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json"
    runtime=CarRuntime(model)
    item=runtime.sample(sid)
    xyz,surface=item.pos.numpy(),item.surface_mask.numpy()
    n=len(xyz);count=round(n*budget)
    operator=runtime.load_operator("cpu")
    context,_=install_compensation(operator)
    ca=ShapeNetCarCA(operator,head,runtime.binding(),"cpu",use_cuda_graph=False)
    coarse_indices=rng_for(sid,seed).choice(n,coarse_count(n),replace=False)
    cp,cx=item.pos[coarse_indices][None],item.x_norm[coarse_indices][None]
    started=time.perf_counter()
    with torch.no_grad():
        context.clear()
        features,cache=ca.feature_pass(cp,cx,n,capture=True)
        head_raw=ca.head(features)[0,:,0]
        pressure=head_raw*ca.metadata["target_std"]+ca.metadata["target_mean"]
        raw=surface_scalar_score(xyz,surface,coarse_indices,pressure.numpy(),feature=cfg["feature"],
                                k=cfg["neighborhood"],surface_band=cfg["surface_band"])
        score=normalize_cpg_score(raw,method=cfg["normalization"],floor=cfg["score_floor"],power=cfg["score_power"])
        sampler=ShapeNetCarStratified(xyz,surface,**{k:cfg[k] for k in ["wall_scale","wall_amplitude","physics_strength","beta","placement","score_bins"]})
        sampling=sampler.refine_score(coarse_indices,score,count,rng_for(sid,seed,"refinement",budget))
        indices=sampling.indices
        weights=torch.as_tensor(sampling.compensation)
        context.set(weights)
        try:
            prediction_norm=ca.final(item.pos[indices][None],item.x_norm[indices][None],coarse_indices,indices,cache,weights)[0].numpy()
        finally:
            context.clear()
    # Ground truth is first accessed after all acquisition, selection and final
    # prediction. It is used only to evaluate this new run's own physical field.
    velocity,pressure_field,surface_indices=reconstruct_car_fields(indices,prediction_norm,xyz,surface,
        prediction_is_normalized=True,output_mean=runtime.output_normalizer.mean.numpy(),
        output_std=runtime.output_normalizer.std.numpy(),k=4)
    field=np.zeros((n,4),dtype=np.float64)
    field[:,:3]=velocity;field[surface_indices,3]=pressure_field
    truth=item.target_phys.numpy()
    metrics=dataclasses.asdict(dual_field_metrics(field,truth,surface))
    elapsed=time.perf_counter()-started
    selected=np.zeros(n,dtype=bool);selected[indices]=True
    coarse=np.zeros(n,dtype=bool);coarse[coarse_indices]=True
    strata=np.zeros(n,dtype=np.int64)
    bands=np.searchsorted(cfg["wall_scale"]*np.array([.25,.5,1.,2.,4.,8.,16.]),sampler.distance)
    next_id=1
    for band in np.unique(bands[~surface]):
        ids=np.flatnonzero((bands==band)&~surface)
        cuts=np.unique(np.quantile(score[ids],np.arange(1,cfg["score_bins"])/cfg["score_bins"]))
        bins=np.searchsorted(cuts,score[ids],side="left")
        for rank in np.unique(bins):
            strata[ids[bins==rank]]=next_id;next_id+=1
    mass=np.bincount(strata,minlength=next_id)
    counts=np.bincount(strata[indices],minlength=next_id)
    assert np.array_equal(counts,sampling.metadata["stratum_selected_counts"])
    allocation=counts[strata]/mass[strata]
    assert np.array_equal(allocation[indices],sampling.sampling_density)
    assert abs(allocation.sum()-count)<1e-10
    mu=allocation/count
    density=np.full(n,np.nan);density[indices]=sampling.sampling_density
    compensation=np.full(n,np.nan);compensation[indices]=sampling.compensation
    pressure_error=np.full(n,np.nan);pressure_error[surface]=np.abs(field[surface,3]-truth[surface,3])
    arrays=dict(index=np.arange(n),xyz=xyz,surface=surface,coarse=coarse,selected=selected,
        actual_selected_indices=indices,coarse_indices=coarse_indices,ca_block0_features=features[0].numpy(),
        ca_head_raw=head_raw.numpy(),coarse_pressure_proxy=pressure.numpy(),raw_score=raw,normalized_score=score,
        stratum=strata,allocation_density=allocation,normalized_allocation_density=mu,
        compensation_density_selected=density,compensation_weight_selected=compensation,
        final_prediction_normalized=prediction_norm,complete_prediction_physical=field,
        velocity_absolute_error=np.linalg.norm(field[:,:3]-truth[:,:3],axis=1),surface_pressure_absolute_error=pressure_error)
    np.savez_compressed(output/"trace.npz",**arrays)
    with (output/"perpoint.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["index","x","y","z","surface","coarse","selected","raw_score","normalized_score","allocation_density","normalized_allocation_density","compensation_density_selected","compensation_weight_selected","velocity_absolute_error","surface_pressure_absolute_error"])
        for i in range(n):
            row=[i,*xyz[i],int(surface[i]),int(coarse[i]),int(selected[i]),raw[i],score[i],allocation[i],mu[i],density[i],compensation[i],arrays["velocity_absolute_error"][i],pressure_error[i]]
            writer.writerow(["" if isinstance(v,(float,np.floating)) and np.isnan(v) else v for v in row])
    # A separate comparison preserves the difference from the archived GPU run.
    with np.load(source/"arrays"/f"{sid.replace('/','_')}_s0_b0.500.npz") as z:
        difference=np.abs(score-z["score"])
        comparison=dict(scope="comparison only; GPU values not used for CPU score/selection/error",
            normalized_score_max_abs=float(difference.max()),normalized_score_violations=int(np.sum(difference>1e-6+1e-5*np.abs(z["score"]))),
            atol=1e-6,rtol=1e-5,ordered_selected_indices_equal=bool(np.array_equal(indices,z["indices"])),
            selected_set_symmetric_difference=len(set(indices)^set(z["indices"])))
    sha=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
    manifest=dict(run_id="car_ca_cpu_score_diagnostic_v1",status="VERIFIED_COMPLETE",scope="one-case independent CPU diagnostic; no formal latency or cross-training claim",
        backend="CPU native eager FP32",device="cpu",GPU_execution=False,case_id=sid,training_seed=0,head_seed=0,sampling_seed=0,budget=budget,
        checkpoint_sha256=runtime.metadata["checkpoint_sha256"],normalizer_sha256=runtime.metadata["normalizer_sha256"],head_sha256=ca.metadata["head_sha256"],
        configuration=cfg,points=n,selected_count=len(indices),coarse_count=len(coarse_indices),selected_ordered_sha256=sha(indices),coarse_ordered_sha256=sha(coarse_indices),
        score_sha256=sha(score),raw_score_sha256=sha(raw),complete_prediction_sha256=sha(field),metrics=metrics,diagnostic_wall_seconds=elapsed,
        truth_loaded_by_dataset_loader=True,truth_used_for_acquisition=False,truth_used_for_diagnostic_error=True,
        truth_dependency="target_phys is first accessed after score, sampling and final model prediction; it supplies only this run's own error",
        normalized_score_rule="max(raw/q95(raw),0.1)^1",raw_score_rule="native surface pressure-proxy gradient energy, nearest surface lift and exp(-surface distance/0.1)",
        density_rule="native integer stratum quota/reference count; normalized_allocation=rate/K is an explicitly derived quota measure, not independent inclusion probability",
        perpoint_csv="perpoint.csv",npz="trace.npz",columns={"index":"index","xyz":["x","y","z"],"selected":"selected","coarse":"coarse","rawscore":"raw_score","normalizedscore":"normalized_score","density":"normalized_allocation_density"},
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_script_identity="executed_current",comparison_to_archived_GPU=comparison,
        files={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"allow_public":False} for p in [output/"trace.npz",output/"perpoint.csv"]})
    (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"run_id":manifest["run_id"],"metrics":metrics,"comparison":comparison,"wall_seconds":elapsed},indent=2))


if __name__=="__main__":
    main()
