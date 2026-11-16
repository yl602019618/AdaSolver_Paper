"""Evidence-bound curves, 3D surface maps and volume-slab reconstruction plates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from adasolver.data.shapenet_car_runtime import CarRuntime
from adasolver.evaluation.shapenet_car_compare import load_run
from adasolver.inference.shapenet_car_contract import BUDGETS, ROOT
from adasolver.inference.pi.shapenet_car import car_surface_distance

COLORS = {"uniform": "#666666", "pi": "#0072B2", "cpg": "#D55E00", "ca": "#009E73"}
MARKERS = {"uniform": "o", "pi": "s", "cpg": "^", "ca": "D"}


def save(fig, path, vector=False):
    # Budget fractions are part of the stem (e.g. _0.375), not extensions.
    fig.savefig(path.parent/(path.name+".png"), dpi=180, bbox_inches="tight")
    if vector:
        fig.savefig(path.parent/(path.name+".pdf"), bbox_inches="tight")
        fig.savefig(path.parent/(path.name+".svg"), bbox_inches="tight")
    plt.close(fig)


def surface_view(ax, pos, values, *, cmap="viridis", norm=None, size=1.6, bounds=None):
    artist = ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=values, cmap=cmap,
                        norm=norm, s=size, linewidths=0, rasterized=True)
    ax.view_init(elev=20, azim=-65)
    reference = pos if bounds is None else bounds
    ax.set_box_aspect(np.ptp(reference[:, [0, 2, 1]], axis=0))
    ax.set_xlim(reference[:,0].min(),reference[:,0].max())
    ax.set_ylim(reference[:,2].min(),reference[:,2].max())
    ax.set_zlim(reference[:,1].min(),reference[:,1].max())
    ax.set_axis_off()
    return artist


@torch.no_grad()
def render(base, *, device="cuda:2"):
    base = Path(base)
    out = base/"figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    runs = []
    for p in sorted((base/"test").glob("*/run_manifest.json")):
        meta, rows, summary = load_run(p.parent)
        runs.append((p.parent, meta, rows, summary))
    full_root = next((base/"fullmesh").glob("*/run_manifest.json")).parent
    full_meta, full_rows, full_summary = load_run(full_root)
    reps = json.loads((base/"representatives.json").read_text())
    runtime = CarRuntime(full_meta["operator_manifest"])
    operator = runtime.load_operator(device)
    visual_manifest = {"representatives": reps, "figures": [], "seed": 0, "runs": {str(p): m["run_id"] for p, m, _, _ in runs}}
    for target in ("field", "cd"):
        target_label = "Cd" if target == "cd" else "field"
        chosen = [r for r in runs if r[1]["config"]["method"] == "uniform" or r[1]["config"]["selection_target"] == target]
        chosen.sort(key=lambda r: ("uniform", "pi", "cpg", "ca").index(r[1]["config"]["method"]))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)
        for ax, metric, label in zip(axes, ("field_error", "cd_relative_error"), ("Full-field error", "Cd relative error")):
            for path, meta, rows, summary in chosen:
                method = meta["config"]["method"]
                mean = [summary["budgets"][f"{b:.3f}"]["mean_"+metric] for b in BUDGETS]
                std = [summary["budgets"][f"{b:.3f}"]["std_"+metric] for b in BUDGETS]
                ax.errorbar(np.asarray(BUDGETS)*100, mean, yerr=std, color=COLORS[method],
                            marker=MARKERS[method], capsize=3, label=method.upper() if method != "uniform" else "Uniform")
            ax.axhline(full_summary["budgets"]["1.000"]["mean_"+metric], color="#999999", ls="--", label="Fullmesh")
            ax.set(xlabel="Final point budget (%)", ylabel=label, xticks=[37.5,50,75])
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"ShapeNet-Car: configurations selected for {target_label}; 100 cars × 3 sampling seeds")
        save(fig, out/f"curves_{target}", vector=True)
        visual_manifest["figures"].append(f"curves_{target}.png")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
        stages = [("coarse_seconds", "head_seconds", "feature_head_seconds"), ("score_seconds", "refinement_seconds"),
                  ("final_inference_seconds",), ("reconstruction_seconds", "cd_seconds"), ("geometry_seconds",)]
        stage_names = ["Acquisition", "Score + sampling", "Final model", "Reconstruction + Cd", "Geometry"]
        for ax, budget in zip(axes, BUDGETS):
            lower = np.zeros(4)
            for keys, name, color in zip(stages, stage_names, ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#999999")):
                values = np.array([sum(r[3]["budgets"][f"{budget:.3f}"].get("mean_"+k,0.) for k in keys) for r in chosen])
                ax.bar(np.arange(4), values, bottom=lower, color=color, label=name)
                lower += values
            ax.set(xticks=np.arange(4), xticklabels=["Uniform", "PI", "CPG", "CA"],
                   title=f"Budget {budget:.1%}", ylabel="Accounted time (s)")
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Standalone cost accounting; {target_label} selection; shared GPU workload")
        save(fig, out/f"runtime_{target}", vector=True)
        visual_manifest["figures"].append(f"runtime_{target}.png")
        for sid in reps["sample_ids"]:
            folder = out/("sample_"+sid.replace("/", "_"))
            folder.mkdir(exist_ok=True)
            item = runtime.sample(sid)
            pos, mask, gt = item.pos.numpy(), item.surface_mask.numpy(), item.target_phys.numpy()
            full = runtime.output_normalizer.decode(operator(item.pos[None].to(device), item.x_norm[None].to(device))[0].cpu()).numpy()
            slab = np.abs(pos[:, 1]-np.median(pos[:, 1])) <= .04*np.ptp(pos[:, 1])
            if not slab.any():
                raise ValueError("empty visualization slab")
            if target == "field":
                fig = plt.figure(figsize=(9, 3), constrained_layout=True)
                base_norm = Normalize(min(float(gt[mask,3].min()),float(full[mask,3].min())),
                                      max(float(gt[mask,3].max()),float(full[mask,3].max())))
                for j, (values, title) in enumerate(((item.x_norm[:, 3].numpy(), "Input SDF (normalized)"),
                                                    (gt[:, 3], "GT surface pressure"), (full[:, 3], "Fullmesh surface pressure"))):
                    ax = fig.add_subplot(1, 3, j+1, projection="3d")
                    select = mask if j else np.ones(len(pos), dtype=bool)
                    plot = surface_view(ax, pos[select], values[select], norm=base_norm if j else None)
                    ax.set_title(title); fig.colorbar(plot, ax=ax, shrink=.65)
                save(fig, folder/"input_gt_fullmesh")
                visual_manifest["figures"].append(str((folder/"input_gt_fullmesh.png").relative_to(out)))
            for budget in BUDGETS:
                snapshots = []
                for path, meta, rows, summary in chosen:
                    row = next(r for r in rows if r["sample_id"] == sid and r["seed"] == 0 and r["budget_fraction"] == budget)
                    with np.load(path/row["arrays_path"]) as z:
                        snapshots.append((meta["config"]["method"], {k:z[k].copy() for k in z.files}, row))
                all_fields = [("GT", gt), ("Fullmesh", full)] + [("Uniform" if m == "uniform" else m.upper(), a["prediction"]) for m,a,_ in snapshots]
                pnorm = Normalize(float(min(v[mask,3].min() for _,v in all_fields)), float(max(v[mask,3].max() for _,v in all_fields)))
                emax = max(float(np.abs(v[mask,3]-gt[mask,3]).max()) for _,v in all_fields)
                fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)
                for j, (name, field) in enumerate(all_fields):
                    ax = fig.add_subplot(2, 6, j+1, projection="3d")
                    plot = surface_view(ax, pos[mask], field[mask,3], norm=pnorm)
                    ax.set_title(name)
                    if j==5: fig.colorbar(plot, ax=ax, shrink=.6, label="Pressure")
                    ax = fig.add_subplot(2, 6, j+7, projection="3d")
                    plot = surface_view(ax, pos[mask], np.abs(field[mask,3]-gt[mask,3]), cmap="magma", norm=Normalize(0,max(emax,1e-12)))
                    if j==5: fig.colorbar(plot, ax=ax, shrink=.6, label="Absolute error")
                fig.suptitle(f"Surface pressure | {target_label} selection | budget {budget:.1%} | seed 0")
                filename=f"pressure_{target}_{budget:.3f}"
                save(fig, folder/filename)
                fig, axes=plt.subplots(2,6,figsize=(15,5.5),constrained_layout=True)
                speed = [np.linalg.norm(v[slab,:3],axis=1) for _,v in all_fields]
                vmax=max(float(v.max()) for v in speed)
                err=[np.linalg.norm(v[slab,:3]-gt[slab,:3],axis=1) for _,v in all_fields]
                emax=max(float(v.max()) for v in err)
                for j, ((name,_), value,error) in enumerate(zip(all_fields,speed,err)):
                    for row, vals, cmap, lim in ((0,value,"viridis",vmax),(1,error,"magma",emax)):
                        plot=axes[row,j].scatter(pos[slab,2],pos[slab,0],c=vals,s=2,cmap=cmap,vmin=0,vmax=max(lim,1e-12),rasterized=True)
                        axes[row,j].set_aspect("equal");axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
                        if row==0: axes[row,j].set_title(name)
                        if j==5: fig.colorbar(plot,ax=axes[row,j],shrink=.7)
                fig.suptitle(f"Velocity magnitude / vector error, central-y slab | {target_label} | {budget:.1%}")
                save(fig,folder/f"velocity_{target}_{budget:.3f}")
                fig=plt.figure(figsize=(11,8),constrained_layout=True)
                density_logs=[np.log10(a["sampling_density"]/np.mean(a["sampling_density"])) for _,a,_ in snapshots]
                density_norm=Normalize(min(float(x.min()) for x in density_logs),max(float(x.max()) for x in density_logs))
                score_logs=[np.log10(np.maximum(a["score"],1e-12)) for _,a,_ in snapshots if "score" in a]
                score_norm=Normalize(min(float(x.min()) for x in score_logs),max(float(x.max()) for x in score_logs))
                for j,(method,arrays,row) in enumerate(snapshots):
                    idx=arrays["indices"]
                    ax=fig.add_subplot(3,4,j+1,projection="3d")
                    surface_view(ax,pos[idx],mask[idx].astype(float),cmap="cividis",size=.5,bounds=pos)
                    label = "Uniform" if method == "uniform" else method.upper()
                    ax.set_title(f"{label}: {len(idx)} points\n{int(mask[idx].sum())} surface")
                    if "coarse_indices" in arrays:
                        cp=pos[arrays["coarse_indices"]]
                        ax.scatter(cp[:,0],cp[:,2],cp[:,1],s=.5,c="#0072B2",alpha=.3)
                    ax=fig.add_subplot(3,4,j+5,projection="3d")
                    score=arrays.get("score")
                    if score is not None:
                        plot=surface_view(ax,pos,np.log10(np.maximum(score,1e-12)),norm=score_norm,size=.6)
                        fig.colorbar(plot,ax=ax,shrink=.6)
                        ax.set_title("Acquisition score\n(log10)")
                    elif method == "pi":
                        config=next(m["config"] for _,m,_,_ in chosen if m["config"]["method"]=="pi")
                        distance=car_surface_distance(pos,mask)
                        prior=(distance<config["r"]).astype(float)
                        plot=surface_view(ax,pos,prior,cmap="cividis",norm=Normalize(0,1),size=.6)
                        ax.set_title(f"PI geometric band\n(distance < {config['r']:g})")
                        fig.colorbar(plot,ax=ax,shrink=.6,ticks=[0,1])
                    else:
                        ax.text2D(.2,.5,"No acquisition score",transform=ax.transAxes,fontsize=9);ax.set_axis_off()
                        ax.set_title("Acquisition score")
                    ax=fig.add_subplot(3,4,j+9,projection="3d")
                    plot=surface_view(ax,pos[idx],density_logs[j],norm=density_norm,size=.6,bounds=pos)
                    ax.set_title("Relative density\n(log10, mean=1)");fig.colorbar(plot,ax=ax,shrink=.6)
                fig.suptitle(f"Sampling, score and density | {target_label} selection | {budget:.1%}; blue dots indicate coarse points")
                save(fig,folder/f"sampling_{target}_{budget:.3f}")
                visual_manifest["figures"].extend(str(p.relative_to(out)) for p in folder.glob(f"*_{target}_{budget:.3f}.png"))
    (out/"visualization_manifest.json").write_text(json.dumps(visual_manifest,indent=2)+"\n")
    return visual_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default=str(ROOT/"results/shapenet_car/budget10_v1"))
    p.add_argument("--device",default="cuda:2")
    render(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
