"""Evidence-bound full20 curves, sampled densities and temporal field plates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adasolver.inference.plasticity2d_contract import BUDGETS, write_json
from adasolver.inference.plasticity2d_sampling import REFERENCE
from adasolver.metrics.plasticity2d import full20_metrics

COLORS = {"uniform": "#666666", "pi": "#E69F00", "cpg": "#0072B2", "ca": "#009E73"}
MARKERS = {"uniform": "o", "pi": "s", "cpg": "^", "ca": "D"}
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, path, vector=False):
    path = Path(path)
    fig.savefig(Path(str(path) + ".png"), dpi=180, bbox_inches="tight")
    if vector:
        fig.savefig(Path(str(path) + ".pdf"), bbox_inches="tight")
        fig.savefig(Path(str(path) + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(comparison, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.7, 4.1), layout="constrained")
    for method, values in comparison["methods"].items():
        ax.plot(np.array(BUDGETS)*100, values["mean_errors"], marker=MARKERS[method],
                color=COLORS[method], label=f"{method.upper() if method!='uniform' else 'Uniform'}  rAUEC={values['rauec']:.3f}")
    ax.set(xlabel="Final spatial budget (%)", ylabel="Full20 field relative L2", xticks=[37.5,50,75])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, output/"budget_curve", vector=True)
    labels = list(comparison["methods"])
    timing = [comparison["methods"][m]["mean_timing"] for m in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, key, title in zip(axes, ("acquisition_seconds", "total_seconds"), ("Network acquisition", "End-to-end request")):
        display_labels = ["Uniform" if m == "uniform" else m.upper() for m in labels]
        ax.bar(display_labels, [x[key]*1000 for x in timing], color=[COLORS[m] for m in labels])
        ax.set(ylabel="Mean time (ms)", title=title)
    save(fig, output/"runtime", vector=True)


def field_plate(fields, frames, path, *, channel=None, error=False, limits=None):
    labels = list(fields)
    arrays = {key: value.reshape(101,31,20,2) for key, value in fields.items()}
    images = {key: np.linalg.norm(value, axis=-1) if channel is None else value[..., channel]
              for key, value in arrays.items()}
    vmax = max(float(np.max(value)) for value in images.values())
    vmin = min(float(np.min(value)) for value in images.values()) if channel is not None else 0.
    cmap = "magma" if error else ("coolwarm" if channel is not None else "viridis")
    if channel is not None and not error:
        vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    if limits is not None:
        vmin, vmax = limits
    fig, axes = plt.subplots(len(labels), len(frames), figsize=(2.15*len(frames), 1.05*len(labels)+.7),
                             squeeze=False, layout="constrained")
    for row, label in enumerate(labels):
        for col, frame in enumerate(frames):
            ax = axes[row,col]
            image = ax.imshow(images[label][:,:,frame].T, origin="upper", aspect="equal",
                              vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Frame {frame}")
            if col == 0:
                ax.set_ylabel(label)
    quantity = "Displacement magnitude" if channel is None else ("u_x" if channel == 0 else "u_y")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.85, pad=.01,
                 label=("Absolute vector error" if error else quantity))
    save(fig, path)


@torch.no_grad()
def visualize(evaluator, configurations, fullmesh, output, comparison=None, *, sample_ids=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(fullmesh["records"], key=lambda r: (r["full_field_relative_l2"], r["sample_id"]))
    if sample_ids is None:
        sample_ids = [ordered[round(q*(len(ordered)-1))]["sample_id"] for q in (.25,.5,.75)]
    split_name = fullmesh["split"]
    split = evaluator.data.splits[split_name]
    manifest = {"sample_ids": sample_ids, "selection": "fullmesh field-error quantiles 25/50/75; ID tie-break",
                "operator_id": evaluator.metadata["checkpoint_sha256"], "seed": 0,
                "frames": list(range(20)), "budgets": list(BUDGETS), "records": []}
    contract = (
        "# Full20 visual contract\n\n"
        "Source: frozen configurations, raw full20 displacement metrics and the same operator.\n"
        "Main figure: three-budget error curves and rAUEC; runtime is reported separately.\n"
        "Qualitative figures: predefined fullmesh-error quantiles; seed 0; all methods and budgets.\n"
        "Field panels use common scales per physical quantity, reference i/j axes and all 20 frames.\n"
        "Key-frame component plates complement all-time displacement magnitude plates.\n"
        "Colors: stable Okabe-Ito method colors plus distinct markers; viridis/magma fields.\n"
        "Formats: PNG field plates; PNG/PDF/SVG curves and runtime.\n"
        "No method ranking is assumed by the plotting code.\n"
    )
    (output/"visual_contract.md").write_text(contract)
    if comparison:
        plot_comparison(comparison, output)
    for sid in sample_ids:
        sample_root = output/f"sample_{sid}"
        sample_root.mkdir(parents=True, exist_ok=True)
        i = split.sample_ids.index(sid)
        evaluator.context.clear()
        full_pred = evaluator.operator(split.pos[i:i+1].to(evaluator.device), split.fx[i:i+1].to(evaluator.device))[0].cpu().numpy()
        truth = split.target[i].numpy()
        profile = evaluator.data.input_normalizer.decode(split.fx[i:i+1])[0,:,0].numpy().reshape(101,31)[:,0]
        fig, ax = plt.subplots(figsize=(6,2.5), layout="constrained")
        ax.plot(np.arange(101), profile, color="#222222")
        ax.set(xlabel="Reference i", ylabel="Input die profile", title=f"Sample {sid}")
        save(fig, sample_root/"input_profile")
        predictions = {(b,method):evaluator.predict(split_name,i,0,b,cfg)
                       for b in range(3) for method,cfg in configurations.items()}
        all_fields = [truth,full_pred]+[item[1]["prediction"] for item in predictions.values()]
        component_limits = [max(float(np.abs(a.reshape(-1,20,2)[...,c]).max()) for a in all_fields) for c in (0,1)]
        magnitude_limit = max(float(np.linalg.norm(a.reshape(-1,20,2),axis=-1).max()) for a in all_fields)
        error_limit = max(float(np.linalg.norm((a-truth).reshape(-1,20,2),axis=-1).max()) for a in all_fields)
        full_rmse = full20_metrics(full_pred,truth)["per_frame_rmse"]
        rmse_limit = max(max(full_rmse),max(max(item[0]["per_frame_rmse"]) for item in predictions.values()))
        for budget_id, budget in enumerate(BUDGETS):
            fields = {"GT": truth, "Fullmesh": full_pred}
            errors, diagnostics = {"Fullmesh":full_pred-truth}, {}
            fig, axes = plt.subplots(4, 3, figsize=(11,7.2), layout="constrained")
            for row, (method, cfg) in enumerate(configurations.items()):
                label = "Uniform" if method == "uniform" else method.upper()
                record, arrays = predictions[budget_id,method]
                fields[label] = arrays["prediction"]
                errors[label] = arrays["prediction"]-truth
                diagnostics[method] = record["per_frame_rmse"]
                indices = arrays["indices"]
                xy = REFERENCE[indices]
                axes[row,0].scatter(xy[:,0], xy[:,1], s=1.5, color=COLORS[method])
                if arrays["coarse"] is not None:
                    cp = REFERENCE[arrays["coarse"]]
                    axes[row,0].scatter(cp[:,0], cp[:,1], s=3, color="#222222")
                for col, (name, values) in enumerate((("Density at selected points", arrays["density"]),
                                                     ("Compensation", arrays["compensation"])), start=1):
                    image = axes[row,col].scatter(xy[:,0], xy[:,1], c=values, s=3, cmap="viridis")
                    fig.colorbar(image, ax=axes[row,col], shrink=.65)
                    if row == 0:
                        axes[row,col].set_title(name)
                axes[row,0].set_ylabel(f"{label}\nN={len(indices)}")
                if row == 0:
                    axes[row,0].set_title("Selected points; coarse in black")
                for ax in axes[row]:
                    ax.set_aspect("equal"); ax.set_xlim(0,100); ax.set_ylim(30,0)
                    ax.set_xticks([]); ax.set_yticks([])
                manifest["records"].append({k:record[k] for k in ("sample_id","seed","method","budget_fraction",
                    "selected_count","coarse_count","selected_indices_hash","interpolation_fallback_count","config_id")})
            tag = f"budget_{budget*100:g}"
            save(fig, sample_root/f"{tag}_sampling")
            for channel in (0,1):
                field_plate(fields, [0,1,5,10,15,19], sample_root/f"{tag}_u{channel}", channel=channel,
                            limits=(-component_limits[channel],component_limits[channel]))
            field_plate(fields, list(range(20)), sample_root/f"{tag}_all20",limits=(0,magnitude_limit))
            field_plate(errors, [0,1,5,10,15,19], sample_root/f"{tag}_error", error=True,limits=(0,error_limit))
            np.savez_compressed(sample_root/f"{tag}_fields.npz", **fields)
            fig, ax = plt.subplots(figsize=(6,3.5), layout="constrained")
            ax.plot(range(20),full_rmse,color="#222222",linestyle="--",label="Fullmesh")
            for method, values in diagnostics.items():
                ax.plot(range(20), values, label="Uniform" if method == "uniform" else method.upper(), color=COLORS[method], marker=MARKERS[method], markersize=3)
            ax.set(xlabel="Frame", ylabel="Displacement RMSE", title=f"Sample {sid}; budget {budget*100:g}%")
            ax.set_ylim(0,max(rmse_limit*1.05,1e-12))
            ax.legend(frameon=False); ax.grid(alpha=.2)
            save(fig, sample_root/f"{tag}_time_error")
    write_json(output/"manifest.json", manifest)
    return manifest
