#!/usr/bin/env python3
"""Generate deterministic paper figures and three-rule tables from verified CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLORS = {"uniform": "#666666", "pi": "#0072B2", "cpg": "#D55E00", "ca": "#009E73", "full": "#222222"}
MARKERS = {"uniform": "o", "pi": "s", "cpg": "^", "ca": "D", "full": "*"}
LABELS = {"uniform": "Uniform", "pi": "PI", "cpg": "CPG", "ca": "CA", "full": "Full"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "savefig.dpi": 300, "figure.dpi": 130})
    qa, captions = [], []

    def read(name):
        with (args.data_dir / name).open(newline="") as f:
            return list(csv.DictReader(f))

    def export(fig, name, source, caption):
        fig.canvas.draw()
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(dest / f"{name}.{extension}", bbox_inches="tight", pad_inches=.06)
        with (dest / f"{name}_source.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(source[0]))
            w.writeheader()
            w.writerows(source)
        with Image.open(dest / f"{name}.png") as im:
            im.load()
            qa.append({"figure": name, "png_size": im.size, "png_dpi": im.info.get("dpi"),
                       "decode": "passed", "source_rows": len(source)})
        captions.append(f"**{name}.** {caption}")
        plt.close(fig)

    train = read("training_seed_summary.csv")
    main_train = [r for r in train if r["metric"] == "field" and r["quantity"] == "rAUEC"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9), layout="constrained")
    for i, method in enumerate(["uniform", "pi", "cpg", "ca"]):
        row = next(r for r in main_train if r["method"] == method)
        values = [float(row[f"seed{s}"]) for s in range(3)]
        ax.scatter(np.array([i-.12, i, i+.12]), values, s=22, color=COLORS[method], alpha=.65, zorder=3)
        ax.errorbar(i, float(row["mean"]), yerr=float(row["sample_sd"]), color="#111111", marker="_",
                    markersize=13, capsize=5, linewidth=1.3, zorder=4)
    ax.axhline(1, color="#999999", lw=.8, ls="--")
    ax.set(xticks=range(4), xticklabels=["Uniform", "PI", "CPG", "CA"], ylim=(.50, 1.04),
           ylabel="Field rAUEC (Uniform = 1)", title="ShapeNet-Car · Transolver3 · 3 training seeds")
    ax.grid(axis="y", alpha=.17)
    export(fig, "car_training_seed_rauec", main_train,
           "Field rAUEC for three independently trained Transolver3 models. Colored points are training seeds0/1/2; black marks and error bars show the mean and sample SD across the three models. Each model value averages three matched sampling-seed area ratios over budgets37.5%,50%,75%. CPG and CA use beta1.")

    train_error = [r for r in train if r["metric"] == "field" and r["quantity"] == "absolute_error"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9), layout="constrained")
    for method in ["uniform", "pi", "cpg", "ca"]:
        rows = sorted([r for r in train_error if r["method"] == method], key=lambda r: float(r["budget"]))
        ax.errorbar([100*float(r["budget"]) for r in rows], [float(r["mean"]) for r in rows],
                    yerr=[float(r["sample_sd"]) for r in rows], label=LABELS[method], color=COLORS[method],
                    marker=MARKERS[method], markerfacecolor="white" if method == "ca" else COLORS[method], capsize=3, lw=1.2)
    full = next(r for r in train_error if r["method"] == "full")
    ax.axhline(float(full["mean"]), color="#222222", ls=":", label="Full100%")
    ax.set(xticks=[37.5, 50, 75], xlabel="Retained point budget (%)", ylabel="Absolute field error",
           title="ShapeNet-Car · Transolver3 · training mean ± SD")
    ax.legend(frameon=False, ncols=2)
    ax.grid(alpha=.17)
    export(fig, "car_training_seed_absolute_error", train_error,
           "Absolute field error, defined as velocity relative L2 plus0.5 times surface-pressure relative L2. Curves average cases and sampling seeds within each trained model, then show the mean and sample SD across three independent training seeds. The dotted line is the same three models evaluated at the full point set.")

    rauec = [r for r in read("backbone_rauec.csv") if r["cohort"] == "E3_seed0" and r["metric"] == "field"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained", sharey=True)
    for ax, backbone in zip(axes, ["Transolver3", "LinearNO"]):
        rows = [next(r for r in rauec if r["backbone"] == backbone and r["method"] == m and r["beta"] == b)
                for m, b in [("uniform", ""), ("cpg", "0.0"), ("cpg", "1.0")]]
        for i, row in enumerate(rows):
            ax.errorbar(i, float(row["mean"]), yerr=float(row["sampling_sd"]), color=COLORS[row["method"]],
                        marker="o" if i == 0 else "^", capsize=4, lw=1.3)
        ax.axhline(1, color="#999999", ls="--", lw=.8)
        ax.set(xticks=[0, 1, 2], xticklabels=["Uniform", "CPG beta0", "CPG beta1"],
               title=backbone, ylim=(.45, 1.04))
        ax.grid(axis="y", alpha=.17)
    axes[0].set_ylabel("Field rAUEC (own Uniform = 1)")
    export(fig, "car_backbone_rauec", rauec,
           "Each backbone uses its own same-checkpoint Uniform denominator. Symbols show the mean and sample SD across sampling seeds0/1/2 for training seed0. CPG beta0 is the paired compensation ablation; beta1 is the primary setting. Ratios do not rank absolute accuracy across backbones.")

    errors = [r for r in read("backbone_absolute_errors.csv") if r["cohort"] == "E3_seed0" and r["metric"] == "field"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained", sharey=True)
    for ax, backbone in zip(axes, ["Transolver3", "LinearNO"]):
        for method, beta, label, style in [("uniform", "", "Uniform", "-"), ("cpg", "0.0", "CPG beta0", "--"), ("cpg", "1.0", "CPG beta1", "-")]:
            rows = sorted([r for r in errors if r["backbone"] == backbone and r["method"] == method and r["beta"] == beta], key=lambda r: float(r["budget"]))
            ax.errorbar([100*float(r["budget"]) for r in rows], [float(r["error"]) for r in rows],
                        yerr=[float(r["sampling_sd"]) for r in rows], color=COLORS[method], marker=MARKERS[method],
                        ls=style, label=label, capsize=3, lw=1.2, markerfacecolor="white" if beta == "0.0" else COLORS[method])
        full = next(r for r in errors if r["backbone"] == backbone and r["method"] == "full")
        ax.axhline(float(full["error"]), color="#222222", ls=":", label="Full100%")
        ax.set(xticks=[37.5, 50, 75], xlabel="Retained point budget (%)", title=backbone)
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Absolute field error")
    axes[1].legend(frameon=False, fontsize=7.5)
    export(fig, "car_backbone_absolute_error", errors,
           "Absolute field errors for the two matched training-seed0 models on100 test cases. Error bars show sample SD across the three sampling-seed case means. Full is a same-checkpoint reference at100% budget, excluded from the rAUEC integral.")

    latency = [r for r in read("backbone_accuracy_latency.csv") if r["metric"] == "field"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained", sharey=True)
    for ax, backbone in zip(axes, ["Transolver3", "LinearNO"]):
        for method, beta, label, style in [("uniform", "", "Uniform", "-"), ("cpg", "0.0", "CPG beta0", "--"), ("cpg", "1.0", "CPG beta1", "-")]:
            rows = sorted([r for r in latency if r["backbone"] == backbone and r["method"] == method and r["beta"] == beta], key=lambda r: float(r["budget"]))
            ax.plot([float(r["complete_latency_median_ms"]) for r in rows], [float(r["error"]) for r in rows], color=COLORS[method], marker=MARKERS[method], ls=style, lw=1.2, label=label, markerfacecolor="white" if beta == "0.0" else COLORS[method])
        full = next(r for r in latency if r["backbone"] == backbone and r["method"] == "full")
        ax.scatter(float(full["complete_latency_median_ms"]), float(full["error"]), marker="*", s=75, color="#222222", label="Full100%", zorder=3)
        ax.set(xlabel="Complete CPU field latency (ms)", title=backbone)
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Absolute field error")
    axes[1].legend(frameon=False, fontsize=7.5)
    export(fig, "car_absolute_error_latency", latency,
           "Error uses100 test cases; latency is the directly measured median over3 fixed geometries,3 sampling seeds, and100 formal repeats after20 warmups. Full has one series per geometry. Both axes use the same backbone checkpoint and native reconstruction protocol. The boundary is CPU input to complete CPU float64 output with batch1 and eager FP32 networks. Points on each method curve correspond to37.5%,50%,75% budgets in order.")

    parity = read("reconstruction_parity.csv")
    groups = [("shapenet_car", "Transolver3"), ("shapenet_car", "LinearNO"), ("darcy", "published_Transolver3"), ("airfoil", "published_Transolver3")]
    groups = [g for g in groups if any(r["dataset"] == g[0] and r["backbone"] == g[1] for r in parity)]
    fig, ax = plt.subplots(figsize=(5.0, 2.9), layout="constrained")
    for i, key in enumerate(groups):
        values = [float(r["max_absolute_difference"]) for r in parity if (r["dataset"], r["backbone"]) == key]
        ax.scatter(i + np.linspace(-.15, .15, len(values)), values, s=16, color="#0072B2", alpha=.65)
    ax.set(xticks=range(len(groups)), xticklabels=[("Car" if d == "shapenet_car" else d.title())+"\n"+b.replace("published_", "") for d, b in groups], ylabel="Maximum absolute element difference", title="Old/new reconstruction · fixed predictions")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=.17)
    export(fig, "reconstruction_fixed_prediction_parity", parity,
           "Elementwise reconstruction agreement on108 fixed-prediction validation pairs at3 budgets and3 sampling seeds. Maximum absolute differences are shown per pair; all satisfy atol1e-6 plus rtol1e-5 times the original magnitude. These pairs validate the interpolation implementation on the recorded original checkpoints. Final new-checkpoint complete-request costs remain pending.")

    # Paper-style three-rule table: E5 field cohort only, three independent models.
    table_rows = []
    for method in ["uniform", "pi", "cpg", "ca"]:
        r = next(x for x in main_train if x["method"] == method)
        table_rows.append([LABELS[method]] + [f"{float(r[f'seed{s}']):.4f}" for s in range(3)] + [f"{float(r['mean']):.4f} ± {float(r['sample_sd']):.4f}"])
    fig, ax = plt.subplots(figsize=(6.6, 1.85), layout="constrained")
    ax.axis("off")
    headers = ["Method", "Training seed0", "Training seed1", "Training seed2", "Mean ± sample SD"]
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center", loc="center", colWidths=[.14,.18,.18,.18,.27])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(.8)
            cell.get_text().set_weight("bold")
        if row == len(table_rows):
            cell.visible_edges = "B"
            cell.set_linewidth(.8)
    export(fig, "car_training_seed_rauec_table", main_train,
           "ShapeNet-Car field rAUEC for Transolver3. Each training-seed entry averages the three matched sampling-seed ratios. Mean ± sample SD uses three independently trained models. CPG and CA use beta1; all ratios integrate37.5%,50%,75% budgets only.")
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method & Seed 0 & Seed 1 & Seed 2 & Mean $\\pm$ sample SD " + chr(92) * 2, "\\midrule"]
    for row in table_rows:
        lines.append(" & ".join(x.replace("±", "$\\pm$") for x in row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (dest / "car_training_seed_rauec_table.tex").write_text("\n".join(lines) + "\n")
    (dest / "captions.md").write_text("\n\n".join(captions) + "\n")
    (dest / "render_qa.json").write_text(json.dumps({"figures": qa, "renderer": matplotlib.__version__, "pdf_fonttype": 42, "svg_text": "editable", "status": "decode_passed; visual inspection recorded separately"}, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
