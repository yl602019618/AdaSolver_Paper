#!/usr/bin/env python3
"""Airfoil seed0 figures from CSV only; all uncertainty is sampling SD."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

COLORS = {"uniform": "#666666", "pi": "#0072B2", "cpg": "#D55E00", "ca": "#009E73"}
MARKERS = {"uniform": "o", "pi": "s", "cpg": "^", "ca": "D"}
LABELS = {"uniform": "Uniform", "pi": "PI", "cpg": "CPG", "ca": "CA"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300})
    qa = []

    def read(name):
        with (args.data_dir / name).open() as f:
            return list(csv.DictReader(f))

    def export(fig, name, rows):
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(args.output_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=.06)
        with (args.output_dir / f"{name}_source.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with Image.open(args.output_dir / f"{name}.png") as im:
            im.load()
            qa.append({"figure": name, "png_decode": "passed", "size": im.size, "dpi": im.info.get("dpi"), "source_rows": len(rows)})
        plt.close(fig)

    rauec = [r for r in read("backbone_rauec.csv") if r["metric"] == "field_relative_l2"
             and r["reconstruction_id"] == "grid_qhull_sparse_apply_v1p1" and r["beta"] != "0.0"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained", sharey=True)
    for ax, backbone in zip(axes, ["Transolver3", "LinearNO"]):
        for i, method in enumerate(["uniform", "pi", "cpg", "ca"]):
            r = next(r for r in rauec if r["backbone"] == backbone and r["method"] == method)
            ax.errorbar(i, float(r["mean"]), yerr=float(r["sampling_sd"]), color=COLORS[method],
                        marker=MARKERS[method], capsize=4, lw=1.3)
        ax.axhline(1, color="#999999", ls="--", lw=.8)
        ax.set(xticks=range(4), xticklabels=["Uniform", "PI", "CPG", "CA"], title=backbone)
        ax.grid(axis="y", alpha=.17)
    axes[0].set_ylabel("Pressure rAUEC (own Uniform = 1)")
    export(fig, "airfoil_seed0_rauec", rauec)

    errors = [r for r in read("backbone_absolute_errors.csv") if r["metric"] == "field_relative_l2"
              and r["reconstruction_id"] == "grid_qhull_sparse_apply_v1p1" and r["beta"] != "0.0"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), layout="constrained", sharey=True)
    for ax, backbone in zip(axes, ["Transolver3", "LinearNO"]):
        for method in ["uniform", "pi", "cpg", "ca"]:
            rows = sorted([r for r in errors if r["backbone"] == backbone and r["method"] == method], key=lambda r: float(r["budget"]))
            ax.errorbar([100*float(r["budget"]) for r in rows], [float(r["error"]) for r in rows],
                        yerr=[float(r["sampling_sd"]) for r in rows], label=LABELS[method], color=COLORS[method],
                        marker=MARKERS[method], markerfacecolor="white" if method == "ca" else COLORS[method], capsize=3, lw=1.2)
        full = next(r for r in errors if r["backbone"] == backbone and r["method"] == "full")
        ax.axhline(float(full["error"]), color="#222222", ls=":", label="Full100%")
        ax.set(xticks=[37.5, 50, 75], xlabel="Retained point budget (%)", title=backbone)
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Absolute pressure relative L2 error")
    axes[1].legend(frameon=False, ncols=2, fontsize=7.5)
    export(fig, "airfoil_seed0_absolute_error", errors)

    table_rows = []
    for method in ["uniform", "pi", "cpg", "ca"]:
        row = [LABELS[method]]
        for backbone in ["Transolver3", "LinearNO"]:
            r = next(r for r in rauec if r["backbone"] == backbone and r["method"] == method)
            row.append(f"{float(r['mean']):.4f} ± {float(r['sampling_sd']):.4f}")
        table_rows.append(row)
    fig, ax = plt.subplots(figsize=(5.2, 1.9), layout="constrained")
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=["Method", "Transolver3", "LinearNO"], cellLoc="center", loc="center", colWidths=[.22,.38,.38])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(.8)
            cell.get_text().set_weight("bold")
        elif row == len(table_rows):
            cell.visible_edges = "B"
            cell.set_linewidth(.8)
    export(fig, "airfoil_seed0_rauec_table", rauec)
    lines = [r"\begin{tabular}{lrr}", r"\toprule", "Method & Transolver3 & LinearNO " + chr(92)*2, r"\midrule"]
    lines += [" & ".join(v.replace("±", r"$\pm$") for v in row) + " " + chr(92)*2 for row in table_rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (args.output_dir / "airfoil_seed0_rauec_table.tex").write_text("\n".join(lines)+"\n")
    caption = """**Airfoil seed0 rAUEC and table.** Pressure rAUEC for one independently trained model per backbone on the published110 positive-lift cases. Mean and sample SD are across sampling seeds0/1/2. Each sampling seed integrates budgets37.5%,50%,75% and divides by its matched same-backbone Uniform area. CPG uses beta1. Full100% is excluded from the integral. The pressure task uses the official LinearNO point-temp variant with its corresponding head.

**Airfoil seed0 absolute error.** Pressure relative L2 on the221×51 evaluation grid. Curves show110-case means and sample SD across sampling seeds0/1/2. Dotted lines show same-checkpoint full-point errors. All curves use the same accepted linear interpolation and nearest fallback rule, with sparse GPU application and float64 complete outputs.

Both figures use training seed0 only. The historical positive-lift population is fixed; no never-touched-test claim is made. Full end-to-end latency and training seeds1/2 are separate pending measurements.
"""
    (args.output_dir / "captions.md").write_text(caption)
    (args.output_dir / "render_qa.json").write_text(json.dumps({"status": "PNG_decoded; PDF_render_pending", "figures": qa}, indent=2)+"\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
