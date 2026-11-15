"""Reproduce Darcy figures from public scalar CSVs without model/data access."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODELS = ["Transolver3", "LinearNO_point_temp"]
MODEL_NAMES = {"Transolver3": "Transolver3", "LinearNO_point_temp": "LinearNO point-temp",
               "LinearNO_conv_temp": "LinearNO conv-temp"}
METHODS = ["uniform", "pi", "cpg_beta0", "cpg", "ca"]
LABELS = {"uniform": "Uniform", "pi": "PI", "cpg_beta0": "CPG β0", "cpg": "CPG β1", "ca": "CA", "full": "Full"}
COLORS = {"uniform": "#777777", "pi": "#0072B2", "cpg_beta0": "#D55E00", "cpg": "#D55E00", "ca": "#009E73", "full": "#222222"}
MARKERS = {"uniform": "o", "pi": "s", "cpg_beta0": "v", "cpg": "^", "ca": "D", "full": "*"}
NEW = "grid_qhull_sparse_apply_v1p1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data, out = args.data_dir.resolve(), (args.output_dir or args.data_dir / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    qa = []

    def read(name):
        with (data / name).open() as stream:
            return list(csv.DictReader(stream))

    def save(fig, name, rows, caption):
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(out / (name + "." + extension), bbox_inches="tight", dpi=300)
        with (out / (name + "_source.csv")).open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        (out / (name + ".caption.md")).write_text(caption + "\n")
        with Image.open(out / (name + ".png")) as image:
            image.load()
            qa.append({"figure": name, "pixels": list(image.size), "dpi": image.info.get("dpi"), "decode": "passed", "source_rows": len(rows)})
        plt.close(fig)

    ratios = [r for r in read("backbone_rauec.csv") if r["reconstruction_id"] == NEW]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained", sharey=True)
    for ax, model in zip(axes, MODELS):
        rows = [next(r for r in ratios if r["backbone"] == model and r["method"] == method) for method in METHODS]
        x = np.arange(len(rows))
        for i, row in enumerate(rows):
            ax.errorbar(i, float(row["mean"]), yerr=float(row["sampling_sd"]), marker=MARKERS[row["method"]],
                        color=COLORS[row["method"]], capsize=4, ls="none", markersize=7)
        ax.axhline(1, ls="--", lw=.8, color="#999999")
        ax.set(xticks=x, xticklabels=[LABELS[m] for m in METHODS], title=MODEL_NAMES[model], ylim=(0, 4.65))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("rAUEC / matched Uniform")
    save(fig, "darcy_rauec", ratios,
         "Darcy 85×85, matched 900/100 training/validation cohort, training seed 0. Each point is the mean of three sampling-seed area ratios; error bars are sampling sample SD. Case-mean error at budgets 37.5%,50%,75% is integrated separately per sampling seed and divided by the same model/checkpoint/reconstruction Uniform area. Full is outside the integral. Both unfavorable CPG beta0 results are retained; beta0/beta1 share coarse predictions and selected indices. These normalized ratios do not rank absolute accuracy across backbones. Conv-temp is Full-only and has no rAUEC.")

    absolute = [r for r in read("backbone_absolute_errors.csv") if r["reconstruction_id"] == NEW]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, model in zip(axes[:2], MODELS):
        for method in METHODS:
            rows = sorted([r for r in absolute if r["backbone"] == model and r["method"] == method], key=lambda r: float(r["budget_fraction"]))
            ax.errorbar([100*float(r["budget_fraction"]) for r in rows], [float(r["mean_error"]) for r in rows],
                        yerr=[float(r["sampling_sd"]) for r in rows], color=COLORS[method], marker=MARKERS[method],
                        ls="--" if method == "cpg_beta0" else "-", capsize=3, lw=1.1, label=LABELS[method])
        full = next(r for r in absolute if r["backbone"] == model and r["method"] == "full")
        ax.axhline(float(full["mean_error"]), color="#222222", ls=":", label="Full reference")
        ax.set(title=MODEL_NAMES[model], xlabel="Retained budget (%)", xticks=[37.5, 50, 75])
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Full-field relative L2 error")
    axes[1].legend(frameon=False, fontsize=8, ncols=2)
    full_rows = [next(r for r in absolute if r["backbone"] == m and r["method"] == "full") for m in MODELS + ["LinearNO_conv_temp"]]
    axes[2].bar(range(3), [float(r["mean_error"]) for r in full_rows], color=["#777777", "#0072B2", "#CC79A7"], width=.55)
    axes[2].set(xticks=range(3), xticklabels=["T3", "LinearNO\npoint", "LinearNO\nconv"], title="Full 100% references", ylabel="Relative L2 error")
    axes[2].grid(axis="y", alpha=.17)
    save(fig, "darcy_absolute_error_budget", absolute,
         "Absolute full-field relative-L2 errors on all 200 historically visible smooth2 cases. Adaptive curves average cases and then sampling seeds; error bars are sample SD across sampling seeds, not training seeds. Dotted lines and the right panel show each checkpoint's Full 100% error. The native convolution model is evaluated only on the complete 85×85 grid. No unfavorable beta0 points are omitted.")

    latency = read("latency_overview.csv")
    joined = []
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), layout="constrained", sharey=True)
    for ax, model in zip(axes, MODELS):
        for method in ["uniform", "pi", "cpg", "ca"]:
            rows = sorted([r for r in latency if r["backbone"] == model and r["method"] == method
                           and r["scope"] == "A_cpu_cpu" and r["implementation"] == "new"], key=lambda r: float(r["budget_fraction"]))
            plot = []
            for row in rows:
                error = next(r for r in absolute if r["backbone"] == model and r["method"] == method and r["budget_fraction"] == row["budget_fraction"])
                plot.append({**row, "mean_error": error["mean_error"], "sampling_error_sd": error["sampling_sd"], "accuracy_cases": 200})
            joined.extend(plot)
            ax.plot([float(r["median_condition_median_ms"]) for r in plot], [float(r["mean_error"]) for r in plot],
                    color=COLORS[method], marker=MARKERS[method], label=LABELS[method], lw=1.1)
        full_time = next(r for r in latency if r["backbone"] == model and r["method"] == "full" and r["scope"] == "A_cpu_cpu")
        full_error = next(r for r in absolute if r["backbone"] == model and r["method"] == "full")
        ax.scatter(float(full_time["median_condition_median_ms"]), float(full_error["mean_error"]), marker="*", s=100, color="#222222", label="Full")
        joined.append({**full_time, "mean_error": full_error["mean_error"], "sampling_error_sd": "", "accuracy_cases": 200})
        ax.set(title=MODEL_NAMES[model], xlabel="Direct complete CPU-field latency (ms)", xscale="log")
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Full-field relative L2 error")
    axes[1].legend(frameon=False, ncols=2)
    save(fig, "darcy_error_latency", joined,
         "Accuracy versus directly measured complete CPU-output latency (scene A). The network is eager FP32 and the complete CPU field is float64. Accuracy uses all 200 cases; latency uses the three fixed cases and three sampling seeds, with 20 warmups and 100 formal repeats per condition. Each latency coordinate is the median of the case/sampling condition medians, not a sum of profiled stages. Curves connect budgets 37.5%,50%,75%. Full is the measured full-grid reference without sampling replicas. CPG beta0 is absent only from this timing plot because it was not separately timed; its accuracy is retained in the other figures.")

    comparison = [r for r in read("reconstruction_summary.csv") if float(r["budget_fraction"]) == .5]
    scopes = ["A_cpu_cpu", "B_gpu_gpu", "C_cpu_cpu_cached_map"]
    scope_names = ["A · CPU to CPU", "B · GPU to GPU", "C · fixed-map reuse"]
    categories = [(m, method) for m in MODELS for method in ["uniform", "pi", "cpg", "ca"]]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.8), layout="constrained")
    cost_rows = []
    for col, (scope, title) in enumerate(zip(scopes, scope_names)):
        for row_index, (old_key, new_key, ylabel) in enumerate([
                ("old_direct_total_ms", "new_direct_total_ms", "Direct request median (ms)"),
                ("old_reconstruction_ms", "new_reconstruction_ms", "Reconstruction profile (ms)")]):
            old_values, new_values = [], []
            for model, method in categories:
                rows = [r for r in comparison if r["backbone"] == model and r["method"] == method and r["scope"] == scope]
                assert len(rows) == 9
                old, new = float(np.median([float(r[old_key]) for r in rows])), float(np.median([float(r[new_key]) for r in rows]))
                old_values.append(old); new_values.append(new)
                cost_rows.append({"backbone": model, "method": method, "budget_fraction": .5, "scope": scope,
                                  "quantity": "direct_request" if row_index == 0 else "separate_reconstruction_profile",
                                  "old_median_over_conditions_ms": old, "new_median_over_conditions_ms": new,
                                  "condition_count": 9, "training_seed": 0})
            ax = axes[row_index, col]; x = np.arange(len(categories))
            ax.bar(x-.18, old_values, width=.36, color="#999999", label="Old")
            ax.bar(x+.18, new_values, width=.36, color="#0072B2", label="New")
            ax.set(title=title, ylabel=ylabel, xticks=x,
                   xticklabels=[("T3" if m == "Transolver3" else "LNO")+"\n"+LABELS[method] for m, method in categories])
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=.16)
    axes[0, 0].legend(frameon=False, ncols=2)
    save(fig, "darcy_reconstruction_cost", cost_rows,
         "Matched old/new costs at the fixed 50% budget, for both point models and four timed sampling methods. Bars are medians over the same three cases×three sampling conditions. Upper row: directly measured request medians. Lower row: separate reconstruction profiling, including required transfers; these values are not added to form the upper row. A and B use fresh reconstruction maps; C explicitly reuses the same geometry/selected-index map, with its first cost recorded separately in the tables. C's old implementation is cached CPU sparse-map application, not A's native griddata timing. All slowdowns and near-equalities are retained. Full has no reconstruction-map comparison because its old/new views share one identity request.")
    imgs = []
    for row in qa:
        image = Image.open(out / (row["figure"]+".png")).convert("RGB")
        image.thumbnail((1500, 900)); imgs.append(image)
    sheet = Image.new("RGB", (3000, 1800), "white")
    for i, image in enumerate(imgs): sheet.paste(image, ((i%2)*1500, (i//2)*900))
    sheet.save(out / "contact_sheet.png")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "render_qa.json"]
    (out / "render_qa.json").write_text(json.dumps({"status": "rendered_decoded", "figures": qa,
        "training_seed_count": 1, "sampling_seed_count": 3, "files": files,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    print(json.dumps({"figures": len(qa), "source": "public scalar CSVs only", "GPU": False}))


if __name__ == "__main__":
    main()
