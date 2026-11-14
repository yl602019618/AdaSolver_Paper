"""Render frozen Airfoil score/selection records; never loads a model or truth data source."""
from pathlib import Path
import hashlib
import json
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
METHODS = ["uniform", "pi", "cpg", "ca"]
LABELS = {"uniform": "Uniform", "pi": "PI", "cpg": "CPG", "ca": "CA"}
XLIM, YLIM = (-.3, 1.3), (-.45, .45)


def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    ROOT = args.data_dir.resolve()
    OUT = args.output_dir.resolve() if args.output_dir else ROOT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    data = {}
    for method in METHODS:
        table = np.genfromtxt(ROOT / f"{method}_points.csv", delimiter=",", names=True, dtype=float)
        d = {name: table[name] for name in table.dtype.names}
        d["coords"] = np.column_stack((d["x"], d["y"]))
        for mask in ["selected", "coarse", "mandatory", "eligible"]:
            d[mask] = d[mask].astype(bool)
        d["indices"] = np.flatnonzero(d["selected"])
        d["coarse_indices"] = np.flatnonzero(d["coarse"])
        data[method] = d
    coords = data["uniform"]["coords"]
    visible = ((coords[:, 0] >= XLIM[0]) & (coords[:, 0] <= XLIM[1]) &
               (coords[:, 1] >= YLIM[0]) & (coords[:, 1] <= YLIM[1]))
    body = coords[data["uniform"]["mandatory"]]
    qa = []

    def finish_axes(ax):
        ax.fill(body[:, 0], body[:, 1], facecolor="white", edgecolor="#222222", linewidth=.7, zorder=8)
        ax.set(xlim=XLIM, ylim=YLIM, aspect="equal", xlabel="x", ylabel="y")
        ax.tick_params(labelsize=8, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    def field(ax, values, norm, cmap="viridis"):
        valid = visible & np.isfinite(values)
        if isinstance(norm, LogNorm):
            valid &= values > 0
        artist = ax.scatter(coords[valid, 0], coords[valid, 1], c=values[valid], s=4,
                            cmap=cmap, norm=norm, linewidths=0, rasterized=False)
        finish_axes(ax)
        return artist

    def save(fig, name, caption, source):
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
        (OUT / f"{name}.caption.md").write_text(caption + "\n")
        with Image.open(OUT / f"{name}.png") as im:
            im.load()
            qa.append({"figure": name, "png_dimensions": list(im.size), "dpi": im.info.get("dpi"),
                       "decoded": True, "source": source})
        plt.close(fig)

    positive = np.concatenate([data[m]["proposal_probability"][data[m]["proposal_probability"] > 0] for m in METHODS])
    proposal_norm = LogNorm(float(positive.min()), float(positive.max()))
    raw_norm = Normalize(0, max(float(data[m]["raw_score"].max()) for m in ["cpg", "ca"]))
    score_norm = Normalize(0, max(float(data[m]["normalized_score"].max()) for m in ["cpg", "ca"]))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), layout="constrained")
    for row, method in enumerate(["cpg", "ca"]):
        for col, key, norm, title in [(0, "raw_score", raw_norm, "Raw pressure-gradient score"),
                                     (1, "normalized_score", score_norm, "Score / q95"),
                                     (2, "proposal_probability", proposal_norm, "Extra-point proposal probability")]:
            artist = field(axes[row, col], data[method][key], norm)
            axes[row, col].set_title(f"{LABELS[method]} · {title}")
            fig.colorbar(artist, ax=axes[row, col], shrink=.67, pad=.02)
    save(fig, "airfoil_score_stages",
         "Airfoil case1001, Transolver3 training seed0, sampling seed0, budget50%. Raw scores are the actual LSQ gradients of CPG coarse pressure or the matching CA head pressure. The next stage is the original q95 normalization, without clipping. Proposal probability uses the actual gamma-powered eligible-point law; its logarithmic color scale excludes already-forced base points. Full per-point values are retained outside the displayed physical-coordinate window. No ground truth enters these stages.",
         ["cpg_points.csv", "ca_points.csv"])

    xedges, yedges = np.linspace(*XLIM, 49), np.linspace(*YLIM, 25)
    native, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=(xedges, yedges))
    bin_data = {"xedges": xedges, "yedges": yedges, "native_count": native}
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 9.6), layout="constrained")
    for row, method in enumerate(METHODS):
        d = data[method]
        artist = field(axes[row, 0], d["proposal_probability"], proposal_norm)
        axes[row, 0].set_title(f"{LABELS[method]} · proposal probability")
        fig.colorbar(artist, ax=axes[row, 0], shrink=.66, pad=.02)
        selected_xy = coords[d["indices"]]
        selected, _, _ = np.histogram2d(selected_xy[:, 0], selected_xy[:, 1], bins=(xedges, yedges))
        fraction = np.divide(selected, native, out=np.full_like(native, np.nan), where=native > 0)
        bin_data[method + "_selected_count"] = selected
        bin_data[method + "_selected_fraction"] = fraction
        artist = axes[row, 1].pcolormesh(xedges, yedges, fraction.T, vmin=0, vmax=1, cmap="cividis", shading="flat")
        finish_axes(axes[row, 1])
        axes[row, 1].set_title(f"{LABELS[method]} · selected / native count")
        fig.colorbar(artist, ax=axes[row, 1], shrink=.66, pad=.02)
        axes[row, 2].scatter(coords[visible, 0], coords[visible, 1], s=2, c="#DDDDDD", linewidths=0)
        mask = visible & d["selected"]
        axes[row, 2].scatter(coords[mask, 0], coords[mask, 1], s=3, c="#222222", linewidths=0)
        coarse = visible & d["coarse"]
        retained_coarse = coarse & d["selected"]
        omitted_coarse = coarse & ~d["selected"]
        if retained_coarse.any():
            axes[row, 2].scatter(coords[retained_coarse, 0], coords[retained_coarse, 1],
                                 s=10, c="#D55E00", linewidths=0, label="Retained coarse")
        if omitted_coarse.any():
            axes[row, 2].scatter(coords[omitted_coarse, 0], coords[omitted_coarse, 1],
                                 s=13, c="#D55E00", marker="x", linewidths=.6, label="Omitted coarse")
        if coarse.any():
            axes[row, 2].legend(loc="upper right", fontsize=7, frameon=False, markerscale=1.2)
        finish_axes(axes[row, 2])
        axes[row, 2].set_title(f"{LABELS[method]} · actual selected points")
    with (OUT / "density_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_lower", "x_upper", "y_lower", "y_upper", "native_count"] +
                        [m + "_selected_count" for m in METHODS] + [m + "_selected_fraction" for m in METHODS])
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                writer.writerow([xedges[i], xedges[i+1], yedges[j], yedges[j+1], native[i, j]] +
                    [bin_data[m+"_selected_count"][i, j] for m in METHODS] +
                    ["" if not np.isfinite(bin_data[m+"_selected_fraction"][i, j]) else bin_data[m+"_selected_fraction"][i, j] for m in METHODS])
    save(fig, "airfoil_sampling_comparison",
         "Same fixed case and budget across Uniform, PI, CPG and CA. Left: actual eligible extra-point proposal probabilities (log scale); these are not marginal inclusion probabilities. Middle: selected/native count in identical48×24 physical-coordinate bins. Right: black retained points, orange filled retained-coarse points, and orange crosses for coarse points omitted from the final set. The locked weighted CPG configuration retains619/1127 coarse points; weighted_nested CA retains1127/1127. Thus this CPG configuration does not implement nested S0 retention. Forced airfoil-surface points are present in every method. Selected indices and compensation exactly match the verified native saved records. Raw Uniform/PI model refinement scores are not defined; PI uses its geometry prior.",
         [m + "_points.csv" for m in METHODS] + ["density_bins.csv"])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), layout="constrained")
    artist = field(axes[0], data["pi"]["geometry_prior"], Normalize(1, 21))
    axes[0].set_title("PI · policy score (geometry prior)")
    fig.colorbar(artist, ax=axes[0], shrink=.7, pad=.02)
    pressures = np.concatenate([data[m]["coarse_pressure"][data[m]["coarse"]] for m in ["cpg", "ca"]])
    norm = Normalize(float(pressures.min()), float(pressures.max()))
    for ax, method in zip(axes[1:], ["cpg", "ca"]):
        indices = data[method]["coarse_indices"]
        view = visible[indices]
        artist = ax.scatter(coords[indices[view], 0], coords[indices[view], 1],
                            c=data[method]["coarse_pressure"][indices[view]], s=8, linewidths=0,
                            cmap="viridis", norm=norm)
        finish_axes(ax)
        ax.set_title("CPG · coarse model pressure" if method == "cpg" else "CA · learned head pressure")
        fig.colorbar(artist, ax=ax, shrink=.7, pad=.02)
    save(fig, "airfoil_acquisition_inputs",
         "Actual acquisition inputs, with no CFD truth access: PI's physical surface-distance prior1+20exp(-d/0.3), CPG pressure from the frozen full operator evaluated on1127 coarse points, and pressure from the matching learned CA head on its1127-point first-block features. The two pressure panels share a color scale. CA's128-dimensional first-block features and8-dimensional head bottleneck remain in the external NPZ archive; this figure is reproduced from public scalar CSVs alone.",
         ["pi_points.csv", "cpg_points.csv", "ca_points.csv"])

    errors = {m: {"absolute_error": np.genfromtxt(ROOT / f"{m}_error_points.csv", delimiter=",", names=True, dtype=float)["absolute_error"]} for m in METHODS}
    error_norm = Normalize(0, max(float(d["absolute_error"].max()) for d in errors.values()))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), layout="constrained")
    for ax, method in zip(axes, METHODS):
        artist = field(ax, errors[method]["absolute_error"].reshape(-1), error_norm, cmap="magma")
        ax.set_title(f"{LABELS[method]} · absolute pressure error")
    fig.colorbar(artist, ax=axes, shrink=.65, pad=.02)
    save(fig, "airfoil_error_only",
         "Post-hoc absolute pressure errors, using ground truth only after all score and sampling artifacts had been written. Predictions reuse the saved verified native outputs because selected indices and compensation match exactly. Error values and ground truth are kept in separate error-only artifacts and do not enter acquisition, normalization or sampling.",
         [m + "_error_points.csv" for m in METHODS])
    surface_ids = np.flatnonzero(data["uniform"]["mandatory"])
    surface_coords = coords[surface_ids]
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(surface_coords, axis=0), axis=1))]
    line_colors = {"uniform": "#666666", "pi": "#0072B2", "cpg": "#D55E00", "ca": "#009E73"}
    line_styles = {"uniform": ":", "pi": "--", "cpg": "-.", "ca": "-"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), layout="constrained")
    for method in ["cpg", "ca"]:
        for ax, key in zip(axes[0], ["raw_score", "normalized_score"]):
            ax.plot(arclength, data[method][key][surface_ids], label=LABELS[method],
                    color=line_colors[method], ls=line_styles[method], lw=1.3)
    for method in METHODS:
        reported = data[method]["reported_sampling_density"][surface_ids]
        axes[1, 0].plot(arclength, reported, label=LABELS[method], color=line_colors[method],
                        ls=line_styles[method], lw=1.3)
        proposal = data[method]["proposal_probability"][surface_ids]
        assert np.all(proposal == 0), "The frozen mandatory surface must have zero extra-draw probability"
        axes[1, 1].plot(arclength, proposal, label=LABELS[method], color=line_colors[method],
                        ls=line_styles[method], lw=1.3)
    axes[0, 0].set(title="Raw refinement score on the surface", ylabel="Pressure-gradient score")
    axes[0, 1].set(title="Normalized refinement score on the surface", ylabel="Score / q95")
    axes[1, 0].set(title="Reported selected-point density", ylabel="SamplingResult density", yscale="log")
    axes[1, 1].set(title="Extra-point proposal on the forced surface", ylabel="Proposal probability", ylim=(-.005, .005))
    axes[1, 1].text(.5, .7, "All 121 mandatory points: p = 0", transform=axes[1, 1].transAxes,
                    ha="center", fontsize=9, color="#444444")
    for ax in axes.flat:
        ax.set_xlabel("Surface arclength (native coordinate units)")
        ax.legend(frameon=False, fontsize=8, ncols=2)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    curve_fields = ["point_index", "surface_arclength", "x", "y"]
    curve_keys = ["raw_score", "normalized_score", "geometry_prior", "normalized_prior", "proposal_probability", "reported_sampling_density"]
    curve_fields += [m+"_"+key for m in METHODS for key in curve_keys]
    with (OUT / "surface_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(curve_fields)
        for j, index in enumerate(surface_ids):
            values = [index, arclength[j], *coords[index]] + [data[m][key][index] for m in METHODS for key in curve_keys]
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in values])
    save(fig, "airfoil_surface_curves",
         "Curves follow the actual121 forced airfoil-surface points in native order, parameterized by cumulative physical arclength. CPG/CA raw pressure-gradient scores and q95-normalized scores are shown separately. The lower-left panel is the unmodified SamplingResult selected-point density; PI's curve is its geometry policy score divided by the full-grid sum, while CPG/CA bookkeeping density differs from the gamma-powered categorical proposal. Every surface point is already mandatory, so its extra-point proposal probability is exactly zero for all methods (lower right); zero is not zero final allocation. The scalar source CSV also records PI's raw geometry policy score and sum-normalized prior.",
         [m + "_points.csv" for m in METHODS] + ["surface_curves.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")
    for ax, key, title, unit in [(axes[0], "coarse_feature_l2", "CA · first-block feature L2", "Latent units"),
                                (axes[1], "ca_standardized_output", "CA · raw pressure-head output", "Standardized pressure")]:
        values = data["ca"][key]
        finite = values[np.isfinite(values)]
        norm = LogNorm(float(finite.min()), float(finite.max())) if key == "coarse_feature_l2" else Normalize(float(finite.min()), float(finite.max()))
        artist = field(ax, values, norm)
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=.7, pad=.02, label=unit)
    save(fig, "airfoil_ca_feature_and_head",
         "The actual CA first-block128-dimensional feature tensor is summarized by its per-coarse-point L2 norm (latent units, logarithmic color scale), alongside the actual scalar head output before pressure decoding (standardized pressure). Both fields are shown only on the1127 input coarse points. Feature mean and population standard deviation across128 channels are also provided as scalar CSV columns; these are feature summaries, not training uncertainty. Physical decoded pressure and the resulting gradient acquisition score are separate stages shown in the acquisition-input and score-stage figures. No new model inference or CFD truth is used to compute these summaries.",
         ["ca_points.csv"])
    images = [Image.open(OUT / (q["figure"] + ".png")).convert("RGB") for q in qa]
    thumbs = []
    for im in images:
        im.thumbnail((1400, 900)); thumbs.append(im)
    sheet = Image.new("RGB", (2800, 900*((len(qa)+1)//2)), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 2)*1400, (i // 2)*900))
    sheet.save(OUT / "contact_sheet.png")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "render_qa.json"]
    (OUT / "render_qa.json").write_text(json.dumps({"status": "rendered_decode_passed", "figures": qa,
            "physical_window": {"x": XLIM, "y": YLIM}, "full_point_count": len(coords),
            "visible_point_count": int(visible.sum()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "files": files}, indent=2) + "\n")
    print(json.dumps({"figures": len(qa), "formats": ["pdf", "svg", "300dpi png"], "visible_points": int(visible.sum())}))


if __name__ == "__main__":
    main()
