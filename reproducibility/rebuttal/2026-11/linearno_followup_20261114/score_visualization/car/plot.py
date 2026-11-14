#!/usr/bin/env python3
"""Plot archived native scores, explicitly defined densities, and actual points."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["uniform", "pi", "cpg", "ca"]
    arrays = {}
    for method in methods:
        with (args.data_dir/f"{method}_perpoint.csv").open() as f:
            rows=list(csv.DictReader(f))
        a={key:np.array([float(r[key]) if r[key] else np.nan for r in rows]) for key in rows[0]}
        a["xyz"]=np.column_stack([a["x"],a["y"],a["z"]]).astype(np.float32)
        for key in ["surface","selected","coarse"]:
            a[key]=a[key].astype(bool)
        a["actual_selected_indices"]=np.flatnonzero(a["selected"])
        arrays[method]=a
    xyz, surface = arrays["uniform"]["xyz"], arrays["uniform"]["surface"]
    # This one geometry-only display mask is fixed for every method and panel.
    center = .5*(xyz[:, 0].min()+xyz[:, 0].max())
    halfwidth = .03*np.ptp(xyz[:, 0])
    display = surface | (np.abs(xyz[:, 0]-center) <= halfwidth)
    point_ids = np.flatnonzero(display)
    coords = xyz[:, [2, 0, 1]]  # z-length, x-width, y-height; values unchanged.
    bounds = np.array([coords.min(0), coords.max(0)])
    aspect = np.ptp(coords, axis=0)
    score_norm = LogNorm(.1, max(arrays[m]["normalized_score_native"].max() for m in ["cpg", "ca"]))
    target_norm = LogNorm(min(arrays[m]["target_density"].min() for m in methods),
                          max(arrays[m]["target_density"].max() for m in methods))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":9,
                         "pdf.fonttype":42, "svg.fonttype":"none", "svg.hashsalt":"adasolver-car-score-v1", "savefig.dpi":300})
    fig = plt.figure(figsize=(13.5, 8.4), layout="constrained")
    drawn = []

    def axis(row, col):
        ax = fig.add_subplot(3, 4, row*4+col+1, projection="3d", proj_type="ortho")
        ax.view_init(elev=18, azim=-62)
        ax.set(xlim=bounds[:,0], ylim=bounds[:,1], zlim=bounds[:,2])
        ax.set_box_aspect(aspect)
        ax.set_axis_off()
        return ax

    def scatter(ax, ids, values=None, norm=None, cmap="viridis", color=None, size=2.5):
        if values is None:
            return ax.scatter(*coords[ids].T, color=color, s=size, depthshade=False, linewidths=0, rasterized=False)
        return ax.scatter(*coords[ids].T, c=values[ids], norm=norm, cmap=cmap, s=size, depthshade=False, linewidths=0, rasterized=False)

    for col, method in enumerate(methods):
        a = arrays[method]
        ax = axis(0, col)
        if method in ["cpg", "ca"]:
            values, norm = a["normalized_score_native"], score_norm
            label = "Saved q95-normalized refinement score"
            title = method.upper()+" · saved native score"
        else:
            values, norm = a["raw_prior"], Normalize(0, 1)
            label = "Equal prior" if method == "uniform" else "Hard-shell membership"
            title = "Uniform · equal prior" if method == "uniform" else "PI · actual hard-shell prior"
        scatter(ax, point_ids, values, norm)
        ax.set_title(title)
        fig.colorbar(ScalarMappable(norm=norm,cmap="viridis"), ax=ax, location="bottom", shrink=.78, pad=.015, label=label)
        ax = axis(1, col)
        ids=point_ids
        values=a["target_density"]
        norm=target_norm
        label="Designed target mass; sum = 1"
        title=("Uniform" if method=="uniform" else method.upper())+" · designed target density"
        scatter(ax, ids, values, norm)
        ax.set_title(title)
        fig.colorbar(ScalarMappable(norm=norm,cmap="viridis"), ax=ax, location="bottom", shrink=.78, pad=.015, label=label, format="%.1e")
        ax = axis(2, col)
        selected_ids = point_ids[a["selected"][point_ids]]
        visible_ids = point_ids[(a["selected"] | surface)[point_ids]]
        point_colors = np.where(a["coarse"][visible_ids], "#D55E00", np.where(a["selected"][visible_ids], "#222222", "#d6d6d6"))
        ax.scatter(*coords[visible_ids].T, c=point_colors, s=2.2, depthshade=False, linewidths=0, rasterized=False)
        ax.set_title(("Uniform" if method=="uniform" else method.upper())+" · actual selected points")
        coarse_total=int(a['coarse'].sum())
        coarse_retained=int((a['selected']&a['coarse']).sum())
        point_note=f"{int(a['selected'].sum()):,} / {len(xyz):,} native points"
        if coarse_total:
            point_note=f"{int(a['selected'].sum()):,} selected; {coarse_retained:,}/{coarse_total:,} coarse"
        ax.text2D(.5,.025,point_note,transform=ax.transAxes,ha="center",fontsize=8)
        drawn.append({"method":method,"displayed_domain_points":len(point_ids),"displayed_selected_points":len(selected_ids),
                      "total_selected_points":int(a["selected"].sum()),"native_surface_retained":int((a["selected"]&surface).sum()),
                      "coarse_total":coarse_total,"coarse_retained":coarse_retained})
    for ext in ["pdf", "svg", "png"]:
        meta={"CreationDate":None,"ModDate":None} if ext=="pdf" else {"Date":None} if ext=="svg" else {}
        fig.savefig(args.output_dir/f"car_score_density_selected.{ext}",bbox_inches="tight",pad_inches=.06,metadata=meta)
    plt.close(fig)
    with Image.open(args.output_dir/"car_score_density_selected.png") as im:
        im.load()
        qa={"png_decode":"passed","png_size":im.size,"dpi":im.info.get("dpi"),"PDF_render":"pending"}
    np.save(args.output_dir/"display_point_indices.npy",point_ids)
    receipt={"case_rule":"first locked fold0 test case; no favorable-case selection", "budget":.5,"sampling_seed":0,
        "display_rule":"all native surface points plus one fixed central slab of native volume points; abs(native_x-mid_x)<=0.03*full_x_extent",
        "display_mask_sha256":hashlib.sha256(display.tobytes()).hexdigest(),"coordinate_permutation":[2,0,1],
        "coordinates_unchanged":True,"camera":{"elevation":18,"azimuth":-62,"projection":"orthographic"},
        "score_source":"CPG/CA use exact archived normalized_score_native; CPU score/head traces are not plotted",
        "density_source":"Full-reference designed target from native CPG/CA priority, actual PI hard-shell mixture configuration, and Uniform equal weights. Actual quota/FPS allocation and selected compensation density remain separate CSV columns.",
        "limits":{"q95_score_log_min":score_norm.vmin,"q95_score_log_max":score_norm.vmax},"drawn":drawn,
        "plot_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_csvs":{m+"_perpoint.csv":hashlib.sha256((args.data_dir/(m+"_perpoint.csv")).read_bytes()).hexdigest() for m in methods},"QA":qa}
    (args.output_dir/"visual_manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (args.output_dir/"caption.md").write_text("ShapeNet-Car acquisition and retained points for the first locked test case, Transolver3 training seed0, sampling seed0, and50% budget. The top row shows the native equal prior, PI hard-shell prior, and the exact archived CPG/CA refinement scores. Pressure values are never relabeled as scores. The middle row shows the designed target probability mass on the complete native reference cloud: equal weights for Uniform, the configured50/50 uniform/shell mixture for PI, and the normalized native geometric/score priority for CPG/CA. It matches the locked paper's design formulas. These are target measures, not independent inclusion probabilities; integer quotas, mandatory anchors, without-replacement selection, FPS and PI fallback govern realized selection. Their allocation densities and native compensation remain separate source columns. The bottom row uses the actual saved selected indices. Every panel shares one fixed geometry-only display mask and camera; all counts refer to the complete native cloud. The separate CA raw-head CPU replay does not meet strict archived-score parity and is excluded from this visual.\n")
    with (args.output_dir/"caption.md").open("a") as f:
        f.write("Orange points denote the actual retained coarse subset for CPG and CA:3,218 of3,218 coarse points are retained in both methods. Uniform and PI have no model-acquisition coarse set in these runs.\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
