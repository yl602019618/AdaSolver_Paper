"""Create reviewable result tables from the verified independent-training runs."""
import csv
from pathlib import Path

from common import ROOT, configuration, read

NAMES = {"elasticity": "Elasticity", "pipe": "Pipe", "darcy": "Darcy"}
METHODS = ("uniform", "pi", "cpg", "ca")


def rows(name):
    with (ROOT / "results" / name).open() as stream:
        return list(csv.DictReader(stream))


def pm(row, places=6):
    return f"{float(row['mean']):.{places}f} ± {float(row['training_seed_sd']):.{places}f}"


def main():
    verification = read(ROOT / "results/verification.json")
    assert verification["status"] == "verified" and verification["raw_records"] == 66600
    rauec = rows("training_seed_rauec.csv")
    absolute = rows("training_seed_summary.csv")
    rlookup = {(r["dataset"], r["method"]): r for r in rauec}
    alookup = {(r["dataset"], r["method"], float(r["budget_fraction"])): r for r in absolute}
    lines = ["# Three independent training seeds on standard benchmarks", "",
        "Elasticity, Pipe and Darcy each use three independently initialized Transolver3 backbones and a separately trained CA head for each backbone. Every model is evaluated on the same 200 test cases, with three inference sampling seeds at each of the 37.5%, 50% and 75% budgets. Full-mesh inference uses the same checkpoint at 100% of its reference points.", "",
        "The tables report mean ± sample standard deviation across the three training runs. Within each training run, errors first average over test cases and sampling seeds. Sampling variability is recorded separately.", "",
        "## Relative area under the error curve", "",
        "| Dataset | Uniform | PI | CPG | CA |", "| --- | ---: | ---: | ---: | ---: |"]
    for benchmark in NAMES:
        lines.append("| " + NAMES[benchmark] + " | " + " | ".join(pm(rlookup[benchmark, m]) for m in METHODS) + " |")
    lines += ["", "For each sampling seed, rAUEC divides the method's trapezoidal error area by the matched Uniform area over budgets [0.375, 0.5, 0.75]. These three ratios are averaged within each training run before computing the training-run mean and SD. This follows the companion ShapeNet-Car E5 analysis.", "",
              "## Individual training runs", "",
              "| Dataset | Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training SD |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for benchmark in NAMES:
        for method in METHODS:
            row = rlookup[benchmark, method]
            values = [f"{float(row[f'training_seed{t}']):.6f}" for t in (0, 1, 2)]
            lines.append("| " + " | ".join([NAMES[benchmark], method.upper(), *values, pm(row)]) + " |")
    lines += ["", "## Full-field relative L2 error", "",
              "| Dataset | Method | 37.5% | 50% | 75% | Full mesh (100%) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for benchmark in NAMES:
        full = alookup[benchmark, "full", 1.]
        lines.append(f"| {NAMES[benchmark]} | Full | — | — | — | {pm(full)} |")
        for method in METHODS:
            values = [pm(alookup[benchmark, method, b]) for b in (.375, .5, .75)]
            lines.append("| " + " | ".join([NAMES[benchmark], method.upper(), *values, "—"]) + " |")
    lines += ["", "## Training and evaluation protocol", "",
              "| Dataset | Full points | Backbone training points | Backbone epochs | CA head epochs | Train / validation / test cases |",
              "| --- | ---: | --- | ---: | ---: | --- |"]
    for benchmark in NAMES:
        cfg = configuration(benchmark)
        minimum, maximum = cfg["training"]["nmin_count"], cfg["nfull"]
        interval = str(minimum) if minimum == maximum else f"{minimum}–{maximum}"
        lines.append(f"| {NAMES[benchmark]} | {maximum} | {interval} | {cfg['training']['epochs']} | {cfg['head']['epochs']} | 1000 / 200 / 200 |")
    lines += ["",
        "All backbones use eight Physics-Attention blocks, hidden width 128, eight attention heads and 64 slices. They train in FP32 with AdamW, a peak learning rate of 1e-3, weight decay 1e-5 and gradient clipping at 0.1. Elasticity uses cosine annealing; Pipe and Darcy use OneCycleLR. Checkpoints minimize physical-space validation relative L2. The test cases are excluded from checkpoint selection.", "",
        "Elasticity uses training IDs 0–999, validation IDs 1000–1199 and test IDs 1600–1799. Pipe uses training IDs 0–999, validation IDs 1200–1399 and test IDs 1000–1199. Darcy trains on smooth1 rows 0–999, validates on smooth2 rows 200–399 and tests on smooth2 rows 0–199. Normalizers are fitted on training samples.", "",
        "CA is trained separately for each backbone. Elasticity uses a 128→4→1 stress head on frozen first-block features from 48-, 72- and 97-point inputs; all three Elasticity heads use FP32 CPU feature extraction and training. Pipe and Darcy retain their complete two-block Transolver heads trained on GPU. Pipe head training resamples 1664–16641 points; Darcy uses 722-point inputs. Coarse inference uses 97, 1664 and 722 points for Elasticity, Pipe and Darcy, respectively.", "",
        "Sampling parameters are fixed to their prior production settings. Pipe parameters were historically calibrated on the first 50 cases of its current test cohort, and Darcy parameters also have a test-informed development history. The new training-seed study retains these policies without retuning them. Its backbone and head checkpoint selection uses the separate validation cohorts above.", "",
        "## Results by dataset", ""]
    for benchmark in NAMES:
        best = min(("pi", "cpg", "ca"), key=lambda method: float(rlookup[benchmark, method]["mean"]))
        winners = [min(("pi", "cpg", "ca"), key=lambda method: float(rlookup[benchmark, method][f"training_seed{t}"])) for t in (0, 1, 2)]
        if benchmark == "darcy":
            delta = abs(float(rlookup[benchmark, "cpg"]["mean"]) - float(rlookup[benchmark, "ca"]["mean"]))
            lines.append(f"- Darcy: CPG and CA have nearly identical three-run mean rAUEC (difference {delta:.6f}). Their ordering varies across training runs: the lowest method in seeds 0, 1 and 2 is {', '.join(x.upper() for x in winners)}.")
        else:
            lines.append(f"- {NAMES[benchmark]}: {best.upper()} has the lowest mean rAUEC ({pm(rlookup[benchmark, best])}) and the lowest rAUEC in all three training runs.")
    lines += ["", "## Files", "",
        "- `training_seed_rauec.csv`: every training-run rAUEC and its three-run mean and SD.",
        "- `training_seed_summary.csv`: absolute errors by method and final point budget.",
        "- `sampling_variability.csv`: sampling-seed variation within each trained model.",
        "- `paired_training_differences.csv`: matched method differences for each training run.",
        "- `checkpoint_registry.csv`: the nine backbone/head pairs, selection epochs, training costs and identities.",
        "- `metrics_records.csv.gz`: all 66,600 per-case observations.",
        "- `verification.json` and `matrix_checks.csv`: matrix, checkpoint, point-set and numerical aggregation checks.", "",
        "Checkpoints and full selected-point arrays remain in the experiment workspace. Registry paths are relative to that workspace; the Git supplement contains the numeric evidence and reproduction drivers.", ""]
    (ROOT / "results/README.md").write_text("\n".join(lines))
    table = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}",
             r"\toprule", r"Dataset & Uniform & PI & CPG & CA \\", r"\midrule"]
    for benchmark in NAMES:
        cells = [NAMES[benchmark]]
        for method in METHODS:
            row = rlookup[benchmark, method]
            cells.append(f"${float(row['mean']):.4f}\\pm{float(row['training_seed_sd']):.4f}$")
        table.append(" & ".join(cells) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Standard benchmarks across three independent training runs. Entries are mean $\pm$ sample SD of rAUEC over three separately trained backbones and their matching CA heads. Each run uses 200 test cases and three inference sampling seeds.}",
              r"\label{tab:standard_training_seeds}", r"\end{table}", ""]
    (ROOT / "results/standard_training_seeds.tex").write_text("\n".join(table))
    print("Rendered standard-benchmark overview, per-seed tables and LaTeX table.")


if __name__ == "__main__":
    main()
