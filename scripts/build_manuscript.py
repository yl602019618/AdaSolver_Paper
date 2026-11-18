#!/usr/bin/env python3
"""Build clean and marked manuscripts against the saved Git revision baseline."""

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "iclr2027_conference"
MARKED_NAME = NAME + "_marked"
BASELINE_FILE = ROOT / "docs/mark-baseline.txt"
GRAPHIC_COMMAND = re.compile(
    r"(?P<start>\\includegraphics\*?\s*(?:\[[^\]]*\]\s*)?\{)"
    r"(?P<path>[^{}]+)(?P<end>\})"
)
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")


def manuscript_sources(folder):
    """Follow the manuscript's literal input/include paths from its entrypoint."""
    sources = []
    seen = set()

    def visit(path):
        if path in seen or not path.is_file():
            return
        seen.add(path)
        text = path.read_text()
        sources.append((path, text))
        uncommented = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for name in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", uncommented):
            child = folder / name
            if not child.suffix:
                child = child.with_suffix(".tex")
            visit(child)

    visit(folder / (NAME + ".tex"))
    return sources


def prepare_graphic_review(baseline_root, revision):
    """Snapshot replaced old assets and expose same-path replacements to latexdiff."""
    old_paths = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", "--name-only", revision], cwd=ROOT
    ).decode().split("\0"))
    changed_paths = set(subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", revision, "--"], cwd=ROOT
    ).decode().split("\0"))
    replacements = {}
    reviews = {}
    sources = manuscript_sources(baseline_root)
    for _, text in sources:
        uncommented = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for match in GRAPHIC_COMMAND.finditer(uncommented):
            name = match.group("path").strip()
            candidates = [Path(name).as_posix()]
            if not Path(name).suffix:
                candidates += [name + suffix for suffix in GRAPHIC_EXTENSIONS]
            old_path = next((path for path in candidates if path in old_paths), None)
            if old_path is None or old_path not in changed_paths:
                continue
            snapshot = Path("review_assets/baseline") / revision / old_path
            replacements[name] = snapshot.as_posix()
            if old_path in reviews:
                continue
            destination = ROOT / snapshot
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = subprocess.check_output(
                    ["git", "show", revision + ":" + old_path], cwd=ROOT
                )
                destination.write_bytes(content)
            reviews[old_path] = dict(
                path=old_path, baseline=snapshot.as_posix(),
                current=old_path if (ROOT / old_path).is_file() else None,
            )
    for path, text in sources:
        updated = GRAPHIC_COMMAND.sub(
            lambda match: match.group("start")
            + replacements.get(match.group("path").strip(), match.group("path"))
            + match.group("end"), text,
        )
        if updated != text:
            path.write_text(updated)
    return list(reviews.values())


def graphic_review_appendix(reviews):
    """Add comparison pages without changing manuscript float or section counters."""
    pages = []
    for index, review in enumerate(reviews, 1):
        current = (
            r"\DIFOincludegraphics[width=\linewidth,height=.36\textheight,keepaspectratio]{"
            + review["current"] + "}"
            if review["current"] else r"\emph{The original graphic file has been removed.}"
        )
        pages.append(
            "\n\\clearpage\n\\begingroup\n\\centering\n"
            + r"{\large\bfseries Figure change review " + str(index) + r"}\par" + "\n"
            + r"{\small\path{" + review["path"] + r"}}\par\medskip" + "\n"
            + r"{\color{red}\bfseries Baseline}\par\smallskip" + "\n"
            + r"\DIFOincludegraphics[width=\linewidth,height=.36\textheight,keepaspectratio]{"
            + review["baseline"] + "}\\par\\medskip\n"
            + r"{\color{blue}\bfseries Current}\par\smallskip" + "\n"
            + current + "\\par\n\\endgroup\n"
        )
    return "".join(pages)


def stack_beta_table_changes(body, revision):
    """Stack changed numeric cells in the dense beta table instead of widening it."""
    table_path = "tables/beta_ablation.tex"
    current = (ROOT / table_path).read_text()
    baseline = subprocess.check_output(
        ["git", "show", revision + ":" + table_path], cwd=ROOT, text=True
    )
    tabular_pattern = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)

    def rows_and_scaffold(source):
        tabular = tabular_pattern.search(source).group(0)
        rows = {}
        scaffold = []
        for line in tabular.splitlines():
            row = re.fullmatch(r"(.*?)\\\\\s*", line)
            cells = [cell.strip() for cell in row.group(1).split("&")] if row else []
            if len(cells) == 8 and cells[1] in {"Field", "Cd", "Cd/Cl"}:
                key = tuple(cells[:2])
                rows[key] = cells
                scaffold.append(key)
            else:
                scaffold.append(line)
        return rows, scaffold

    old_rows, old_scaffold = rows_and_scaffold(baseline)
    new_rows, scaffold = rows_and_scaffold(current)
    # Leave structural table edits to latexdiff; this formatter is for cell updates.
    if scaffold != old_scaffold or new_rows.keys() != old_rows.keys():
        return body
    lines = []
    for item in scaffold:
        if isinstance(item, str):
            lines.append(item)
            continue
        cells = []
        for old, new in zip(old_rows[item], new_rows[item]):
            if old == new:
                cells.append(new)
            else:
                cells.append(
                    r"\shortstack[r]{\DIFdelFL{" + old
                    + r"}\\[1pt]\DIFaddFL{" + new + "}}"
                )
        lines.append(" & ".join(cells) + r" \\")
    replacement = "\n".join(lines)

    def replace_table(match):
        table = match.group(0)
        if r"\label{tab:beta_ablation}" not in table:
            return table
        return tabular_pattern.sub(lambda _: replacement, table, count=1)

    return re.sub(
        r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}",
        replace_table, body, flags=re.DOTALL,
    )


def generate_marked(build, baseline):
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", baseline + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    sources = subprocess.check_output(
        ["git", "archive", revision, "--", "*.tex", "*.bib", "*.sty", "*.bst"],
        cwd=ROOT,
    )
    with tempfile.TemporaryDirectory(prefix="mark-baseline-", dir=build) as folder:
        with tarfile.open(fileobj=io.BytesIO(sources)) as archive:
            archive.extractall(folder, filter="data")
        graphic_reviews = prepare_graphic_review(Path(folder), revision)
        result = subprocess.run(
            [
                "latexdiff", "--flatten", "--type=UNDERLINE",
                "--math-markup=whole", "--graphics-markup=new-only",
                "--append-safecmd=eqref", "--append-mboxsafecmd=eqref",
                "--exclude-safecmd=label",
                "--label=baseline " + revision, "--label=current clean manuscript",
                str(Path(folder) / (NAME + ".tex")), str(ROOT / (NAME + ".tex")),
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    (build / "latexdiff.log").write_text(result.stderr)
    (build / "graphic_changes.json").write_text(json.dumps(graphic_reviews, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(result.stderr)
    # latexdiff 1.3.1 can retain active labels inside deleted display equations.
    # Disable those labels without adding blank paragraphs inside marked math.
    preamble, body = result.stdout.split(r"\begin{document}", 1)
    body = re.sub(
        r"\\DIFdelbegin(?:FL)?\b.*?\\DIFdelend(?:FL)?\b",
        lambda match: re.sub(r"\\label\s*\{[^{}]*\}", r"\\relax{}", match.group(0)),
        body, flags=re.DOTALL,
    )
    # Keep graphics-state resets outside alignments before booktabs top rules.
    body = re.sub(
        r"(\\begin\{tabular\*?\}[^\n]*\n)\s*"
        r"(\\DIF(?:add|del)endFL)\s*(?=\\(?:toprule|hline)\b)",
        r"\2 \1", body,
    )
    body = stack_beta_table_changes(body, revision)
    body = body.replace(
        r"\end{document}", graphic_review_appendix(graphic_reviews) + r"\end{document}"
    )
    # Fit the review frame inside full-width figures instead of adding outer padding.
    graphic_preamble = r"""
\renewcommand{\DIFaddincludegraphics}[2][]{%
  \begingroup
  \sbox{\DIFdelgraphicsbox}{\DIFOincludegraphics[#1]{#2}}%
  \setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.8pt}%
  {\color{blue}\fbox{\resizebox{\dimexpr\wd\DIFdelgraphicsbox-2\fboxrule\relax}{!}{\usebox{\DIFdelgraphicsbox}}}}%
  \endgroup
}
"""
    # Deleted headings can reuse section numbers; keep PDF link targets unique.
    source = preamble + graphic_preamble + "\\hypersetup{hypertexnames=false}\n" + r"\begin{document}" + body
    marked = ROOT / (MARKED_NAME + ".tex")
    if not marked.exists() or marked.read_text() != source:
        marked.write_text(source)
    return revision


def compile_manuscript(name, build, force):
    build.mkdir(exist_ok=True)
    cmd = [
        "latexmk", "-pdf", "-synctex=1", "-interaction=nonstopmode",
        "-halt-on-error", "-file-line-error", "-outdir=" + str(build), name + ".tex",
    ]
    if force:
        cmd.insert(1, "-g")
    result = subprocess.run(
        cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (build / "build.log").write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-10000:], file=sys.stderr)
        raise SystemExit(result.returncode)
    log = (build / (name + ".log")).read_text(errors="replace")
    unresolved = [
        line for line in log.splitlines()
        if ("undefined" in line.lower()
            and any(word in line for word in ["Reference", "Citation", "There were"]))
        or "multiply defined" in line or "duplicate ignored" in line
    ]
    if unresolved:
        raise RuntimeError("\n".join(unresolved))
    pdf = build / (name + ".pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(build / "paper.txt")], check=True
    )
    qa = dict(
        entrypoint=name + ".tex", pages=pages, undefined_references=unresolved,
        overfull_boxes=[line for line in log.splitlines() if line.startswith("Overfull")],
    )
    return pdf, qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--baseline", help="Git revision for this comparison (default: docs/mark-baseline.txt)"
    )
    args = parser.parse_args()
    required = ["git", "latexdiff", "latexmk", "pdflatex", "bibtex", "pdfinfo", "pdftotext"]
    missing = [tool for tool in required if not shutil.which(tool)]
    if missing:
        parser.error("Missing build tools: " + ", ".join(missing))
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    baseline = args.baseline or BASELINE_FILE.read_text().strip()
    revision = generate_marked(build, baseline)
    print("Mark baseline: " + revision, flush=True)
    clean_pdf, clean_qa = compile_manuscript(NAME, build, args.force)
    print("Clean manuscript compiled; compiling marked manuscript.", flush=True)
    marked_pdf, marked_qa = compile_manuscript(MARKED_NAME, build / "marked", args.force)

    # Refresh the published pair only after both versions compile successfully.
    for pdf, qa, name, folder in [
        (clean_pdf, clean_qa, "paper", build),
        (marked_pdf, marked_qa, "paper_marked", build / "marked"),
    ]:
        shutil.copy2(pdf, ROOT / (name + ".pdf"))
        shutil.copy2(pdf, ROOT / pdf.name)
        qa.update(pdf=name + ".pdf", mark_baseline=revision)
        (folder / "build_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
        print(
            f"Built {name}.pdf: {qa['pages']} pages, no undefined references; "
            f"{len(qa['overfull_boxes'])} overfull boxes"
        )


if __name__ == "__main__":
    main()
