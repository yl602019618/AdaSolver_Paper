#!/usr/bin/env python3
"""Build the appendix inference timing table from measured summaries."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/inference_timing'
ORDER = ['airfoil2d', 'pipe', 'darcy', 'elasticity', 'plasticity2d',
         'shapenet_car', 'driverml', 'superwing']
NAMES = dict(airfoil2d='Airfoil2D', pipe='Pipe', darcy='Darcy', elasticity='Elasticity',
             plasticity2d='Plasticity', shapenet_car='ShapeNet-Car',
             driverml='DrivAerML', superwing='SuperWing')
METHODS = ['uniform', 'pi', 'cpg', 'ca']
BUDGETS = [.375, .5, .75]


def read(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def time_cell(row):
    digits = 1 if row['benchmark'] == 'driverml' else 3
    return f"${float(row['mean_ms']):.{digits}f}\\!\\pm\\!{float(row['case_std_ms']):.{digits}f}$"


def main():
    summary = read('summary.csv')
    lookup = {(r['benchmark'], float(r['budget']), r['method'], r['mode']): r for r in summary}
    assert len(lookup) == len(summary) == 312
    first = [r'\begin{table}[!ht]', r'\centering', r'\begingroup',
             r'\footnotesize', r'\setlength{\tabcolsep}{3.0pt}', r'\renewcommand{\arraystretch}{1.08}',
             r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Task & Budget & Full mesh & Uniform & PI & CPG & CA \\', r'\midrule']
    for i, benchmark in enumerate(ORDER):
        full = lookup[(benchmark, 1., 'full', 'original')]
        assert int(full['cases']) == 3
        for j, budget in enumerate(BUDGETS):
            name = NAMES[benchmark] if j == 0 else ''
            if benchmark == 'elasticity' and j == 0:
                name += r'$\dagger$'
            cells = [name, f'{budget * 100:g}\\%', time_cell(full) if j == 0 else '']
            for method in METHODS:
                row = lookup[(benchmark, budget, method, 'sampled')]
                assert int(row['cases']) == 9
                cells.append(time_cell(row))
            first.append(' & '.join(cells) + r' \\')
        if i < len(ORDER) - 1:
            first.append(r'\addlinespace[2pt]')
    first += [r'\bottomrule', r'\end{tabular}', r'\endgroup',
              r'\caption{\textbf{Inference latency at different point budgets (ms).}',
              r'Values are means $\pm$ standard deviations across test cases and repeated samplings.',
              r'The full-mesh column uses all reference points and three cases, and is shown once per task.',
              r'$\dagger$ The Elasticity timing configuration uses a $25\%$ coarse set and a two-block head.',
              r'Timings cover prediction at the selected points and online refinement, excluding full-field reconstruction.}',
              r'\label{tab:inference_time_sampled}', r'\end{table}', '']
    (ROOT / 'tables/inference_time_sampled.tex').write_text('\n'.join(first))
    print('Generated 24 budget rows from measured means and standard deviations.')


if __name__ == '__main__':
    main()
