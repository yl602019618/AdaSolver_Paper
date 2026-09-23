#!/usr/bin/env python3
"""Build appendix timing tables from the archived measurement summaries."""
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
    accuracy = read('accuracy.csv')
    errors = {(r['benchmark'], float(r['budget']), r['method'], r['mode'], r['metric']): r for r in accuracy}
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
              r'\caption{\textbf{Prepared sampled-field request latency (ms).}',
              r'Values are means $\pm$ sample standard deviations across three cases and three sampling seeds.',
              r'The full-mesh column uses all reference points and three cases, and is shown once per task.',
              r'$\dagger$ The Elasticity timing configuration uses a $25\%$ coarse set and a two-block head.',
              r'Complete-field reconstruction is measured separately in Table~\ref{tab:inference_time_reconstruction}.}',
              r'\label{tab:inference_time_sampled}', r'\end{table}', '']
    second = [r'\begin{table}[!ht]', r'\centering', r'\begingroup', r'\footnotesize',
              r'\setlength{\tabcolsep}{3.0pt}', r'\renewcommand{\arraystretch}{1.08}',
              r'\begin{tabular}{llrrrrr}', r'\toprule',
              r'Task & Recon. & Uniform & PI & CPG & CA & $E_{\mathrm{CA}}$ (\%) \\', r'\midrule']
    for i, benchmark in enumerate(ORDER):
        metric = 'field_error' if benchmark == 'shapenet_car' else 'relative_l2'
        for j, mode in enumerate(['original', 'fast']):
            cells = [NAMES[benchmark] if j == 0 else '', 'Original' if j == 0 else 'GPU']
            cells += [time_cell(lookup[(benchmark, .5, method, mode)]) for method in METHODS]
            error = errors[(benchmark, .5, 'ca', mode, metric)]
            assert int(error['cases']) == 9
            cells.append(f"${100 * float(error['mean']):.3f}\\!\\pm\\!{100 * float(error['case_std']):.3f}$")
            second.append(' & '.join(cells) + r' \\')
        if i < len(ORDER) - 1:
            second.append(r'\addlinespace[2pt]')
    second += [r'\bottomrule', r'\end{tabular}', r'\endgroup',
               r'\caption{\textbf{Full-field request latency at a $50\%$ point budget.}',
               r'Times are in ms and include reconstruction and complete-field transfer.',
               r'$E_{\mathrm{CA}}$ gives the CA field error on the same profiling cases, as a percentage;',
               r'ShapeNet-Car uses its weighted velocity--pressure metric.',
               r'The GPU variant changes the reconstruction rule on the five tasks specified in the text.',
               r'Elasticity, ShapeNet-Car, and DrivAerML retain the original rule and share the same recorded executions in both rows.}',
               r'\label{tab:inference_time_reconstruction}', r'\end{table}', '']
    (ROOT / 'tables/inference_time_sampled.tex').write_text('\n'.join(first))
    (ROOT / 'tables/inference_time_reconstruction.tex').write_text('\n'.join(second))
    print('Generated 24 budget rows and 16 reconstruction rows from measured means and standard deviations.')


if __name__ == '__main__':
    main()
