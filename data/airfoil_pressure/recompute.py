#!/usr/bin/env python3
"""Recompute the two Airfoil columns and all 24 budget points from case records."""
import csv
import json
import math
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
BUDGETS = [.375, .5, .75]
METHODS = ['uniform', 'pi', 'cpg', 'ca']


def rows(name):
    return list(csv.DictReader((HERE / name).open()))


def check(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-13), (actual, expected)


def integrate(values):
    return sum((b - a) * (x + y) / 2 for a, b, x, y in zip(BUDGETS, BUDGETS[1:], values, values[1:]))


def main():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    full = rows('fullmesh_records.csv')
    ids = {int(r['sample_id']) for r in full}
    assert len(full) == len(ids) == 110 and ids == set(protocol['sample_ids'])
    assert all(float(r['true_cl']) > .02 for r in full)
    curves = list(csv.DictReader((PAPER / 'figures/appendix/normalized_curve_points.csv').open()))
    table = (PAPER / 'tables/standard_benchmarks.tex').read_text()
    budget_summary = rows('budget_summary.csv')
    checked = {}
    for metric, key, prefix in [('field', 'pressure_field_relative_l2', ''), ('cdcl', 'ratio_abs_error', 'cdcl_')]:
        records = rows(prefix + 'test_records.csv')
        summary = json.loads((HERE / (prefix + 'test_summary.json')).read_text())
        assert len(records) == 3960
        denominator = stats.mean(float(r[key]) for r in full)
        check(denominator, protocol['fullmesh_denominators'][metric])
        means = {}
        for method in METHODS:
            means[method] = []
            for seed in [0, 1, 2]:
                curve = []
                for budget in BUDGETS:
                    rr = [r for r in records if r['method'] == method and int(r['seed']) == seed and float(r['budget_fraction']) == budget]
                    assert len(rr) == 110 and {int(r['sample_id']) for r in rr} == ids
                    assert all(math.isfinite(float(r[key])) for r in rr)
                    curve.append(stats.mean(float(r[key]) for r in rr))
                for value, expected in zip(curve, summary[method]['per_seed_mean_errors'][str(seed)]):
                    check(value, expected)
                means[method].append(curve)
        for method in METHODS:
            d = summary[method]
            ratios = [integrate(means[method][seed]) / integrate(means['uniform'][seed]) for seed in [0, 1, 2]]
            check(stats.mean(ratios), d['rauec_mean'])
            check(stats.stdev(ratios), d['rauec_std'])
            for seed, value in enumerate(ratios):
                check(value, d['per_seed_rauec'][str(seed)])
            if method != 'uniform':
                line = next(line for line in table.splitlines() if line.strip().startswith(method.upper() + ' &'))
                cell = line.split(' & ')[1 if metric == 'field' else 2]
                assert f"{stats.mean(ratios):.4f}" in cell and f"{stats.stdev(ratios):.4f}" in cell
            for i, budget in enumerate(BUDGETS):
                values = [means[method][s][i] for s in [0, 1, 2]]
                mean, sd = stats.mean(values), stats.stdev(values)
                check(mean, d['mean'][i]); check(sd, d['std'][i])
                r = next(r for r in curves if (r['benchmark'], r['metric'], r['method'], float(r['budget_percent'])) == ('airfoil2d', metric, method, budget * 100))
                assert int(r['sample_count']) == 110
                check(mean, r['raw_mean_error']); check(sd, r['raw_std_error'])
                check(denominator, r['fullmesh_mean_error'])
                check(mean / denominator, r['normalized_mean']); check(sd / denominator, r['normalized_std'])
                b = next(r for r in budget_summary if (r['metric'], r['method'], float(r['budget_fraction'])) == (metric, method, budget))
                for value, name in [(mean, 'mean'), (sd, 'std'), (mean/denominator, 'normalized_mean')]:
                    check(value, b[name])
            checked[f'{metric}/{method}'] = [stats.mean(ratios), stats.stdev(ratios)]
    references = list(csv.DictReader((PAPER / 'figures/appendix/fullmesh_training_curve_points.csv').open()))
    airfoil = [r for r in references if r['benchmark'] == 'airfoil2d']
    assert len(references) == (48 if airfoil else 40)
    if airfoil:
        manifest = json.loads((PAPER / 'data/airfoil_fullmesh_reference/training_manifest.json').read_text())
        assert len(airfoil) == 8 and manifest['target_channel'] == 3 and manifest['nmin_fraction'] == 1.
    print(json.dumps(dict(status='pass', cases=110, compact_records=7920, verified_curve_points=24,
        matched_reference_points=len(references), rauec=checked), indent=2))


if __name__ == '__main__':
    main()
