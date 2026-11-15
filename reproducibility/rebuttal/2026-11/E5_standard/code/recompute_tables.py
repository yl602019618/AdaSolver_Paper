"""Recompute published tables from metrics_records.csv.gz without model weights."""
import argparse
from collections import defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    data = parser.parse_args().data
    groups, keys = defaultdict(list), set()
    with gzip.open(data / "metrics_records.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            b, t, m = row["benchmark"], int(row["training_seed"]), row["method"]
            budget = float(row["budget_fraction"])
            seed = None if row["sampling_seed"] == "" else int(row["sampling_seed"])
            key = b, t, m, budget, seed, int(row["sample_id"])
            assert key not in keys
            keys.add(key)
            value = float(row["relative_l2"])
            calculated = float(row["relative_l2_numerator"]) / max(float(row["relative_l2_denominator"]), 1e-12)
            assert abs(value - calculated) <= 1e-14
            groups[b, t, m, budget, seed].append(value)
    assert len(keys) == 66600 and all(len(v) == 200 for v in groups.values())
    means = {key: math.fsum(values) / len(values) for key, values in groups.items()}
    budgets, absolute, area = (.375, .5, .75), {}, {}
    for benchmark in ("elasticity", "pipe", "darcy"):
        for method in ("full", "uniform", "pi", "cpg", "ca"):
            for budget in ((1.,) if method == "full" else budgets):
                runs = [statistics.mean(means[benchmark, t, method, budget, s]
                                         for s in ((None,) if method == "full" else (0, 1, 2)))
                        for t in (0, 1, 2)]
                absolute[benchmark, method, budget] = runs
            if method == "full":
                continue
            runs = []
            for t in (0, 1, 2):
                ratios = []
                for s in (0, 1, 2):
                    v = [means[benchmark, t, method, b, s] for b in budgets]
                    u = [means[benchmark, t, "uniform", b, s] for b in budgets]
                    numerator = (v[0] + v[1]) / 16 + (v[1] + v[2]) / 8
                    denominator = (u[0] + u[1]) / 16 + (u[1] + u[2]) / 8
                    ratios.append(numerator / denominator)
                runs.append(statistics.mean(ratios))
            area[benchmark, method] = runs
    max_difference, count = 0., 0
    for filename, lookup in (("training_seed_summary.csv", absolute), ("training_seed_rauec.csv", area)):
        with (data / filename).open() as stream:
            for row in csv.DictReader(stream):
                key = ((row["dataset"], row["method"], float(row["budget_fraction"]))
                       if "budget_fraction" in row else (row["dataset"], row["method"]))
                values = lookup[key]
                expected = [*values, statistics.mean(values), statistics.stdev(values)]
                actual = [float(row[f"training_seed{t}"]) for t in (0, 1, 2)]
                actual += [float(row["mean"]), float(row["training_seed_sd"])]
                differences = [abs(a - b) for a, b in zip(actual, expected)]
                max_difference = max(max_difference, *differences)
                assert max(differences) <= 1e-12
                count += 1
    print(f"Verified {len(keys)} unique records and {count} summary rows; maximum numeric difference {max_difference:.3g}.")


if __name__ == "__main__":
    main()
