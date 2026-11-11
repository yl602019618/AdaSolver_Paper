#!/usr/bin/env python3
"""Remeasure the frozen 97-point Elasticity policies with the paper timing protocol.

Usage: python scripts/benchmark_elasticity_cpg10.py --adasolver-root ../AdaSolver_official
       --gpu 4 --output ../elasticity_cpg10_revision/timing
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adasolver-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "AdaSolver_official")
    parser.add_argument("--gpu", type=int, default=4)
    args, remaining = parser.parse_known_args()
    root = args.adasolver_root.resolve()
    sys.path.insert(0, str(root))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from elasticity_timing.suite import Elasticity10Suite
    from elasticity_timing.runtime import ElasticityAdaptiveRuntime
    from inference_speed import all_runtime, benchmark
    original_runtime = all_runtime.AllMethodRuntime

    def runtime_factory(case, prepared_mass=None):
        if case.method in {"cpg", "ca"}:
            return ElasticityAdaptiveRuntime(case, prepared_mass)
        return original_runtime(case, prepared_mass)

    benchmark.suite_for = lambda name, device: Elasticity10Suite(root, device)
    all_runtime.AllMethodRuntime = runtime_factory
    sys.argv = [sys.argv[0], "--gpu", str(args.gpu), "--benchmarks", "elasticity", *remaining]
    from inference_speed.all_methods import main as measure
    measure()


if __name__ == "__main__":
    main()
