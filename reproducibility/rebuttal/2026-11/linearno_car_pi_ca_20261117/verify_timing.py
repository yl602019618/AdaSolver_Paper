"""CPU request-path checks for the complete Car LinearNO CA timing extension."""
import ast
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from common import ROOT, E3, sha256, verify_lock, write_json
from formal_timing import activate_request_sources, configuration_bank, source_identity
activate_request_sources()
import numpy as np
import torch
import adasolver
from evaluate_linearno import LinearNORuntime
from linearno_adapter import install_linear_compensation
from linearno_ca import LinearNOCA
from adasolver.inference.ca.shapenet_car import CarPressureHead
from car_request import CarRequest
from timing_helper import measure_paired


class Fixture:
    def __len__(self):
        return len(self.pos)


def main():
    verify_lock()
    protocol = source_identity()
    assert Path(adasolver.__file__).resolve().is_relative_to(E3 / "code/vendor")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261114)
    runtime = LinearNORuntime(E3 / "checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json")
    model = runtime.load_operator("cpu")
    context, _ = install_linear_compensation(model)
    fixture = Fixture()
    fixture.pos = torch.rand(120, 3)
    fixture.x_norm = torch.randn(120, 7)
    # Keep the production requirement that the budget retains the full surface.
    fixture.surface_mask = torch.arange(120) < 20
    fixture.sample_id = protocol["timing_cases"][0]
    checks = []
    with tempfile.TemporaryDirectory(prefix="car_ca_request_cpu_") as temp:
        directory = Path(temp)
        checkpoint = directory / "fixture_head.pt"
        torch.save(CarPressureHead(256, 64).state_dict(), checkpoint)
        metadata = {**runtime.binding(), "backbone": "LinearNO", "feature_backend": "official_linearno_native_block0",
                    "target": "surface_pressure", "coarse_fraction": .1, "checkpoint": str(checkpoint),
                    "head_sha256": sha256(checkpoint), "input_dim": 256, "width": 64,
                    "target_std": 1., "target_mean": 0., "status": "development_fixture_untrained"}
        head_manifest = directory / "head.json"
        write_json(head_manifest, metadata)
        ca = LinearNOCA(model, head_manifest, runtime.binding(), "cpu")
        for label, config in configuration_bank(1).items():
            budget, seed = (1., None) if label == "full" else (.375, 0)
            request = CarRequest(model, context, fixture, runtime.output_normalizer, config, budget, seed, ca=ca, device="cpu")
            old = request("old", "A")
            old_indices = request.last_indices.copy()
            coarse = None if request.last_coarse_values is None else request.last_coarse_values.copy()
            new = request("new", "A")
            error = np.abs(new-old)
            passed = bool(np.all(error <= 1e-6 + 1e-5*np.abs(old)))
            assert old.shape == new.shape == (120, 4)
            assert old.dtype == new.dtype == np.float64
            assert np.array_equal(old_indices, request.last_indices)
            if coarse is not None:
                assert np.array_equal(coarse, request.last_coarse_values)
            assert np.all(new[~fixture.surface_mask.numpy(), 3] == 0)
            assert passed and context.weights is None
            checks.append({"variant": label, "points": 120, "old_new_max_absolute_difference": float(error.max()),
                           "elementwise_parity_passed": passed, "selected_indices_unchanged": True,
                           "coarse_prediction_unchanged": True, "output_dtype": str(new.dtype)})
    calls = {"old": 0, "new": 0}
    sync_calls = [0]
    def sync():
        sync_calls[0] += 1
    def request(name):
        calls[name] += 1
        return np.ones((2, 4))
    measured = measure_paired({name: lambda n=name: request(n) for name in calls}, scope="A",
                              warmups=20, repeats=100, synchronize=sync)
    assert calls == {"old": 121, "new": 121}
    assert sync_calls[0] == 484
    assert len([r for r in measured["raw"] if r["kind"] == "formal"]) == 200
    assert len(protocol["timing_cases"]) == 3 and len(set(protocol["timing_cases"])) == 3
    assert set(configuration_bank(0)) == {"pi", "ca"}
    assert set(configuration_bank(1)) == {"full", "uniform", "pi", "cpg_beta0", "cpg_beta1", "ca"}
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    result = {"passed": True, "scope": "CPU full request-path integration on the actual seed0 complete 8-layer/256-hidden backbone; synthetic 120-point geometry and explicitly untrained temporary head. No GPU accuracy, latency or memory observation.",
              "gpu_used": False, "training_started": False, "trained_head_results": False,
              "checkpoint_sha256": runtime.metadata["checkpoint_sha256"], "request_checks": checks,
              "frozen_adasolver_vendor_selected": True,
              "adasolver_source": str(Path(adasolver.__file__).resolve().relative_to(E3)),
              "timing_helper_checks": {"warmups": 20, "formal_repeats": 100, "calls_including_first": calls,
                                       "boundary_sync_invocations": sync_calls[0], "formal_observations": 200},
              "formal_condition_plan_per_scope": {"seed0": 108, "seed1": 273, "seed2": 273},
              "timing_cases": protocol["timing_cases"],
              "resource_lock_contract": "formal_timing.py called directly by queue; it holds only a task lock, then ExclusiveGPU acquires the GPU profile lock. It is not a run_stage.py child.",
              "source_sha256": {p.name: sha256(p) for p in ROOT.glob("*.py")},
              "pending": ["matching trained-head real-case checks", "GPU elementwise old/new parity", "all formal timing observations", "independent numerical and memory-scope acceptance"]}
    write_json(ROOT / "timing_preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
