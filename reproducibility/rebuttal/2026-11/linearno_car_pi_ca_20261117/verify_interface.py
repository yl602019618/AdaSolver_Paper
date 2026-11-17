"""CPU checks on the actual seed-0 full architecture; no benchmark result."""
import json
import tempfile
from pathlib import Path
from common import E3, ROOT, activate_sources, sha256, verify_lock, write_json
activate_sources()
import numpy as np
import torch
from evaluate_linearno import LinearNORuntime
from linearno_adapter import install_linear_compensation
from linearno_ca import LinearNOCA, coarse_features
from adasolver.inference.ca.shapenet_car import CarPressureHead


def main():
    verify_lock()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261114)
    manifest = E3 / "checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json"
    runtime = LinearNORuntime(manifest)
    operator = runtime.load_operator("cpu")
    names = list(operator.state_dict())
    cases = []
    fixture_inputs = []
    for n in (3, 7, 17):
        pos, fx = torch.randn(2, n, 3), torch.randn(2, n, 7)
        observed = []
        handle = operator.model.blocks[0].register_forward_hook(lambda m, a, y: observed.append(y.detach().clone()))
        with torch.no_grad():
            before = operator(pos, fx)
        handle.remove()
        hidden, cache = coarse_features(operator, pos, fx, nfull=10*n, capture=True)
        expected = torch.cat(observed, dim=0)
        with torch.no_grad():
            after = operator(pos, fx)
        assert torch.equal(before, after)
        assert torch.equal(hidden, expected)
        assert hidden.shape == (2, n, 256) and before.shape == (2, n, 4)
        assert cache is None
        cases.append({"points": n, "batch": 2, "feature_channels": 256,
                      "output_channels": 4, "first_block_hook_max_abs_diff": float((hidden-expected).abs().max()),
                      "unchanged_prediction_max_abs_diff": float((before-after).abs().max())})
        fixture_inputs.append((pos, fx, before))
    context, layers = install_linear_compensation(operator)
    unit_diff = 0.
    for pos, fx, before in fixture_inputs:
        context.set(torch.ones(fx.shape[1]))
        with torch.no_grad():
            after = operator(pos, fx)
        unit_diff = max(unit_diff, float((before-after).abs().max()))
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
        context.clear()
    with tempfile.TemporaryDirectory(prefix="car_ca_interface_") as temp:
        directory = Path(temp)
        head = CarPressureHead(256, 64)
        checkpoint = directory / "development_fixture_head.pt"
        torch.save(head.state_dict(), checkpoint)
        metadata = {**runtime.binding(), "backbone": "LinearNO", "feature_backend": "official_linearno_native_block0",
                    "target": "surface_pressure", "coarse_fraction": .1, "checkpoint": str(checkpoint),
                    "head_sha256": sha256(checkpoint), "input_dim": 256, "width": 64,
                    "target_std": 1., "target_mean": 0., "status": "development_fixture_untrained"}
        head_manifest = directory / "manifest.json"
        write_json(head_manifest, metadata)
        ca = LinearNOCA(operator, head_manifest, runtime.binding(), "cpu")
        pos, fx, _ = fixture_inputs[1]
        pressure, cache = ca.acquire(pos[:1], fx[:1], 70)
        assert pressure.shape == (7,)
        weights = torch.linspace(.5, 1.5, 7)
        context.set(weights)
        with torch.no_grad():
            expected = operator(pos[:1], fx[:1])
        context.clear()
        actual = ca.final(pos[:1], fx[:1], np.arange(7), np.arange(7), cache, weights)
        assert torch.equal(actual, expected) and context.weights is None
        final_diff = float((actual-expected).abs().max())
        rejected = False
        metadata["backbone"] = "Transolver3"
        write_json(head_manifest, metadata)
        try:
            LinearNOCA(operator, head_manifest, runtime.binding(), "cpu")
        except ValueError:
            rejected = True
        assert rejected
    assert names == list(operator.state_dict())
    result = {"status": "passed", "passed": True, "scope": "CPU interface/development validation, actual seed0 checkpoint and full 8-layer/256-hidden architecture; synthetic point inputs and an explicitly untrained temporary head fixture; no benchmark accuracy or latency claim",
              "checkpoint_sha256": runtime.metadata["checkpoint_sha256"], "device": "cpu",
              "torch": str(torch.__version__), "cases": cases, "patched_official_layers": layers,
              "unit_compensation_max_abs_diff": unit_diff, "ca_final_matches_official_compensated_prediction_max_abs_diff": final_diff,
              "rejects_head_backbone_mismatch": rejected, "checkpoint_parameter_names_unchanged": True,
              "trained_heads_available": 0, "gpu_work_started": False,
              "source_sha256": {p.name: sha256(p) for p in ROOT.glob("*.py")}}
    write_json(ROOT / "interface_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
