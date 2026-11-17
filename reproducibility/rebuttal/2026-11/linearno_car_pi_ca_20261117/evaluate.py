"""Run existing Car policies/metrics/reconstruction with a LinearNO CA head."""
import argparse
import json
import sys
from pathlib import Path
from common import ROOT, activate_sources, sha256, verify_lock, write_json
activate_sources()
from linearno_ca import LinearNOCA
from adasolver.evaluation import shapenet_car as evaluator
from evaluate_linearno import main as e3_main


if __name__ == "__main__":
    verify_lock()
    evaluator.ShapeNetCarCA = LinearNOCA
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method")
    args, _ = parser.parse_known_args()
    if args.method == "fullmesh" and "--seeds" not in sys.argv:
        # The inherited evaluator requires a seed token, but Full performs
        # no sampling. Use one canonical execution rather than three copies.
        sys.argv.extend(["--seeds", "0"])
    e3_main()
    for manifest in args.output.glob("*/run_manifest.json"):
        metadata = json.loads(manifest.read_text())
        metadata["linearno_ca_extension_sources"] = {
            name: sha256(ROOT / name) for name in ("linearno_ca.py", "evaluate.py")}
        metadata["ca_feature_backend"] = "official_linearno_native_block0" if metadata["config"]["method"] == "ca" else None
        metadata["ca_final_execution"] = "official full forward without Transolver slice cache" if metadata["config"]["method"] == "ca" else None
        if metadata["config"]["method"] == "ca":
            metadata["implementation"].update(cache="none_for_final_operator", ca_acquisition="native_linearno_block0_eager")
        if args.method == "fullmesh":
            metadata["sampling_seed_semantics"] = "not applicable; inherited seed=0 is an execution sentinel and supplies no independent replicate"
        write_json(manifest, metadata)
