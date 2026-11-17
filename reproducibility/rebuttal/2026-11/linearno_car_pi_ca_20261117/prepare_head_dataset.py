"""Generate the complete E5 pressure supervision with LinearNO block-0 features."""
import argparse
import json
from pathlib import Path
from common import E3, ROOT, OFFICIAL_COMMIT, activate_sources, sha256, verify_lock, write_json
activate_sources()
from evaluate_linearno import LinearNORuntime
from linearno_ca import coarse_features
from adasolver.train import shapenet_car_ca_dataset as dataset_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    verify_lock()
    dataset_module.CarRuntime = LinearNORuntime
    dataset_module.coarse_features = coarse_features
    metadata = dataset_module.build_dataset(**vars(args), masks=2, smoke=False)
    metadata.update(backbone="LinearNO", feature_backend="official_linearno_native_block0",
                    official_commit=OFFICIAL_COMMIT,
                    extension_sources={p.name: sha256(p) for p in (ROOT / "linearno_ca.py", Path(__file__))})
    write_json(Path(args.output) / "manifest.json", metadata)


if __name__ == "__main__":
    main()
