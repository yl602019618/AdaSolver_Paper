"""Fit a new Car pressure head to its corresponding frozen LinearNO features."""
import argparse
import json
from pathlib import Path
from common import ROOT, OFFICIAL_COMMIT, activate_sources, sha256, verify_lock, write_json
activate_sources()
from adasolver.train.shapenet_car_ca_head import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    verify_lock()
    dataset = json.loads((Path(args.dataset) / "manifest.json").read_text())
    if dataset.get("backbone") != "LinearNO" or dataset.get("feature_backend") != "official_linearno_native_block0":
        raise ValueError("Expected this LinearNO's own feature dataset")
    operator = json.loads(Path(dataset["operator_manifest"]).read_text())
    if operator["seed"] != args.seed or operator["checkpoint_sha256"] != dataset["operator_sha256"]:
        raise ValueError("The head seed/operator must match the training seed")
    metadata = train(**vars(args), width=64, epochs=200, batch_points=65536, smoke=False)
    metadata.update(backbone="LinearNO", feature_backend=dataset["feature_backend"],
                    official_commit=OFFICIAL_COMMIT, training_seed=args.seed, head_seed=args.seed,
                    extension_sources={p.name: sha256(p) for p in (ROOT / "linearno_ca.py", Path(__file__))})
    write_json(Path(args.output_dir) / f"{args.name}_manifest.json", metadata)


if __name__ == "__main__":
    main()
