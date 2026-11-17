"""Bind the unchanged E5 case IDs to a newly trained LinearNO checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from common import E5, activate_sources, verify_lock, write_json
activate_sources()
from evaluate_linearno import LinearNORuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = verify_lock()
    runtime = LinearNORuntime(args.manifest)
    if runtime.metadata["normalizer_sha256"] != protocol["normalizer_sha256"]:
        raise ValueError("Normalizer differs from the matched cohort")
    result = json.loads((E5 / "config/sample_ids.json").read_text())
    result["operator_checkpoint_sha256"] = runtime.metadata["checkpoint_sha256"]
    result.pop("split_hash")
    result["split_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = Path(args.output)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError("Existing split binding differs; preserve it")
    else:
        write_json(output, result)
    runtime.splits(output)


if __name__ == "__main__":
    main()
