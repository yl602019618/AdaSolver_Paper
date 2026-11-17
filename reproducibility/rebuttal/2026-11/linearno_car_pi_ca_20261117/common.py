"""Local paths and immutable E3 source identities for the Car seed extension."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
WORKING_ROOT = ROOT.parents[2]
E3 = WORKING_ROOT / "rebuttal_20261112/experiments/E3_backbones"
E5 = WORKING_ROOT / "rebuttal_20261112/experiments/E5_training_seeds"
OFFICIAL_COMMIT = "3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269"
PAPER_COMMIT = "91c1fa7cc81b8c1cb61a43618296c1c892d818c0"


def activate_sources():
    for path in (E3 / "code/vendor", E3 / "code"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_lock():
    protocol = json.loads((ROOT / "training_tasks.json").read_text())
    for entry in protocol["source_files"]:
        path = WORKING_ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Frozen source mismatch: " + entry["path"])
    for entry in protocol["extension_source_files"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Extension source changed after protocol lock: " + entry["path"])
    return protocol
