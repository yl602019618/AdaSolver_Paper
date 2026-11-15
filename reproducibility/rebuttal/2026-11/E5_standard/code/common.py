"""Shared paths and serialization for the standard-benchmark training runs."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
source = os.environ.get("ADASOLVER_SOURCE_ROOT")
snapshot = ROOT / "code/vendor"
sys.path.insert(0, str(snapshot) if (snapshot / "adasolver").exists() else source or str(snapshot))
for name in ("data", "logs", "runtime", "checkpoints", "records", "results", "internal"):
    (ROOT / name).mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def configuration(benchmark):
    return read(ROOT / "config" / f"{benchmark}.json")
