"""Verify the exact published payload and uncompressed scientific records."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'snapshot_manifest.json').read_text())
    expected = set()
    for record in manifest['files']:
        path = ROOT / record['path']; expected.add(record['path'])
        payload = path.read_bytes()
        assert len(payload) == record['bytes'], record['path']
        assert hashlib.sha256(payload).hexdigest() == record['sha256'], record['path']
        if record.get('encoding') == 'gzip':
            decoded = gzip.decompress(payload)
            assert len(decoded) == record['source_bytes'], record['path']
            assert hashlib.sha256(decoded).hexdigest() == record['source_sha256'], record['path']
        assert path.suffix not in {'.pt', '.pth', '.ckpt', '.npy', '.npz'}, record['path']
    print(json.dumps(dict(passed=True, verified_files=len(expected),
        all_requested_experiments_complete=manifest['all_requested_experiments_complete'],
        declared_scope=manifest['completed_accuracy_scope']), indent=2))


if __name__ == '__main__':
    main()
