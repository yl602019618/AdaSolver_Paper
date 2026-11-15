"""Check the consolidated publication manifest and distinct model identities."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,xml.etree.ElementTree as ET

def table(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo.resolve()
    data=repo/'data/rebuttal/2026-11/current_results_20261115'
    m=json.loads((data/'release_manifest.json').read_text())
    for r in m['files']:
        f=repo/r['path'];assert f.is_file(),r['path']
        assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['path']
        assert f.suffix not in ['.pt','.pth','.ckpt'], 'Weights must remain external'
        if f.suffix=='.svg':ET.parse(f)
        if f.name.endswith('.csv.gz'):
            with gzip.open(f,'rt') as stream:
                reader=csv.reader(stream);assert next(reader)
                assert sum(1 for _ in reader)>0
    models=table(data/'model_inventory.csv')
    assert len(models)==19
    assert len({r['checkpoint_sha256'] for r in models})==19
    heads={r['head_sha256'] for r in models if r['head_sha256']}
    assert len(heads)==16
    for r in models:
        assert len(r['checkpoint_sha256'])==len(r['normalizer_sha256'])==64
        assert int(r['epochs_completed'])>0 and int(r['optimizer_updates'])>0
        if r['full_only'].lower()=='true':assert not r['head_sha256']
    assert any(r['dataset']=='darcy' and r['train_cases']=='1000' for r in models)
    assert any(r['dataset']=='darcy' and r['train_cases']=='900' for r in models)
    # The original standard release is corrected for the destination bytes.
    source=json.loads((repo/'rebuttal/2026-11/E5_standard_RELEASE_MANIFEST.json').read_text())
    for r in source['files']:
        f=repo/r['path'];assert f.stat().st_size==r['size_bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
    from PIL import Image
    figures=repo/'figures/rebuttal/2026-11/current_results_20261115'
    for f in figures.rglob('*.png'):
        with Image.open(f) as im:im.verify()
    print(json.dumps(dict(status='passed',manifest_files=len(m['files']),backbones=len(models),heads=len(heads),scope_rows=len(table(data/'test_coverage.csv'))),indent=2))

if __name__=='__main__':main()
