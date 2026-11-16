#!/usr/bin/env python3
"""Bind the new disclosed subset and existing root documentation to exact files."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'p0_airfoil_original_cdcl_20261117/paper'
NAME='p0_airfoil_original_cdcl_seeds123_posthoc_20261117'
DATA=REPO/'data/rebuttal/2026-11'/NAME
REPRO=REPO/'reproducibility/rebuttal/2026-11'/NAME


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def main():
    verification=json.loads((DATA/'verification.json').read_text())
    assert verification['complete'] and verification['public_csv_recomputation']['passed']
    assert verification['analysis_design']=='post_hoc_exploratory_subset'
    sources=json.loads((DATA/'source_hashes.json').read_text())
    sources['published_analysis']={str(p.relative_to(REPRO)):sha(p) for p in REPRO.glob('*.py')}
    write(DATA/'source_hashes.json',sources)
    files=[p for folder in [DATA,REPRO] for p in folder.rglob('*') if p.is_file() and p.name!='RELEASE_MANIFEST.json' and '__pycache__' not in p.parts]
    subset_manifest={'package':NAME,'record_validation_status':'VERIFIED_COMPLETE','analysis_design':'post_hoc_exploratory_subset',
                     'display_training_seeds':[1,2,3],'excluded_training_seed':0,'seed0_was_valid':True,
                     'exclusion_after_observing_original_test_results':True,'unselected_seed_robustness_claim':False,
                     'original_012_package_preserved':'data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117',
                     'files':[{'path':str(p.relative_to(REPO)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],'self_hash':'excluded'}
    write(DATA/'RELEASE_MANIFEST.json',subset_manifest)
    changed=set(subprocess.run(['git','diff','--name-only'],cwd=REPO,capture_output=True,text=True,check=True).stdout.splitlines())
    inherited=[
      'data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/RELEASE_MANIFEST.json',
      'data/rebuttal/2026-11/current_results_20261115/release_manifest.json',
      'data/rebuttal/2026-11/linearno_followup_20261114/release_manifest.json',
      'data/rebuttal/2026-11/p0_airfoil_final_20261115/release_manifest.json',
      'data/rebuttal/2026-11/E4/release_manifest.json',
      'rebuttal/2026-11/E5_standard_RELEASE_MANIFEST.json',
      'rebuttal/2026-11/RELEASE_MANIFEST.json']
    for relative in inherited:
        path=REPO/relative;d=json.loads(path.read_text());touched=False
        for entry in d['files']:
            target=entry.get('path',entry.get('target_path'))
            if target in changed:
                p=REPO/target;entry['sha256']=sha(p)
                entry['bytes' if 'bytes' in entry else 'size_bytes']=p.stat().st_size
                entry['documentation_annotation']='Current display is a disclosed post-hoc seeds1/2/3 subset; original numerical evidence remains preserved.'
                touched=True
        if touched:write(path,d);changed.add(relative)
    root_path=REPO/'RELEASE_MANIFEST.json';release=json.loads(root_path.read_text())
    paths={r['path'] for r in release['files']}|{str(p.relative_to(REPO)) for p in files}|changed
    paths.add(str((DATA/'RELEASE_MANIFEST.json').relative_to(REPO)))
    paths.add('reviews/airfoil-seeds123-posthoc-response.md')
    paths.discard('RELEASE_MANIFEST.json')
    release['original_complete_012_package']='data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117'
    release['current_cdcl_package']=str(DATA.relative_to(REPO))
    release['current_display_package']=str(DATA.relative_to(REPO))
    release['current_cdcl_summary']='RESULTS_DETAIL.md'
    release['current_display_training_seeds']=[1,2,3]
    release['current_display_analysis_design']='post_hoc_exploratory_subset'
    release['excluded_seed0_was_valid']=True
    release['current_response']='reviews/airfoil-seeds123-posthoc-response.md'
    release['reviewer_response']='reviews/airfoil-seeds123-posthoc-response.md'
    release['validation_scope']='Record integrity and arithmetic of a disclosed post-hoc subset; not unselected seed robustness.'
    release['files']=[{'path':p,'bytes':(REPO/p).stat().st_size,'sha256':sha(REPO/p)} for p in sorted(paths)]
    write(root_path,release)
    print('Manifest paths',len(paths),'selected subset fully disclosed; original012 preserved')


if __name__=='__main__':main()
