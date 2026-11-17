"""Refresh only changed document bindings and add the PI/CA package manifest."""
from pathlib import Path
import hashlib
import json
import subprocess
from common import ROOT

PAPER=ROOT/'paper'
NAME='linearno_car_pi_ca_20261117'
DATA=PAPER/'data/rebuttal/2026-11'/NAME
REPRO=PAPER/'reproducibility/rebuttal/2026-11'/NAME


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def main():
    v=json.loads((DATA/'verification.json').read_text());assert v['complete'] and v['public_numeric_recomputation']['passed']
    sources=json.loads((DATA/'source_hashes.json').read_text());sources['published_reproduction_code']={p.name:sha(p) for p in REPRO.glob('*.py')};write(DATA/'source_hashes.json',sources)
    files=[p for folder in [DATA,REPRO] for p in folder.rglob('*') if p.is_file() and p.name!='RELEASE_MANIFEST.json' and '__pycache__' not in p.parts]
    manifest={'package':NAME,'status':'VERIFIED_COMPLETE','backbone':'official ShapeNet-Car LinearNO','training_seeds':[0],
              'new_backbone_training_runs':0,'new_complete_CA_heads':1,'head_epochs':200,'PI_training':'not applicable; geometry only',
              'policy_targets':['field','cd'],'sampling_seeds':[0,1,2],'test_cases':100,'native_accuracy_records':8200,
              'formal_inference_conditions':222,'formal_repetitions':22200,'warmups':4440,'training_sd_claimed':False,
              'files':[{'path':str(p.relative_to(PAPER)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],'self_hash':'excluded'}
    write(DATA/'RELEASE_MANIFEST.json',manifest)
    changed=set(subprocess.run(['git','diff','--name-only'],cwd=PAPER,capture_output=True,text=True,check=True).stdout.splitlines())
    tracked=subprocess.run(['git','ls-files'],cwd=PAPER,capture_output=True,text=True,check=True).stdout.splitlines()
    active=[s for s in tracked if s!='RELEASE_MANIFEST.json' and 'release_manifest' in Path(s).name.lower() and s.endswith('.json')
            and '/provenance/' not in s and 'source_release_manifest_at_' not in s]
    cache={}
    def previous_hash(path):
        if path not in cache:
            r=subprocess.run(['git','show','HEAD:'+path],cwd=PAPER,capture_output=True)
            cache[path]=hashlib.sha256(r.stdout).hexdigest() if r.returncode==0 else None
        return cache[path]
    for _ in range(8):
        updates=0
        for relative in active:
            path=PAPER/relative;d=json.loads(path.read_text());entries=d.get('files');dirty=False
            if not isinstance(entries,list):continue
            for e in entries:
                target=e.get('path',e.get('target_path')) if isinstance(e,dict) else None
                if target not in changed or not (PAPER/target).is_file():continue
                # Resolve bare README paths using their prior content identity,
                # rather than confusing a package README with the repository root.
                if '/' not in target and e.get('sha256') not in [previous_hash(target),sha(PAPER/target)]:continue
                digest=sha(PAPER/target);key='bytes' if 'bytes' in e else 'size_bytes' if 'size_bytes' in e else 'size'
                if e.get('sha256')!=digest or e.get(key)!=(PAPER/target).stat().st_size:
                    e['sha256']=digest;e[key]=(PAPER/target).stat().st_size;dirty=True
            if dirty:write(path,d);changed.add(relative);updates+=1
        if not updates:break
    else:raise RuntimeError('manifest dependency bindings did not settle')
    root=PAPER/'RELEASE_MANIFEST.json';d=json.loads(root.read_text())
    paths={e['path'] for e in d['files']}|changed|{str(p.relative_to(PAPER)) for p in files}
    paths.add(str((DATA/'RELEASE_MANIFEST.json').relative_to(PAPER)));paths.add('LINEARNO_CAR_PI_CA.md');paths.discard('RELEASE_MANIFEST.json')
    d['linearno_car_pi_ca_update']={'data_directory':str(DATA.relative_to(PAPER)),'status':'VERIFIED_COMPLETE','training_seeds':[0],
                                   'new_CA_heads':1,'sampling_seeds':[0,1,2],'formal_conditions':222,
                                   'independent_training_seeds_1_2_included':False}
    d['files']=[{'path':s,'bytes':(PAPER/s).stat().st_size,'sha256':sha(PAPER/s)} for s in sorted(paths)]
    write(root,d);print('Root release paths',len(paths),'new package files',len(files))


if __name__=='__main__':main()
