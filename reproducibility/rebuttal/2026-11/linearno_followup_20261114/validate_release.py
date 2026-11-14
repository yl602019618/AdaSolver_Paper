"""Recompute public CSV summaries and verify release bytes. No training/data access."""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-10),(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    root=a.repo.resolve();data=root/'data/rebuttal/2026-11/linearno_followup_20261114'
    counts={p.name:len(rows(p)) for p in data.glob('*.csv')}
    training=rows(data/'training_seed_summary.csv')
    for r in training:
        v=[float(r[f'seed{i}']) for i in range(3)]
        assert int(r['independent_training_count'])==3
        close(r['mean'],statistics.mean(v));close(r['sample_sd'],statistics.stdev(v))
    samples=rows(data/'sampling_variability.csv')
    keys=['dataset','backbone','training_seed','metric','policy_target','cohort','checkpoint_hash','population_id','reconstruction_id']
    lookup={tuple(r[k] for k in keys)+(r['method'],float(r['budget']),r['sampling_seed'],r['beta']):float(r['mean']) for r in samples}
    assert len(lookup)==len(samples),'duplicate sampling summary identity'
    ratios=rows(data/'backbone_rauec.csv')
    for r in ratios:
        assert r['method']!='full'
        grid=[float(x) for x in r['budget_grid'].split(';')];assert grid==[.375,.5,.75]
        values=[]
        for seed in range(3):
            prefix=tuple(r[k] for k in keys)
            def curve(method,beta):
                return [lookup[prefix+(method,b,str(seed),beta)] for b in grid]
            # Uniform compensation has no beta; recover its explicit CSV identity.
            uniform=[x for x in samples if tuple(x[k] for k in keys)==prefix and x['method']=='uniform' and x['sampling_seed']==str(seed)]
            assert len(uniform)==3
            uy={float(x['budget']):float(x['mean']) for x in uniform}
            y=curve(r['method'],r['beta'])
            area=lambda v:sum((grid[i+1]-grid[i])*(v[i+1]+v[i])/2 for i in range(2))
            denominator=area([uy[b] for b in grid]);assert denominator>0
            value=area(y)/denominator;values.append(value)
            close(r[f'matched_uniform_denominator_seed{seed}'],denominator)
            close(r[f'sampling_seed{seed}'],value)
        close(r['mean'],statistics.mean(values));close(r['sampling_sd'],statistics.stdev(values))
        if r['method']=='uniform':close(r['mean'],1)
    parity=rows(data/'reconstruction_parity.csv')
    index=data/'parity_record_index.csv'
    if index.exists():
        entries=rows(index)
        assert len(entries)==9900, 'Airfoil parity index must contain9900 unique measured pairs'
        assert len({tuple(r.values()) for r in entries})==len(entries),'duplicate parity index rows'
    for r in parity:
        assert r['pass'].lower()=='true'
        close(r['atol'],1e-6);close(r['rtol'],1e-5)
        assert float(r['max_absolute_difference'])>=0
    for r in rows(data/'native_e3_latency.csv'):
        assert int(r['formal_repeats_per_condition'])==100 and int(r['warmups_per_condition'])==20
        assert float(r['complete_latency_median_ms'])>0
    for r in rows(data/'checkpoint_registry.csv'):
        for k in ['checkpoint_hash','normalizer_hash','split_hash']:
            assert len(r[k])==64 and all(c in '0123456789abcdef' for c in r[k])
    manifest=json.loads((data/'release_manifest.json').read_text())
    for r in manifest['files']:
        f=root/r['path'];assert f.is_file() and f.stat().st_size==r['bytes']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],str(f)
    completed=data/'completed_training'
    if completed.exists():
        model_rows=rows(completed/'model_index.csv')
        assert len({r['checkpoint_sha256'] for r in model_rows})==len(model_rows)
        published={r['checkpoint_hash']:r for r in rows(data/'checkpoint_registry.csv')}
        extra=json.loads((completed/'training_evidence.json').read_text())
        verified={a['row']['checkpoint_sha256']:a['row'] for a in extra['completed_assets'] if a['row']['asset_type']=='backbone'}
        assert set(r['checkpoint_sha256'] for r in model_rows)==set(published)|set(verified)
        for r in model_rows:
            assert r['training_status']=='VERIFIED_COMPLETE' and r['weights_in_git'].lower()=='false'
            if r['checkpoint_sha256'] in published:
                original=published[r['checkpoint_sha256']]
                assert r['head_sha256']==original['head_hash']
                assert r['normalizer_sha256']==original['normalizer_hash']
            else:
                original=verified[r['checkpoint_sha256']]
                assert original['training_verified'] and original['allowed_public']
                assert int(r['epochs_completed'])==original['epochs_requested']==original['epochs_completed']
                assert int(r['optimizer_updates'])==original['optimizer_updates']
                close(r['validation_score'],original['validation_score'])
                assert r['accuracy_status']=='PENDING' and r['latency_status']=='PENDING'
            if r['dataset']=='darcy':assert r['protocol_group']=='legacy_alternative_to_matched900'
        assert len(model_rows)==manifest['completed_backbones']
        assert sum(bool(r['head_sha256']) for r in model_rows)==manifest['completed_matching_heads']
    from PIL import Image
    figure_root=root/'figures/rebuttal/2026-11/linearno_followup_20261114'
    for f in figure_root.rglob('*.png'):
        with Image.open(f) as im:im.verify()
    print(json.dumps({'status':'passed','csv_rows':counts,'training_summary_rows_recomputed':len(training),'rauec_rows_recomputed':len(ratios),'parity_rows':len(parity),'manifest_files':len(manifest['files'])},indent=2))

if __name__=='__main__':main()
