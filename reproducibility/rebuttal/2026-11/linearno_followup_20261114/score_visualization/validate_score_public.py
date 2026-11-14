"""Validate the published scalar score data without models or CFD truth files."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.int64).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args();root=a.data_dir
    manifest=rows(root/'score_visualization_manifest.csv');index=rows(root/'score_fields_index.csv')
    index={x['score_id']:x for x in index};summary={x['score_id']:x for x in rows(root/'score_summary.csv')}
    reports=[]
    for m in manifest:
        info=index[m['score_id']];path=root/info['path'];assert sha(path)==info['sha256']
        with np.load(path,allow_pickle=False) as z:d={k:np.array(z[k]) for k in z.files}
        n=len(d['point_id']);assert np.array_equal(d['point_id'],np.arange(n))
        assert d['coordinates'].shape[0]==n and np.isfinite(d['coordinates']).all()
        selected=d['point_id'][d['selected'].astype(bool)]
        order=d['selected_order'][d['selected'].astype(bool)].astype(np.int64)
        assert sorted(order.tolist())==list(range(len(order)))
        actual=selected[np.argsort(order)];assert array_sha(actual)==m['selected_index_hash']
        assert int(summary[m['score_id']]['selected_count'])==len(actual)
        density=d['density'];assert np.isfinite(density).all() and (density>=0).all()
        assert np.isclose(density.sum(),1,rtol=1e-10,atol=1e-12)
        ess=1/np.square(density).sum();assert np.isclose(float(summary[m['score_id']]['effective_sample_size']),ess)
        assert m['truth_access']=='false' and m['truth_access_scope']=='score_computation'
        assert m['script_commit'] and len(m['script_commit'])==40
        assert m['status'] in ['VERIFIED_COMPLETE','PARTIAL']
        reports.append(dict(score_id=m['score_id'],points=n,selected=len(actual),selected_index_hash=m['selected_index_hash'],density_sum=float(density.sum())))
    qa=json.loads((root/'score_visualization_qa.json').read_text())
    assert qa['failed_numerical_checks']==0
    assert qa['FZ2_have_revised_allowed'] is False
    print(json.dumps(dict(status='passed_public_scalar_checks',records=reports,FZ2_status=qa['FZ2_status']),indent=2))

if __name__=='__main__':main()
