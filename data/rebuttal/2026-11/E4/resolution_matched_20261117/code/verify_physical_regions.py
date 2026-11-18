"""Check strict regional records and a direct-distance oracle for figure cases."""
from pathlib import Path
import csv,datetime as dt,hashlib,json,math,os,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
import numpy as np

OUT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_csv(path):
    for attempt in range(5):
        try:
            with Path(path).open() as f:return list(csv.DictReader(f))
        except OSError:
            if attempt==4:raise
            time.sleep(.2)


def main():
    receipt=json.loads((OUT/'results/physical_region_acceptance.json').read_text());assert receipt['passed'] and receipt['complete']
    common=np.load(OUT/'data/r211/reference_coords.npy')
    coeff=np.load(OUT/'data/r211/test_coeff.npy',mmap_mode='r')
    reports=[];oracles=[]
    for n in [85,141,211]:
        r=json.loads((OUT/f'results/physical_region_r{n}_acceptance.json').read_text())
        rows=read_csv(OUT/r['physical_region_csv']);assert len(rows)==22200
        old={ (x['physical_request_id'],x['region']):x for x in read_csv(OUT/f'records/r{n}/region_records.csv')}
        groups={};boundary_expected={85:664,141:1656,211:4120}[n]
        for row in rows:
            assert row['region_protocol_id']==receipt['region_protocol_id']
            key=row['physical_request_id'];groups.setdefault(key,[]).append(row)
            count=int(row['observed_points']);chosen=int(row['selected_points'])
            if row['region']=='boundary':assert count==boundary_expected
            if count:assert abs(float(row['retained_fraction'])-chosen/count)<1e-14
            else:assert row['retained_fraction']==''
            prior=old[key,row['region']]
            assert row['common_relative_l2']==prior['common_relative_l2'] and row['evaluation_points']==prior['evaluation_points']
        assert len(groups)==7400
        for key,values in groups.items():
            assert {v['region'] for v in values}=={'boundary','interface','smooth'}
            assert sum(int(v['observed_points']) for v in values)==n*n
            expected=round(float(values[0]['budget_fraction'])*n*n)
            assert sum(int(v['selected_points']) for v in values)==expected
            assert abs(sum(float(v['selected_share']) for v in values)-1)<1e-14
        labels=read_csv(OUT/r['example_labels_csv']);reference=np.load(OUT/f'data/r{n}/reference_coords.npy')
        for sid in [1000,1099,1199]:
            values=np.asarray(coeff[sid-1000]).reshape(211,211)
            change_x=np.abs(np.diff(values,axis=1))>1e-6;change_y=np.abs(np.diff(values,axis=0))>1e-6
            edge=np.zeros((211,211),dtype=bool)
            edge[:,:-1]|=change_x;edge[:,1:]|=change_x;edge[:-1,:]|=change_y;edge[1:,:]|=change_y
            edge_points=common[edge.ravel()]
            direct=np.zeros(n*n,dtype=np.int8)
            # A bounded dense oracle verifies all query nodes for the three
            # predeclared figures, independently of cKDTree or EDT.
            for lo in range(0,n*n,128):
                hi=min(lo+128,n*n);q=reference[lo:hi]
                dx=q[:,None,0]-edge_points[None,:,0];dy=q[:,None,1]-edge_points[None,:,1]
                nearest_squared=np.min(dx*dx+dy*dy,axis=1)
                direct[lo:hi]=(nearest_squared<=.02**2).astype(np.int8)
            boundary=np.minimum.reduce([reference[:,0],reference[:,1],1-reference[:,0],1-reference[:,1]])<=.02
            direct[boundary]=2
            saved=sorted((row for row in labels if int(row['case_id'])==sid),key=lambda row:int(row['node_index']))
            assert len(saved)==n*n and np.array_equal(direct,np.array([int(row['region']) for row in saved]))
            oracles.append(dict(resolution=n,case_id=sid,all_nodes=n*n,exact_label_match=True,
                boundary_points=int((direct==2).sum()),interface_points=int((direct==1).sum()),edge_points=len(edge_points)))
        assert sha(OUT/f'records/r{n}/metrics_records.csv')==r['field_csv_sha256']
        assert sha(OUT/f'records/r{n}/region_records.csv')==r['legacy_region_csv_sha256']
        assert sha(OUT/r['physical_region_csv'])==r['physical_region_csv_sha256']
        reports.append(dict(resolution=n,region_rows=22200,physical_requests=7400,boundary_points=boundary_expected,
            all_region_partitions_and_counts_passed=True,common_region_field_errors_unchanged=True,field_csv_unchanged=True))
    payload=dict(passed=True,complete=True,region_protocol_id=receipt['region_protocol_id'],region_rows=66600,physical_requests=22200,
        resolutions=reports,direct_distance_oracles=oracles,oracle_nodes=sum(row['all_nodes'] for row in oracles),
        independent_oracle='Explicit squared distances to all shared coefficient-edge nodes in bounded chunks, plus direct domain boundary distance',
        source_sha256=sha(__file__),physical_region_acceptance_sha256=sha(OUT/'results/physical_region_acceptance.json'),
        verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    path=OUT/'results/physical_region_independent_checks.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(path);print(path)


if __name__=='__main__':main()
