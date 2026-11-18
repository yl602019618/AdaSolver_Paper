"""Reclassify saved samples using the same physical bands at every resolution.

No model prediction, retained point, score, target density, or field metric is
changed. The original nearest-mapped region records remain as legacy evidence.
"""
from pathlib import Path
import argparse,csv,datetime as dt,hashlib,json,os,sys,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

OUT=Path(__file__).resolve().parents[1]
REGIONS=('boundary','interface','smooth')
ID='physical_bands_0p02_direct_queries_v1_20261119'


def retry(operation):
    for attempt in range(5):
        try:return operation()
        except OSError:
            if attempt==4:raise
            time.sleep(.2*(attempt+1))


def read(path):return retry(lambda:json.loads(Path(path).read_text()))
def digest(path):
    def operation():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    return retry(operation)


def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    def operation():tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    retry(operation)


def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    def operation():
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        tmp.replace(path)
    retry(operation)


def source_edges(coefficient):
    values=np.asarray(coefficient).reshape(211,211);edge=np.zeros_like(values,dtype=bool)
    dx=np.abs(values[:,1:]-values[:,:-1])>1e-6;dy=np.abs(values[1:]-values[:-1])>1e-6
    edge[:,1:]|=dx;edge[:,:-1]|=dx;edge[1:]|=dy;edge[:-1]|=dy
    return edge


def physical_masks(common_coefficient,common_coordinates,query_coordinates):
    """Distances use the actual FP64-promoted physical coordinate values."""
    edge=source_edges(common_coefficient);q=np.asarray(query_coordinates,dtype=np.float64)
    boundary=np.min(np.column_stack((q,1-q)),axis=1)<=.02
    if edge.any():distance=cKDTree(np.asarray(common_coordinates)[edge.ravel()]).query(q,k=1)[0]
    else:distance=np.full(len(q),np.inf)
    interface=(distance<=.02)&~boundary
    return {'boundary':boundary,'interface':interface,'smooth':~boundary&~interface}


def legacy_masks(coefficient,coordinates):
    edge=source_edges(coefficient)
    boundary=np.min(np.column_stack((coordinates,1-coordinates)),axis=1)<=.02
    interface=(distance_transform_edt(~edge,sampling=1/210).ravel()<=.02)&~boundary
    return {'boundary':boundary,'interface':interface,'smooth':~boundary&~interface}


def labels(masks):
    out=np.zeros(len(masks['boundary']),dtype=np.int8);out[masks['interface']]=1;out[masks['boundary']]=2
    return out


def near(actual,expected,name):
    if expected in ['',None]:assert actual is None,(name,actual,expected);return 0.
    assert actual is not None,name
    difference=abs(float(actual)-float(expected))
    assert difference<=1e-12+1e-10*abs(float(expected)),(name,actual,expected,difference)
    return difference


def recompute(n,protocol):
    gate=read(OUT/f'results/r{n}_accuracy_acceptance.json');assert gate['passed']
    directory=OUT/f'records/r{n}'
    field=directory/'metrics_records.csv';oldpath=directory/'region_records.csv';path=directory/'physical_region_records.csv'
    field_hash=digest(field);old_hash=digest(oldpath)
    assert field_hash==gate['raw_csv_sha256'] and old_hash==gate['region_csv_sha256']
    receipt_path=OUT/f'results/physical_region_r{n}_acceptance.json'
    if receipt_path.exists():
        old=read(receipt_path)
        if old.get('passed') and old['region_protocol_id']==ID and old['field_csv_sha256']==field_hash and old['legacy_region_csv_sha256']==old_hash:
            assert old['physical_region_csv_sha256']==digest(path)
            assert old['source_sha256']==digest(__file__)
            return old
    with oldpath.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==22200
    by_request={}
    for row in rows:
        key=row['physical_request_id'];by_request.setdefault(key,{})[row['region']]=row
    assert len(by_request)==7400 and all(set(r)==set(REGIONS) for r in by_request.values())
    common=np.load(OUT/'data/r211/reference_coords.npy');native=np.load(OUT/f'data/r{n}/reference_coords.npy')
    common_coeff=np.load(OUT/'data/r211/test_coeff.npy',mmap_mode='r');native_coeff=np.load(OUT/f'data/r{n}/test_coeff.npy',mmap_mode='r')
    geometry=np.load(OUT/f'evaluation/r{n}_patch_geometry.npz');pid=geometry['patch_id']
    mapping=cKDTree(common).query(native,k=1)[1]
    sys.path.insert(0,str(OUT/'code/vendor'))
    from adasolver.inference.pi.darcy import coefficient_interface_distance
    source_manifest=read(OUT/'config/source_manifest.json')['files']
    assert digest(OUT/'code/vendor/adasolver/inference/pi/darcy.py')==source_manifest['inference/pi/darcy.py']
    output=[];geometry_rows=[];examples=[];max_legacy_difference=0.;started=time.perf_counter()
    for sid in range(1000,1200):
        coefficient=np.asarray(common_coeff[sid-1000]).reshape(-1)
        cm=physical_masks(coefficient,common,common);original=legacy_masks(coefficient,common)
        assert all(np.array_equal(cm[key],original[key]) for key in REGIONS),'Common regional field metrics require recomputation'
        strict=physical_masks(coefficient,common,native);legacy={key:original[key][mapping] for key in REGIONS}
        assert np.array_equal(sum(strict[key].astype(np.int8) for key in REGIONS),np.ones(n*n,dtype=np.int8))
        changes=int(np.count_nonzero(labels(strict)!=labels(legacy)))
        geometry_rows.append(dict(resolution=n,case_id=sid,changed_observation_labels=changes,
            **{f'{key}_{scope}_points':int(masks[key].sum()) for scope,masks in [('legacy',legacy),('physical',strict)] for key in REGIONS}))
        if sid in [1000,1099,1199]:
            lab=labels(strict);previous=labels(legacy)
            examples.extend(dict(case_id=sid,resolution=n,node_index=i,x=float(native[i,0]),y=float(native[i,1]),
                region=int(lab[i]),legacy_region=int(previous[i]),region_protocol_id=ID) for i in range(n*n))
        case=read(directory/f'all/case{sid}.json');assert case['execution_status']=='completed'
        distances=coefficient_interface_distance(np.asarray(native_coeff[sid-1000]).reshape(-1),shape=(n,n))
        pi_score=1.+.25*np.exp(-distances/(2/84))
        nearest_cache={}
        for record in case['records']:
            method=record['method'];identity=record['request_id'];old_regions=by_request[identity]
            with np.load(OUT/record['source_array'],allow_pickle=False) as a:
                selected=a['indices'];coarse=a['coarse_indices'];k=len(selected);ratio=k/(n*n)
                assert hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest()==record['selected_indices_sha256']
                assert len(np.unique(selected))==k and k==record['retained_points']
                if method in ['full','uniform']:
                    support=np.arange(n*n);score=np.ones(n*n);display=score;target=score;allocation=np.full(n*n,ratio)
                elif method=='pi':
                    support=np.arange(n*n);score=pi_score;display=score/score.mean();target=score/score.sum()*(n*n);allocation=None
                else:
                    support=coarse;score=a['coarse_score'];key=record['coarse_indices_sha256']
                    if key not in nearest_cache:nearest_cache[key]=cKDTree(native[coarse]).query(native,k=1)[1]
                    display=score[nearest_cache[key]]/max(float(score.mean()),1e-12)
                    allocation=a['target_patch_counts'][pid]/a['patch_sizes'][pid];target=allocation/ratio
                    assert int(a['target_patch_counts'].sum())==k
                def statistics(mask):
                    available=int(mask.sum());chosen=int(mask[selected].sum());on_support=mask[support]
                    return dict(observed_points=available,selected_points=chosen,
                        retained_fraction=None if not available else chosen/available,selected_share=chosen/k,
                        support_score_points=int(on_support.sum()),mean_support_score=None if not on_support.any() else float(score[on_support].mean()),
                        mean_display_relative_score=None if not available else float(display[mask].mean()),
                        mean_target_relative_density=None if not available else float(target[mask].mean()),
                        mean_target_allocation_rate=None if not available or allocation is None else float(allocation[mask].mean()))
                for region in REGIONS:
                    old=old_regions[region];replayed=statistics(legacy[region])
                    for key,value in replayed.items():max_legacy_difference=max(max_legacy_difference,near(value,old[key],key))
                    fresh=dict(old);fresh.update(statistics(strict[region]));fresh['region_protocol_id']=ID
                    fresh['legacy_region_protocol_id']='common211_boolean_nearest_mapping_v0'
                    assert int(fresh['evaluation_points'])==int(cm[region].sum())
                    output.append(fresh)
        if (sid-999)%25==0:print(f'physical regions r{n}: {sid-999}/200 cases',flush=True)
    assert len(output)==22200
    assert digest(field)==field_hash and digest(oldpath)==old_hash
    save_csv(path,output)
    geometry_path=OUT/f'results/physical_region_geometry_r{n}.csv';save_csv(geometry_path,geometry_rows)
    example_path=OUT/f'results/physical_region_labels_r{n}.csv';save_csv(example_path,examples)
    result=dict(passed=True,resolution=n,region_protocol_id=ID,physical_requests=7400,region_rows=22200,cases=200,
        source_sha256=digest(__file__),protocol_sha256=digest(OUT/'config/region_protocol_physical_20261119.json'),
        field_csv_sha256=field_hash,legacy_region_csv_sha256=old_hash,physical_region_csv_sha256=digest(path),
        physical_region_csv=str(path.relative_to(OUT)),geometry_csv=str(geometry_path.relative_to(OUT)),geometry_csv_sha256=digest(geometry_path),
        example_labels_csv=str(example_path.relative_to(OUT)),example_labels_csv_sha256=digest(example_path),
        all_200_common_masks_unchanged=True,common_region_field_metric_reused_with_identical_masks=True,
        all_7400_retained_index_hashes_verified=True,legacy_statistics_reproduced=True,max_legacy_statistic_absolute_difference=max_legacy_difference,
        field_metrics_unchanged=True,legacy_region_records_preserved=True,
        changed_observation_region_labels_total=sum(r['changed_observation_labels'] for r in geometry_rows),
        seconds=time.perf_counter()-started,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(receipt_path,result);return result


def main(resolutions):
    protocol_path=OUT/'config/region_protocol_physical_20261119.json'
    protocol=dict(region_protocol_id=ID,boundary_width=.02,interface_width=.02,
        boundary='Direct minimum distance of each physical query coordinate to x/y domain boundaries <=0.02',
        interface='Exact Euclidean nearest distance to adjacent coefficient-change nodes on the same common211 grid <=0.02, excluding boundary',
        smooth='Remaining query nodes',region_codes={'smooth':0,'interface':1,'boundary':2},
        coordinates='Actual model FP32 coordinate values promoted unchanged to FP64',
        source_geometry='Same common211 coefficient case for all three observation resolutions',
        correction='Remove nearest-boolean-mask classification artifacts; 141 boundary includes301 out-of-band observations under the legacy mapping',
        unchanged=['backbone','head','prediction','retained_indices','refinement_score','target_density','full_field_metric','rAUEC'],
        legacy_records='records/rN/region_records.csv remain unchanged',main_records='records/rN/physical_region_records.csv',
        data_selection='All200 declared test cases, all5 methods, all3 budgets and sampling seeds; no favorable-condition filtering')
    if protocol_path.exists():assert read(protocol_path)==protocol
    else:write(protocol_path,protocol)
    results=[recompute(n,protocol) for n in resolutions]
    if resolutions==[85,141,211]:
        write(OUT/'results/physical_region_acceptance.json',dict(passed=True,complete=True,region_protocol_id=ID,
            protocol_sha256=digest(protocol_path),source_sha256=digest(__file__),resolutions=results,
            physical_requests=22200,region_rows=66600,all_common_masks_unchanged=True,field_metrics_unchanged=True))
    print(json.dumps(dict(passed=True,resolutions=resolutions,region_protocol_id=ID)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolutions',type=int,nargs='+',choices=[85,141,211],default=[85,141,211])
    main(parser.parse_args().resolutions)
