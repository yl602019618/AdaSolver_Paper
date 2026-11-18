"""Recompute matched Darcy scalar tables from public lossless CSVs; no GPU needed."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,gzip,hashlib,io,json,math,statistics,datetime as dt

RESOLUTIONS=(85,141,211)
METHODS=('uniform','pi','cpg','ca')
BUDGETS=(.375,.5,.75)

def digest(payload):return hashlib.sha256(payload).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def raw_payload(path):
    path=Path(path)
    if not path.exists():path=Path(str(path)+'.gz')
    raw=path.read_bytes();raw=gzip.decompress(raw) if path.suffix=='.gz' else raw
    return raw
def rows(path):return list(csv.DictReader(io.StringIO(raw_payload(path).decode(),newline='')))
def close(a,b):
    difference=abs(float(a)-float(b));assert difference<=2e-12+2e-11*abs(float(b)),(a,b,difference)
    return difference
def area(values):return .125*(values[0]+values[1])/2+.25*(values[1]+values[2])/2

def verify(package):
    package=Path(package).resolve()
    manifest_path=package/'release_manifest.json'
    if not manifest_path.exists():manifest_path=package/'payload_manifest.json'
    manifest=read(manifest_path)
    assert manifest['status']=='verified_complete_resolution_matched_training'
    files=manifest['files']
    for item in files:
        path=package/item['path'];assert path.is_file(),item['path']
        payload=path.read_bytes();assert digest(payload)==item['sha256'],item['path']
        assert len(payload)==item['bytes']
        if item.get('encoding')=='gzip':
            plain=gzip.decompress(payload);assert digest(plain)==item['source_sha256']
    assert read(package/'results/final_acceptance.json')['passed']
    region_acceptance=read(package/'results/physical_region_acceptance.json')
    assert region_acceptance['passed'] and region_acceptance['complete']
    assert read(package/'results/physical_region_independent_checks.json')['passed']
    region_id='physical_bands_0p02_direct_queries_v1_20261119'
    assert region_acceptance['region_protocol_id']==region_id
    assert read(package/'config/region_protocol_physical_20261119.json')['region_protocol_id']==region_id
    protocol=read(package/'protocol.json');assert digest((package/'protocol.json').read_bytes())==manifest['protocol_sha256']
    split=read(package/'split_manifest.json')
    assert split['training_ids']==list(range(900)) and split['validation_ids']==list(range(900,1000))
    assert split['test_ids']==list(range(1000,1200)) and split['test_source_rows']==list(range(200))
    cases=defaultdict(list);raw_count=0;region_count=0;physical_count=0;regional=defaultdict(list);models={};label_count=0
    for n in RESOLUTIONS:
        b=read(package/f'checkpoints/r{n}/manifest.json');h=read(package/f'checkpoints/r{n}/head/manifest.json')
        assert b['execution_status']==h['execution_status']=='completed'
        assert b['epochs_completed']==h['epochs_completed']==1000
        assert b['total_optimization_steps']==h['total_optimization_steps']==225000
        assert b['trained_from_scratch'] and b['training_seed']==0
        assert b['nfull']==n*n and b['train_coefficient_shape']==[900,n*n,1]
        assert b['training_ids']==h['training_ids']==list(range(900))
        assert b['validation_ids']==h['validation_ids']==list(range(900,1000))
        assert b['test_used_for_selection'] is False and h['test_used_for_selection'] is False
        assert b['checkpoint_sha256']==h['operator_sha256'] and b['normalizer_sha256']==h['normalizer_sha256']
        models[n]=b
        metrics=rows(package/f'records/r{n}/metrics_records.csv');assert len(metrics)==14800
        ids=set();keys=set();request_keys={}
        for row in metrics:
            seed=None if row['sampling_seed']=='' else int(row['sampling_seed'])
            key=(int(row['case_id']),row['method'],float(row['budget_fraction']),seed,row['evaluation_grid'])
            assert key not in keys;keys.add(key);ids.add(row['physical_request_id'])
            request_key=key[:-1]
            assert request_keys.get(row['physical_request_id'],request_key)==request_key
            request_keys[row['physical_request_id']]=request_key
            assert row['checkpoint_sha256']==b['checkpoint_sha256'] and row['normalizer_sha256']==b['normalizer_sha256']
            assert row['protocol_sha256']==manifest['protocol_sha256'] and row['training_seed']=='0'
            assert int(row['training_resolution'])==int(row['observation_resolution'])==n
            assert int(row['evaluation_resolution'])==(211 if row['evaluation_grid']=='common' else n)
            assert row['dataset']=='darcy' and row['metric']=='field_relative_l2' and row['metric_dtype']=='float64'
            assert int(row['coarse_points'])==(round(n*n*.1) if row['method'] in ['cpg','ca'] else 0)
            if row['method'] in ['full','uniform']:assert row['beta']==''
            else:assert float(row['beta'])==1.
            assert row['head_sha256']==(h['checkpoint_sha256'] if row['method']=='ca' else '')
            assert int(row['retained_points'])==round(n*n*float(row['budget_fraction']))
            assert row['status']=='ok' and math.isfinite(float(row['value']))
            cases[(n,key[-1],key[1],key[2],seed)].append((key[0],float(row['value'])))
        expected={(case,m,budget,seed,grid) for case in range(1000,1200) for m in ('full',)+METHODS
            for budget in ([1.] if m=='full' else BUDGETS) for seed in ([None] if m=='full' else [0,1,2])
            for grid in ['common','native']}
        assert keys==expected and len(ids)==7400
        raw_count+=len(metrics);physical_count+=len(ids)
        region_receipt=read(package/f'results/physical_region_r{n}_acceptance.json')
        assert region_receipt['passed'] and region_receipt['region_protocol_id']==region_id
        assert region_receipt['all_7400_retained_index_hashes_verified'] and region_receipt['all_200_common_masks_unchanged']
        for name,key in [('metrics_records.csv','field_csv_sha256'),('region_records.csv','legacy_region_csv_sha256'),
                         ('physical_region_records.csv','physical_region_csv_sha256')]:
            assert digest(raw_payload(package/f'records/r{n}'/name))==region_receipt[key]
        legacy_region_rows=rows(package/f'records/r{n}/region_records.csv');assert len(legacy_region_rows)==22200
        region_rows=rows(package/f'records/r{n}/physical_region_records.csv');assert len(region_rows)==22200
        region_count+=len(region_rows)
        for row in region_rows:
            assert row['region_protocol_id']==region_id and row['legacy_region_protocol_id']=='common211_boolean_nearest_mapping_v0'
            seed=None if row['sampling_seed']=='' else int(row['sampling_seed'])
            key=(n,row['method'],float(row['budget_fraction']),seed,row['region'])
            assert request_keys[row['physical_request_id']]==(int(row['case_id']),row['method'],float(row['budget_fraction']),seed)
            assert row['region'] in ['boundary','interface','smooth']
            observed=int(row['observed_points']);selected=int(row['selected_points'])
            assert 0<=selected<=observed
            if observed:close(row['retained_fraction'],selected/observed)
            else:assert row['retained_fraction']==''
            regional[key].append(row)
        label_path=package/f'results/physical_region_labels_r{n}.csv'
        assert digest(raw_payload(label_path))==region_receipt['example_labels_csv_sha256']
        labels=rows(label_path);assert len(labels)==3*n*n
        assert {int(row['case_id']) for row in labels}=={1000,1099,1199}
        for row in labels:
            x=float(row['x']);y=float(row['y']);boundary=min(x,y,1-x,1-y)<=.02
            assert (int(row['region'])==2)==boundary
            assert row['region_protocol_id']==region_id
        label_count+=len(labels)
    assert raw_count==44400 and physical_count==22200 and region_count==66600
    means={}
    for key,values in cases.items():
        assert len(values)==200 and {x[0] for x in values}==set(range(1000,1200))
        means[key]=math.fsum(x[1] for x in values)/200
    ratios={};differences=[]
    for n in RESOLUTIONS:
        for grid in ['common','native']:
            for method in METHODS:
                for seed in [0,1,2]:
                    numerator=area([means[n,grid,method,b,seed] for b in BUDGETS])
                    denominator=area([means[n,grid,'uniform',b,seed] for b in BUDGETS])
                    assert denominator>0;ratios[n,grid,method,seed]=(numerator,denominator,numerator/denominator)
    budget_rows=rows(package/'results/budget_metrics.csv');assert len(budget_rows)==216
    for r in budget_rows:
        key=(int(r['resolution']),r['evaluation_grid'],r['method'],float(r['budget_fraction']),int(r['sampling_seed']))
        differences.append(close(r['error_mean'],means[key]))
        differences.append(close(r['full_error'],means[key[0],key[1],'full',1.,None]))
    area_rows=rows(package/'results/rauec_by_seed.csv');assert len(area_rows)==72
    for r in area_rows:
        key=(int(r['resolution']),r['evaluation_grid'],r['method'],int(r['sampling_seed']))
        for field,value in zip(['auec','matched_uniform_auec','rauec'],ratios[key]):differences.append(close(r[field],value))
        assert r['full_in_integral']=='False'
    summaries=rows(package/'results/rauec_summary.csv');assert len(summaries)==24
    for r in summaries:
        key=(int(r['resolution']),r['evaluation_grid'],r['method'])
        assert r['checkpoint_sha256']==models[key[0]]['checkpoint_sha256']
        assert r['normalizer_sha256']==models[key[0]]['normalizer_sha256']
        assert r['protocol_sha256']==manifest['protocol_sha256']
        values=[ratios[key+(seed,)][2] for seed in [0,1,2]]
        differences.append(close(r['rauec_mean'],statistics.mean(values)))
        differences.append(close(r['sampling_sd'],statistics.stdev(values)))
        assert r['training_seed_count']=='1' and r['sampling_seed_count']=='3'
        if key[2]=='uniform':assert values==[1.,1.,1.]
    absolute=rows(package/'results/absolute_errors.csv');assert len(absolute)==78
    for r in absolute:
        n=int(r['resolution']);grid=r['evaluation_grid'];method=r['method'];budget=float(r['budget_fraction'])
        values=[means[n,grid,method,budget,seed] for seed in ([None] if method=='full' else [0,1,2])]
        differences.append(close(r['error_mean'],statistics.mean(values)))
        if method=='full':assert r['sampling_sd']==''
        else:differences.append(close(r['sampling_sd'],statistics.stdev(values)))
    # Recompute regional retention from original counts, separately by sampling seed.
    retained={}
    for key,records in regional.items():
        assert len(records)==200 and {int(r['case_id']) for r in records}==set(range(1000,1200))
        valid=[int(r['selected_points'])/int(r['observed_points']) for r in records if int(r['observed_points'])]
        retained[key]=None if not valid else statistics.mean(valid)
    regional_seed_rows=rows(package/'results/regional_by_seed.csv');assert len(regional_seed_rows)==333
    for r in regional_seed_rows:
        assert r['region_protocol_id']==region_id
        seed=None if r['sampling_seed']=='' else int(float(r['sampling_seed']))
        key=(int(r['resolution']),r['method'],float(r['budget_fraction']),seed,r['region'])
        value=retained[key]
        if value is None:assert r['mean_case_retained_fraction']==''
        else:
            differences.append(close(r['mean_case_retained_fraction'],value))
            differences.append(close(r['mean_case_retention_enrichment'],value/float(r['actual_budget_fraction'])))
    result=dict(passed=True,checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),physical_requests=physical_count,
        common_native_metric_rows=raw_count,regional_rows=region_count,legacy_regional_rows_preserved=66600,verified_per_seed_ratios=72,
        verified_rauec_summary_rows=24,verified_absolute_rows=78,verified_regional_seed_rows=333,
        maximum_scalar_difference=max(differences),package_files_checked=len(files),strict_example_boundary_nodes_checked=label_count,
        region_protocol_id=region_id,
        scope='Public scalar and metadata reproduction; does not rerun model training, inference, or private selected-point arrays',
        no_gpu=True,training_seed_count=1,sampling_seed_count=3,protocol_sha256=manifest['protocol_sha256'])
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path);args=parser.parse_args();result=verify(args.package)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
