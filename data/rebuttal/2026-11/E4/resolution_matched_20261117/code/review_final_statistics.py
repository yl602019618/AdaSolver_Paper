"""Independent closeout of the complete field and strict-regional evidence."""
from pathlib import Path
from collections import defaultdict
import csv,datetime as dt,hashlib,json,math,statistics,time

OUT=Path(__file__).resolve().parents[1]


def read(path):
    for attempt in range(5):
        try:return json.loads(Path(path).read_text())
        except OSError:
            if attempt==4:raise
            time.sleep(.2)


def rows(path):
    for attempt in range(5):
        try:
            with Path(path).open() as f:return list(csv.DictReader(f))
        except OSError:
            if attempt==4:raise
            time.sleep(.2)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def close(a,b):
    delta=abs(float(a)-float(b));assert delta<=2e-12+2e-11*abs(float(b)),(a,b,delta)
    return delta


def main():
    stat=read(OUT/'results/statistics_acceptance.json');strict=read(OUT/'results/physical_region_independent_checks.json')
    assert stat['passed'] and stat['complete'] and strict['passed'] and strict['complete']
    official={(int(r['resolution']),r['evaluation_grid'],r['method']):r for r in rows(OUT/'results/rauec_summary.csv')}
    regions_official={(int(r['resolution']),r['method'],float(r['budget_fraction']),r['region']):r for r in rows(OUT/'results/regional_summary.csv')}
    assert len(official)==24 and len(regions_official)==117
    final=[];fulls=[];counts=[];region_conditions=[];max_difference=0.;sources=[]
    for n in [85,141,211]:
        gate=read(OUT/f'results/r{n}_accuracy_acceptance.json');pr=read(OUT/f'results/physical_region_r{n}_acceptance.json')
        path=OUT/f'records/r{n}/metrics_records.csv';rp=OUT/f'records/r{n}/physical_region_records.csv'
        assert gate['passed'] and pr['passed'] and sha(path)==gate['raw_csv_sha256']==pr['field_csv_sha256']
        assert sha(rp)==pr['physical_region_csv_sha256']
        raw=rows(path);regional=rows(rp);assert len(raw)==14800 and len(regional)==22200
        conditions=defaultdict(dict);requests=defaultdict(list)
        for r in raw:
            key=(r['evaluation_grid'],r['method'],float(r['budget_fraction']),r['sampling_seed'])
            case=int(r['case_id']);assert case not in conditions[key];conditions[key][case]=float(r['value'])
            assert r['checkpoint_sha256']==gate['checkpoint_sha256'] and r['normalizer_sha256']==gate['normalizer_sha256']
            assert r['training_seed']=='0' and r['metric']=='field_relative_l2'
            assert r['reconstruction_id']=='E4_linear_delaunay_nearest_original_rule'
            requests[r['physical_request_id']].append(r)
        assert len(conditions)==74 and len(requests)==7400
        for request,views in requests.items():
            assert len(views)==2 and {v['evaluation_grid'] for v in views}=={'common','native'}
            assert views[0]['selected_indices_sha256']==views[1]['selected_indices_sha256']
            if n==211:max_difference=max(max_difference,close(views[0]['value'],views[1]['value']))
        assert all(set(cases)==set(range(1000,1200)) for cases in conditions.values())
        means={key:math.fsum(cases.values())/200 for key,cases in conditions.items()}
        for grid in ['common','native']:
            fulls.append(dict(resolution=n,evaluation_grid=grid,field_relative_l2=means[grid,'full',1.,'']))
            for method in ['uniform','pi','cpg','ca']:
                ratios=[]
                for seed in ['0','1','2']:
                    # Three-node trapezoidal integration reduces to these
                    # exact binary weights: (1,3,2)/16.
                    numerator=math.fsum(w*means[grid,method,b,seed] for b,w in [(.375,1/16),(.5,3/16),(.75,2/16)])
                    denominator=math.fsum(w*means[grid,'uniform',b,seed] for b,w in [(.375,1/16),(.5,3/16),(.75,2/16)])
                    ratios.append(numerator/denominator)
                mean=statistics.mean(ratios);sd=statistics.stdev(ratios);reported=official[n,grid,method]
                max_difference=max(max_difference,close(mean,reported['rauec_mean']),close(sd,reported['sampling_sd']))
                assert reported['sampling_seed_count']=='3' and reported['training_seed_count']=='1'
                if method=='uniform':assert ratios==[1.,1.,1.]
                final.append(dict(resolution=n,evaluation_grid=grid,method=method,rauec_mean=mean,sampling_sd=sd,
                    seed_values=ratios,seeds_below_one=sum(value<1 for value in ratios)))
        regional_groups=defaultdict(list)
        for r in regional:
            assert r['region_protocol_id']==strict['region_protocol_id']
            if r['method']=='full':continue
            key=(r['method'],float(r['budget_fraction']),r['sampling_seed'],r['region'])
            regional_groups[key].append(r)
        assert len(regional_groups)==108 and all(len(v)==200 for v in regional_groups.values())
        for method in ['uniform','pi','cpg','ca']:
            for budget in [.375,.5,.75]:
                for region in ['boundary','interface','smooth']:
                    enrich=[];pooled=[]
                    for seed in ['0','1','2']:
                        rr=regional_groups[method,budget,seed,region]
                        valid=[r for r in rr if int(r['observed_points'])]
                        ratio=float(rr[0]['actual_budget_fraction'])
                        enrich.append(statistics.mean(float(r['retained_fraction']) for r in valid)/ratio)
                        pooled.append(sum(int(r['selected_points']) for r in rr)/sum(int(r['observed_points']) for r in rr)/ratio)
                    reported=regions_official[n,method,budget,region]
                    mean=statistics.mean(enrich);pool=statistics.mean(pooled)
                    max_difference=max(max_difference,close(mean,reported['mean_case_retention_enrichment']),
                        close(pool,reported['pooled_retention_enrichment']),close(statistics.stdev(enrich),reported['mean_case_retention_enrichment_sampling_sd']))
                    region_conditions.append(dict(resolution=n,method=method,budget_fraction=budget,region=region,
                        mean_case_enrichment=mean,pooled_enrichment=pool,sampling_sd=statistics.stdev(enrich)))
        counts.append(dict(resolution=n,physical_requests=len(requests),field_metric_rows=len(raw),regional_rows=len(regional)))
        sources.append(dict(resolution=n,field_csv_sha256=sha(path),physical_region_csv_sha256=sha(rp)))
    claim_counts=[]
    for method in ['pi','cpg','ca']:
        rr=[r for r in region_conditions if r['method']==method]
        boundary=[r['mean_case_enrichment'] for r in rr if r['region']=='boundary']
        smooth=[r['mean_case_enrichment'] for r in rr if r['region']=='smooth']
        interface=[r['mean_case_enrichment'] for r in rr if r['region']=='interface']
        assert len(boundary)==len(smooth)==len(interface)==9
        claim_counts.append(dict(method=method,condition_count=9,boundary_above_one=sum(v>1 for v in boundary),
            smooth_below_one=sum(v<1 for v in smooth),interface_above_one=sum(v>1 for v in interface),
            boundary_range=[min(boundary),max(boundary)],smooth_range=[min(smooth),max(smooth)],interface_range=[min(interface),max(interface)]))
    table=rows(OUT/'results/table_source.csv');assert len(table)==36
    main=[r for r in table if r['table']=='rebuttal_main']
    assert len(main)==12 and {r['column'] for r in main}=={'Original 85² (historical)','New 141²','New 211²'}
    for r in main:
        assert (r['resolution']=='85' and r['evaluation_grid']=='historical' and r['status']=='historical_reference') or (r['resolution'] in ['141','211'] and r['evaluation_grid']=='common' and r['status']=='accepted_matched')
    conclusion=(OUT/'results/conclusion_zh.md').read_text()
    for r in claim_counts:
        assert f"{r['method'].upper()}的边界保留率高于全局预算有{r['boundary_above_one']}/9个条件，平滑区低于全局预算有{r['smooth_below_one']}/9个条件" in conclusion
    payload=dict(passed=True,complete=True,reviewed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        counts=counts,field_metric_rows=44400,physical_requests=22200,regional_rows=66600,
        rauec=final,full_errors=fulls,regional_conditions=region_conditions,regional_claim_counts=claim_counts,
        max_summary_absolute_difference=max_difference,source_bindings=sources,
        region_protocol_id=strict['region_protocol_id'],training_seeds=[0],sampling_seeds=[0,1,2],
        original85_only_historical_in_main=True,new85_only_internal_control=True,
        statistics_acceptance_sha256=sha(OUT/'results/statistics_acceptance.json'),
        strict_regional_independent_checks_sha256=sha(OUT/'results/physical_region_independent_checks.json'),source_sha256=sha(__file__))
    path=OUT/'results/independent_closeout_review.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');tmp.replace(path);print(path)
    for row in final:
        if row['evaluation_grid']=='common':print(row)
    print(claim_counts)


if __name__=='__main__':main()
