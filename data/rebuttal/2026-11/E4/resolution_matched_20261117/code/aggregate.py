"""Derive seed-first rAUEC and regional summaries from accepted raw records."""
from __future__ import annotations
import argparse,csv,json,os,re,shutil
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='4'
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUT,CODE,read,write,digest

RESOLUTIONS=(85,141,211)
METHODS=('uniform','pi','cpg','ca')
BUDGETS=(.375,.5,.75)
GRID_LABEL={'common':'Common 211² output','native':'Native output'}
TABLE=OUT/'results/tables'


def save_csv(name,rows):
    path=OUT/'results'/name;path.parent.mkdir(parents=True,exist_ok=True)
    columns={
        'budget_metrics.csv':['resolution','evaluation_grid','method','budget_fraction','sampling_seed','error_mean','case_count'],
        'absolute_errors.csv':['resolution','evaluation_grid','method','budget_fraction','error_mean','sampling_sd'],
        'rauec_by_seed.csv':['resolution','evaluation_grid','method','sampling_seed','auec','matched_uniform_auec','rauec'],
        'rauec_summary.csv':['resolution','evaluation_grid','method','rauec_mean','sampling_sd'],
        'regional_by_seed.csv':['resolution','method','budget_fraction','sampling_seed','region','mean_case_retained_fraction'],
        'regional_summary.csv':['resolution','method','budget_fraction','region','mean_case_retained_fraction','mean_case_retention_enrichment'],
        'checkpoint_registry.csv':['resolution','kind','checkpoint_path','checkpoint_sha256','training_wall_seconds','head_bank_seconds']}
    pd.DataFrame(rows,columns=None if rows else columns.get(name)).to_csv(path,index=False,float_format='%.17g')
    return path


def load_accepted(allow_partial):
    frames=[];regions=[];completion=[];accepted=[]
    for n in RESOLUTIONS:
        gate=OUT/f'results/r{n}_accuracy_acceptance.json'
        path=OUT/f'records/r{n}/metrics_records.csv'
        count=0
        if path.exists():
            frame=pd.read_csv(path);count=frame.physical_request_id.nunique()
        ok=False
        if gate.exists():
            receipt=read(gate)
            ok=receipt.get('passed') and receipt.get('status')=='complete_matched_resolution_accuracy'
            if ok:
                assert receipt['raw_csv_sha256']==digest(path)
                assert receipt['region_csv_sha256']==digest(OUT/f'records/r{n}/region_records.csv')
                region_receipt=read(OUT/f'results/physical_region_r{n}_acceptance.json')
                assert region_receipt['passed'] and region_receipt['field_metrics_unchanged']
                assert region_receipt['field_csv_sha256']==receipt['raw_csv_sha256']
                assert region_receipt['legacy_region_csv_sha256']==receipt['region_csv_sha256']
                region_path=OUT/f'records/r{n}/physical_region_records.csv'
                assert region_receipt['physical_region_csv_sha256']==digest(region_path)
                assert region_receipt['region_protocol_id']==read(OUT/'config/region_protocol_physical_20261119.json')['region_protocol_id']
                assert len(frame)==14800 and count==7400
                frames.append(frame);regions.append(pd.read_csv(region_path))
                accepted.append(n)
        completion.append(dict(resolution=n,expected_physical_requests=7400,actual_exported_physical_requests=count,
            expected_metric_rows=14800,accepted=bool(ok),status='complete' if ok else 'N/A: matched accuracy acceptance pending'))
    save_csv('matrix_completion.csv',completion)
    if not allow_partial and accepted!=list(RESOLUTIONS):
        raise RuntimeError(f'Accepted resolutions {accepted}; complete 85/141/211 required (or use --allow-partial)')
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),
            pd.concat(regions,ignore_index=True) if regions else pd.DataFrame(),accepted)


def metric_tables(frame):
    budget_rows=[];absolute=[];areas=[];summaries=[]
    if frame.empty:return budget_rows,absolute,areas,summaries
    for (n,grid),group in frame.groupby(['training_resolution','evaluation_grid'],sort=True):
        assert set(group.case_id)==set(range(1000,1200))
        identity={key:group[key].iloc[0] for key in ['checkpoint_sha256','normalizer_sha256','protocol_sha256','reconstruction_id']}
        assert all(group[key].nunique()==1 for key in identity)
        common=dict(resolution=int(n),evaluation_grid=grid,evaluation_resolution=211 if grid=='common' else int(n),
                    training_seed=0,metric='field_relative_l2',case_count=200,**identity)
        full=group[group.method=='full'];assert len(full)==200 and full.sampling_seed.isna().all()
        full_error=float(full.value.mean())
        absolute.append(dict(**common,method='full',budget_fraction=1.,retained_points=int(n*n),
            error_mean=full_error,sampling_sd=None,sampling_seed_count=0,full_error=full_error,
            full_case_sd=float(full.value.std(ddof=1)),beta=None))
        curves={}
        for method in METHODS:
            curve=[]
            for budget in BUDGETS:
                seed_errors=[]
                for seed in [0,1,2]:
                    condition=group[(group.method==method)&(group.budget_fraction==budget)&(group.sampling_seed==seed)]
                    assert len(condition)==200 and condition.case_id.nunique()==200
                    error=float(condition.value.mean());seed_errors.append(error)
                    curves[method,budget,seed]=error
                    budget_rows.append(dict(**common,method=method,budget_fraction=budget,sampling_seed=seed,
                        retained_points=int(round(n*n*budget)),actual_budget_fraction=round(n*n*budget)/(n*n),
                        coarse_points=round(.1*n*n) if method in ['cpg','ca'] else 0,
                        error_mean=error,case_sd=float(condition.value.std(ddof=1)),full_error=full_error,
                        beta=None if method=='uniform' else 1.))
                absolute.append(dict(**common,method=method,budget_fraction=budget,retained_points=int(round(n*n*budget)),
                    error_mean=float(np.mean(seed_errors)),sampling_sd=float(np.std(seed_errors,ddof=1)),
                    sampling_seed_count=3,full_error=full_error,full_case_sd=None,beta=None if method=='uniform' else 1.))
        for method in METHODS:
            ratios=[]
            for seed in [0,1,2]:
                vals=[curves[method,b,seed] for b in BUDGETS]
                denom=[curves['uniform',b,seed] for b in BUDGETS]
                area=sum((BUDGETS[i+1]-BUDGETS[i])*(vals[i]+vals[i+1])/2 for i in [0,1])
                uniform_area=sum((BUDGETS[i+1]-BUDGETS[i])*(denom[i]+denom[i+1])/2 for i in [0,1])
                assert uniform_area>0
                ratio=area/uniform_area;ratios.append(ratio)
                areas.append(dict(**common,method=method,sampling_seed=seed,auec=area,matched_uniform_auec=uniform_area,
                    rauec=ratio,budget_min=.375,budget_max=.75,budget_count=3,full_in_integral=False,
                    beta=None if method=='uniform' else 1.))
            summaries.append(dict(**common,method=method,rauec_mean=float(np.mean(ratios)),sampling_sd=float(np.std(ratios,ddof=1)),
                sampling_seed_count=3,training_seed_count=1,beta=None if method=='uniform' else 1.))
    return budget_rows,absolute,areas,summaries


def region_tables(frame):
    seeds=[];summaries=[]
    if frame.empty:return seeds,summaries
    groupkeys=['resolution','method','budget_fraction','sampling_seed','region']
    for keys,group in frame.groupby(groupkeys,dropna=False,sort=True):
        n,method,budget,seed,region=keys;assert len(group)==200 and group.case_id.nunique()==200
        observed=int(group.observed_points.sum());chosen=int(group.selected_points.sum())
        overall=float(group.actual_budget_fraction.iloc[0])
        row=dict(resolution=int(n),method=method,budget_fraction=budget,sampling_seed=None if pd.isna(seed) else int(seed),
            region=region,case_count=200,cases_with_region=int(group.retained_fraction.notna().sum()),
            region_protocol_id=group.region_protocol_id.iloc[0],
            actual_budget_fraction=overall,pooled_observed_points=observed,pooled_selected_points=chosen,
            mean_case_observed_points=float(group.observed_points.mean()),mean_case_selected_points=float(group.selected_points.mean()),
            mean_case_retained_fraction=float(group.retained_fraction.mean()),
            pooled_retained_fraction=None if not observed else chosen/observed,
            mean_case_retention_enrichment=float(group.retained_fraction.mean())/overall,
            pooled_retention_enrichment=None if not observed else chosen/observed/overall,
            mean_case_selected_share=float(group.selected_share.mean()),
            target_density_semantics=group.target_density_semantics.iloc[0])
        for source in ['mean_support_score','mean_display_relative_score','mean_target_relative_density','mean_target_allocation_rate','common_relative_l2']:
            row[source]=None if group[source].notna().sum()==0 else float(group[source].mean())
        seeds.append(row)
    per_seed=pd.DataFrame(seeds)
    metrics=['mean_case_retained_fraction','pooled_retained_fraction','mean_case_retention_enrichment','pooled_retention_enrichment',
        'mean_case_selected_points','mean_case_observed_points','mean_case_selected_share','mean_support_score',
        'mean_display_relative_score','mean_target_relative_density','mean_target_allocation_rate','common_relative_l2']
    for keys,group in per_seed.groupby(['resolution','method','budget_fraction','region'],sort=True):
        n,method,budget,region=keys
        assert len(group)==(1 if method=='full' else 3)
        row=dict(resolution=int(n),method=method,budget_fraction=budget,region=region,case_count=200,
            region_protocol_id=group.region_protocol_id.iloc[0],
            sampling_seed_count=0 if method=='full' else 3,actual_budget_fraction=float(group.actual_budget_fraction.iloc[0]),
            target_density_semantics=group.target_density_semantics.iloc[0])
        for metric in metrics:
            valid=group[metric].dropna()
            row[metric]=None if valid.empty else float(valid.mean())
            row[metric+'_sampling_sd']=float(valid.std(ddof=1)) if len(valid)>1 else None
        summaries.append(row)
    return seeds,summaries


def historical_reference():
    frozen=read(OUT/'config/historical85_reference.json')
    path=OUT.parent/'paper_git/tables/standard_benchmarks.tex'
    snapshot=OUT/'results/historical_source/standard_benchmarks.tex';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():shutil.copyfile(path,snapshot)
    assert digest(snapshot)==frozen['source_sha256'],'Historical table snapshot differs from the locked reference'
    source=snapshot.read_text();rows=[]
    for method in ['uniform','pi','cpg','ca']:
        if method=='uniform':mean,sd=1.,0.
        else:
            line=next(line for line in source.splitlines() if line.strip().startswith(method.upper()+' &'))
            values=re.findall(r'\d+\.\d+',line.split('&')[4]);assert len(values)==2
            mean,sd=map(float,values)
        assert mean==frozen['field_rauec'][method]['mean'] and sd==frozen['field_rauec'][method]['sampling_sd']
        rows.append(dict(method=method,resolution=85,rauec_mean=mean,sampling_sd=sd,
            identity='paper_original_85_historical_reference',evaluation_grid='historical_native85',
            source_path=str(path),source_snapshot=str(snapshot.relative_to(OUT)),source_sha256=digest(snapshot),
            comparability='historical training/checkpoint selection; PI beta0.5; native85 metric; excluded from controlled trends'))
    save_csv('historical_original85.csv',rows);return rows


def format_value(mean,sd,latex=False):
    if mean is None:return r'\text{N/A}' if latex else 'N/A — matched evaluation pending'
    if latex:return r'\begin{gathered}'+f'{mean:.4f}'+r'\\[-2pt]\pm '+f'{sd:.4f}'+r'\end{gathered}'
    return f'{mean:.4f} ± {sd:.4f}'


def table_files(summary,historical):
    TABLE.mkdir(parents=True,exist_ok=True)
    index={(r['resolution'],r['evaluation_grid'],r['method']):r for r in summary}
    historic={r['method']:r for r in historical};source_rows=[]
    specs=[('rebuttal_main',[(85,'historical','Original 85² (historical)'),(141,'common','New 141²'),(211,'common','New 211²')]),
           ('controlled_common',[(n,'common',f'Matched {n}²') for n in RESOLUTIONS]),
           ('controlled_native',[(n,'native',f'Matched {n}²') for n in RESOLUTIONS])]
    for name,columns in specs:
        md=['| Method | '+' | '.join(c[2] for c in columns)+' |','|---|'+'---:|'*len(columns)]
        tex=[r'\begin{tabular}{l'+'c'*len(columns)+'}',r'\toprule','Method & '+' & '.join(c[2].replace('²',r'$^2$') for c in columns)+r' \\',r'\midrule']
        for method in METHODS:
            markdown=[];latex=[]
            for n,grid,label in columns:
                row=historic[method] if grid=='historical' else index.get((n,grid,method))
                mean=None if row is None else row['rauec_mean'];sd=None if row is None else row['sampling_sd']
                markdown.append(format_value(mean,sd));latex.append('$'+format_value(mean,sd,True)+'$')
                source_rows.append(dict(table=name,column=label,method=method,resolution=n,evaluation_grid=grid,
                    rauec_mean=mean,sampling_sd=sd,status='N/A: matched evaluation pending' if row is None else 'historical_reference' if grid=='historical' else 'accepted_matched'))
            label='Uniform' if method=='uniform' else method.upper()
            md.append('| '+label+' | '+' | '.join(markdown)+' |')
            tex.append(label+' & '+' & '.join(latex)+r' \\')
        tex.extend([r'\bottomrule',r'\end{tabular}'])
        note='Mean ± sample SD across three sampling seeds for one training seed. Uniform is normalized within each model and evaluation grid; Full is excluded.'
        if name=='rebuttal_main':
            note+=' Original 85² is the paper historical reference (native 85² evaluation, historical training/selection and PI β=0.5). New 141²/211² use the matched 900/100 split, PI β=1 and common 211² evaluation. Use the internal matched 85² table for controlled resolution trends.'
        else:note+=' '+GRID_LABEL['native' if name.endswith('native') else 'common']+'. All columns use the same physical cases and matched training protocol.'
        (TABLE/f'{name}.md').write_text('\n'.join(md)+'\n\n'+note+'\n')
        (TABLE/f'{name}.tex').write_text('\n'.join(tex)+'\n% '+note+'\n')
    save_csv('table_source.csv',source_rows)


def checkpoint_registry(accepted):
    rows=[]
    for n in accepted:
        b=read(OUT/f'checkpoints/r{n}/manifest.json');h=read(OUT/f'checkpoints/r{n}/head/manifest.json')
        bank=read(OUT/f'head_banks/r{n}/manifest.json')
        for kind,m,path in [('backbone',b,OUT/f'checkpoints/r{n}/best.pt'),('ca_head',h,OUT/f'checkpoints/r{n}/head/best.pt')]:
            rows.append(dict(resolution=n,kind=kind,training_seed=0,checkpoint_path=str(path),checkpoint_sha256=m['checkpoint_sha256'],
                backbone_sha256=b['checkpoint_sha256'],normalizer_sha256=b['normalizer_sha256'],
                manifest_path=str(path.parent/'manifest.json'),manifest_sha256=digest(path.parent/'manifest.json'),
                training_ids='smooth1 rows[0,900)',validation_ids='smooth1 rows[900,1000)',test_ids='smooth2 rows[0,200)',
                epochs_completed=m.get('epochs_completed'),optimizer_updates=m.get('total_optimization_steps',m.get('optimization_steps')),
                selected_epoch=m.get('best_epoch',m.get('selected_epoch')),parameters=m.get('parameter_count'),
                training_wall_seconds=m.get('training_wall_seconds',m.get('training_seconds',m.get('elapsed_seconds'))),
                head_bank_seconds=bank.get('bank_seconds') if kind=='ca_head' else None,
                model_class=m.get('model_class',kind)))
    save_csv('checkpoint_registry.csv',rows)


def generate_report(accepted,summary,region_summary):
    completed=accepted==list(RESOLUTIONS)
    lines=['# Darcy resolution-matched training experiment','',
        'Status: '+('all three accuracy matrices independently accepted; statistics and figure acceptance are recorded separately in results/statistics_acceptance.json and results/figures_receipt.json.' if completed else 'in progress; only accepted resolutions are summarized.'),'',
        'The three models are trained from scratch at 85², 141² and 211², using direct stride 5/3/2 nodes from the same 421² source cases. Each model uses smooth1 rows 0–899 for training and 900–999 for validation; smooth2 rows 0–199 are used for final evaluation only. The historical test population had been viewed in previous work and is not claimed to be a previously unseen holdout.','',
        'The new 85² model is an internal matched control. The author-facing main table retains the original paper 85² as a labelled historical reference and adds 141²/211². Its historical and new columns differ in training/selection, PI β and evaluation grid; controlled resolution conclusions use all three newly trained models.','',
        'Common-grid relative L2 is measured after reconstruction to 211². Native-grid values are stored separately. The original E4 linear-Delaunay plus nearest-fallback reconstruction, original flattened PI distance rule and original CPG grad_lap definition are preserved. Full native output is an identity path.','',
        'Every sampling seed first averages 200 case errors at each budget, integrates the 37.5/50/75% curve, and divides by its own model/grid/seed Uniform area. The table reports the mean and sample SD of these three ratios. These are three sampling seeds for training seed 0, not three trained models.','',
        'Regional statistics use physical_bands_0p02_direct_queries_v1_20261119: boundary distance is evaluated directly at each native coordinate, and interface distance is measured to the same common211 coefficient-edge nodes, with both widths fixed at 0.02. This offline classification correction removes 301 out-of-band boundary assignments at 141² from the original nearest-boolean mapping. Original regional records remain as legacy evidence; training, retained points, scores, target density, full-field errors and rAUEC are unchanged. Raw refinement scores, normalized display scores, pre-FPS planned allocation and actual retained indices are distinct stored quantities. PI target density is a nominal proposal, not an exact inclusion probability.','',
        'Accepted resolutions: '+(', '.join(str(n)+'²' for n in accepted) or 'none yet')+'.','',
        'Tables: [rebuttal main](results/tables/rebuttal_main.md), [controlled common](results/tables/controlled_common.md), [controlled native](results/tables/controlled_native.md).','',
        'Machine-readable statistics: results/budget_metrics.csv, absolute_errors.csv, rauec_by_seed.csv, rauec_summary.csv, regional_by_seed.csv and regional_summary.csv. Raw long-form records and retained prediction/index arrays are in records/r85, records/r141 and records/r211.','',
        'The experiment makes no formal latency claim. Training and head costs are kept in their manifests and curves.','']
    conclusion='最终趋势结论尚待三个分辨率的完整训练与评价验收，当前不以中途模型推断重新训练的收益。'
    if completed:
        idx={(r['resolution'],r['evaluation_grid'],r['method']):r for r in summary}
        parts=[]
        for m in ['pi','cpg','ca']:
            vals=[idx[n,'common',m]['rauec_mean'] for n in RESOLUTIONS]
            parts.append(m.upper()+'为'+' → '.join(f'{v:.4f}' for v in vals))
        first='在相同211²输出网格上，85²→141²→211²独立训练后的rAUEC，'+'；'.join(parts)+'（sampling seeds=3，training seed=0）。'
        region=pd.DataFrame(region_summary);claims=[]
        for method in ['pi','cpg','ca']:
            b=region[(region.method==method)&(region.region=='boundary')]
            s=region[(region.method==method)&(region.region=='smooth')]
            claims.append(method.upper()+f'的边界保留率高于全局预算有{int((b.mean_case_retention_enrichment>1).sum())}/9个条件，平滑区低于全局预算有{int((s.mean_case_retention_enrichment<1).sum())}/9个条件')
        conclusion=first+'\n\n'+'；'.join(claims)+'（条件为3个分辨率×3个预算，先按case和sampling seed聚合）。'
    lines+=['## Result interpretation','',conclusion,'',
        'Recompute: `python3 code/aggregate.py` then `python3 code/verify_statistics.py` and `python3 code/plot_results.py`. During execution, `aggregate.py --allow-partial` emits N/A for pending matched columns.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'results/conclusion_zh.md').write_text(conclusion+'\n')
    (OUT/'CLAIM_CARD.md').write_text('# Claim card\n\n'+
        ('All three matched accuracy matrices are accepted.\n\n'+conclusion if completed else 'Matched resolution trends, accuracy improvement and preservation of regional sampling behavior are pending complete evidence. No numerical claim is assigned to unfinished models.')+
        '\n\nHistorical 85² is a reference, not a matched training cohort. The random-initialization interface checks are engineering validations only. Sampling SD does not estimate training variability.\n')


def main(allow_partial=False):
    frame,regions,accepted=load_accepted(allow_partial)
    budget,absolute,areas,summary=metric_tables(frame)
    regional_seeds,regional_summary=region_tables(regions)
    for name,rows in [('budget_metrics.csv',budget),('absolute_errors.csv',absolute),('rauec_by_seed.csv',areas),
        ('rauec_summary.csv',summary),('regional_by_seed.csv',regional_seeds),('regional_summary.csv',regional_summary)]:
        save_csv(name,rows)
    historical=historical_reference();table_files(summary,historical);checkpoint_registry(accepted)
    generate_report(accepted,summary,regional_summary)
    receipt=dict(passed=True,complete=accepted==list(RESOLUTIONS),accepted_resolutions=accepted,
        expected_metric_rows=44400,accepted_metric_rows=len(frame),sampling_seeds=[0,1,2],training_seeds=[0],
        aggregation_source_sha256=digest(__file__),protocol_sha256=digest(OUT/'protocol.json'),
        region_protocol_id=read(OUT/'config/region_protocol_physical_20261119.json')['region_protocol_id'],
        regional_acceptance_sha256={str(n):digest(OUT/f'results/physical_region_r{n}_acceptance.json') for n in accepted},
        result_files={str(path.relative_to(OUT)):digest(path) for path in sorted((OUT/'results').glob('*.csv'))})
    write(OUT/'results/aggregation_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true')
    main(parser.parse_args().allow_partial)
