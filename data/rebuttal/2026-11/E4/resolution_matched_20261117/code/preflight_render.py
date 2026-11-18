"""Watermarked layout check from random-model validation engineering records."""
from __future__ import annotations
import json,os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='4'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from common import OUT as REAL_OUT,read,write,digest
from evaluate import physical_regions
from adasolver.inference import adaptive
from adasolver.inference.pi.darcy import coefficient_interface_distance
import plot_results as plot


def main():
    fixture=REAL_OUT/'evaluation/render_fixture';fixture.mkdir(parents=True,exist_ok=True)
    (fixture/'results').mkdir(exist_ok=True)
    (fixture/'README.md').write_text('# Engineering render fixture only\n\nThese plots exercise layout and exports using random-model, validation-case-900 records. The displayed zero error bars are fixture placeholders, not measured sampling uncertainty. No plot or value here is a matched experiment result.\n')
    plot.OUT=fixture;plot.FIGURES=fixture/'figures';plot.SOURCES=plot.FIGURES/'source_data';plot.setup()
    original_save=plot.save
    def watermarked(fig,name,caption,sources):
        fig.text(.5,-.035,'ENGINEERING FIXTURE — not experiment results',ha='center',va='top',color='#8B0000',fontsize=12)
        return original_save(fig,name,'Engineering fixture only. '+caption,sources)
    plot.save=watermarked
    points=[];binrows=[];sourcerows=[];absolute=[];rauec=[];regions=[]
    common=np.load(REAL_OUT/'data/r211/reference_coords.npy')
    cc=np.load(REAL_OUT/'data/r211/validation_coeff.npy',mmap_mode='r')[0]
    masks=physical_regions(cc,common);labels=np.zeros(len(common),dtype=np.int8)
    labels[masks['interface']]=1;labels[masks['boundary']]=2
    for n in [85,141,211]:
        records=read(REAL_OUT/f'evaluation/engineering_record_fixture/r{n}_validation_serialization.json')['records']
        for grid in ['common','native']:
            for row in records:
                absolute.append(dict(resolution=n,evaluation_grid=grid,method=row['method'],budget_fraction=row['budget_fraction'],
                    error_mean=row['field_relative_l2_'+grid],sampling_sd=0.,engineering_fixture=True))
            def area(method):
                values=[next(r['field_relative_l2_'+grid] for r in records if r['method']==method and r['budget_fraction']==b) for b in [.375,.5,.75]]
                return .125*(values[0]+values[1])/2+.25*(values[1]+values[2])/2
            for method in plot.METHODS:
                rauec.append(dict(resolution=n,evaluation_grid=grid,method=method,rauec_mean=area(method)/area('uniform'),sampling_sd=0.,engineering_fixture=True))
        for row in records:
            for region,value in row['regions'].items():
                regions.append(dict(resolution=n,method=row['method'],region=region,budget_fraction=row['budget_fraction'],
                    mean_case_retention_enrichment=value['retained_fraction']/row['actual_budget_fraction'],
                    mean_case_retention_enrichment_sampling_sd=0.,engineering_fixture=True))
        reference=np.load(REAL_OUT/f'data/r{n}/reference_coords.npy');mapping=cKDTree(common).query(reference,k=1)[1]
        coefficient=np.load(REAL_OUT/f'data/r{n}/validation_coeff.npy',mmap_mode='r')[0].reshape(-1)
        pid,_=adaptive.build_patches(reference,32,seed=0);size=np.bincount(pid,minlength=32)
        for method in ['pi','cpg','ca']:
            row=next(r for r in records if r['method']==method and r['budget_fraction']==.5)
            path=REAL_OUT/'evaluation/engineering_record_fixture'/row['source_array']
            with np.load(path,allow_pickle=False) as a:
                ids=a['indices'];indicator=np.zeros(n*n,dtype=np.uint8);indicator[ids]=1
                if method=='pi':
                    support=np.arange(n*n);score=1+.25*np.exp(-coefficient_interface_distance(coefficient,(n,n))/(2/84))
                    lifted=score;density=score/score.mean()
                else:
                    support=a['coarse_indices'];score=a['coarse_score'];lifted=score[cKDTree(reference[support]).query(reference,k=1)[1]]
                    density=a['target_patch_counts'][pid]/size[pid]/(len(ids)/(n*n))
                raw=np.full(n*n,np.nan);raw[support]=score
                points.append(pd.DataFrame(dict(case_id=900,resolution=n,method=method,node_index=np.arange(n*n),x=reference[:,0],y=reference[:,1],
                    region=labels[mapping],raw_support_score=raw,relative_display_score=lifted/max(score.mean(),1e-12),target_relative_density=density,retained=indicator,engineering_fixture=True)))
                edges=np.linspace(0,1,17);observed=np.histogram2d(reference[:,1],reference[:,0],bins=(edges,edges))[0]
                selected=np.histogram2d(reference[ids,1],reference[ids,0],bins=(edges,edges))[0]
                for iy in range(16):
                    for ix in range(16):binrows.append(dict(case_id=900,resolution=n,method=method,bin_y=iy,bin_x=ix,
                        actual_retained_fraction=selected[iy,ix]/observed[iy,ix],engineering_fixture=True))
            sourcerows.append(dict(case_id=900,resolution=n,method=method,path=str(path),sha256=digest(path),engineering_fixture=True))
    for name,rows in [('rauec_summary.csv',rauec),('absolute_errors.csv',absolute),('regional_summary.csv',regions)]:
        pd.DataFrame(rows).to_csv(fixture/'results'/name,index=False)
    pointpath=plot.SOURCES/'qualitative_values.csv.gz';binpath=plot.SOURCES/'qualitative_bins.csv';sourcepath=plot.SOURCES/'qualitative_sources.csv'
    pd.concat(points,ignore_index=True).to_csv(pointpath,index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(binrows).to_csv(binpath,index=False);pd.DataFrame(sourcerows).to_csv(sourcepath,index=False)
    plot.export_examples=lambda:(pointpath,binpath,sourcepath)
    output=[plot.rauec_plot(),plot.error_plot(),plot.region_plot()]+plot.example_plots((900,))
    write(REAL_OUT/'evaluation/render_preflight.json',dict(passed=True,engineering_fixture_only=True,trained_weights=False,
        validation_case_ids=[900],test_opened=False,figure_groups=len(output),exported_files=len(output)*3,
        fixture_directory=str(fixture),figures=output,visual_review='pending PNG inspection'))


if __name__=='__main__':main()
