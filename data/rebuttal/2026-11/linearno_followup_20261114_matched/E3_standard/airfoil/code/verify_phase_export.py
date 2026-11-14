"""Phase immutability/schema fixture; temporary synthetic rows are never research evidence."""
from common import *
import evaluate,tempfile,csv


def main():
    true_out=evaluate.OUT
    with tempfile.TemporaryDirectory(dir=OUT/'results',prefix='phase_export_development_') as directory:
        root=Path(directory);evaluate.OUT=root;(root/'records/t3').mkdir(parents=True)
        metrics={k:1. for k in ['field_relative_l2','cd_abs_error','cl_abs_error','cdcl_abs_error']}
        common=dict(case_id=4,checkpoint_sha256='development_fixture',normalizer_sha256='development_fixture',old=metrics,new=metrics)
        full=dict(common,method='full',sampling_seed=None,budget_fraction=1.,retained_points=11271)
        pair=[dict(common,method=m,sampling_seed=0,budget_fraction=.375,retained_points=4227,beta=beta,
            indices_hash='fixture_selected',coarse_indices_hash='fixture_coarse',coarse_prediction_hash='fixture_prediction')
            for m,beta in [('cpg_beta0',0.),('cpg',1.)]]
        baseline=root/'baseline.jsonl';baseline.write_text(''.join(json.dumps(row)+'\n' for row in [full,*pair]))
        gate={'old_id':'old_fixture','new_id':'new_fixture'}
        preliminary=evaluate.export_metrics('t3',gate,[baseline],'preliminary');payload=Path(preliminary['metrics_csv']).read_bytes()
        ca=root/'ca.jsonl';ca.write_text(json.dumps(dict(common,method='ca',sampling_seed=0,budget_fraction=.375,
            retained_points=4227,beta=1.,indices_hash='fixture_selected',head_sha256='fixture_head'))+'\n')
        complete=evaluate.export_metrics('t3',gate,[baseline,ca],'complete')
        assert Path(preliminary['metrics_csv']).read_bytes()==payload
        assert Path(complete['metrics_csv']).read_bytes()==Path(complete['compatibility_metrics_csv']).read_bytes()
        rows=list(csv.DictReader(Path(preliminary['metrics_csv']).open()));assert len(rows)==24
        cpg=[r for r in rows if r['method']=='cpg'];assert all(r['coarse_prediction_hash']=='fixture_prediction' and r['retained_points']=='4227' for r in cpg)
        evaluate.export_metrics('t3',gate,[baseline],'preliminary')
        try:evaluate.export_metrics('t3',gate,[baseline,ca],'preliminary')
        except FileExistsError:pass
        else:raise AssertionError('immutable preliminary unexpectedly changed')
    evaluate.OUT=true_out
    write(OUT/'results/phase_export_development_validation.json',dict(passed=True,
        preliminary_unchanged_after_complete=True,coarse_prediction_hash_and_retained_count_exported=True,
        conflicting_phase_rewrite_rejected=True,scope='temporary schema fixture; no measured model results; temporary rows deleted',
        export_source_sha256=sha(OUT/'code/evaluate.py')))


if __name__=='__main__':main()
