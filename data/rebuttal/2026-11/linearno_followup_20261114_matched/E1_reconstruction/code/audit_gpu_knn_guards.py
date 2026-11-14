"""Rerun the independent CPU oracle suite with early-rejection guard probes.

The original audit script/results are immutable. This runner reuses its exact
12 valid-domain cases and replaces only the formerly permissive failure probes
with assertions that invalid requests are rejected before torch.cdist.
"""
from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'results/gpu_knn_candidate_cpu_audit.py'
RUN=ROOT/'records/gpu_knn_candidate_input_guards/postfix_audit'
RUN.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('original_knn_cpu_audit',ORIGINAL)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
candidate=audit.candidate


def early_rejection_probes():
    rng=np.random.default_rng(782);base_source=torch.from_numpy(rng.normal(size=(11,3)));base_query=torch.from_numpy(rng.normal(size=(5,3)))
    probes=[]
    def reject(name,*,source=base_source,query=base_query,exception=ValueError,**kwargs):
        calls=[];original=candidate.torch.cdist
        def observed(*args,**kw):
            calls.append(True)
            raise AssertionError('Invalid input reached torch.cdist')
        candidate.torch.cdist=observed
        try:
            try:candidate.build_exact_gpu_knn(source,query,**kwargs)
            except exception as exc:
                result=dict(id=name,guard_passed=True,rejected_by_builder=True,exception=type(exc).__name__,message=str(exc))
            else:raise AssertionError(f'{name} returned a map')
        finally:candidate.torch.cdist=original
        assert not calls,name
        probes.append(dict(**result,cdist_calls=len(calls)))
    for name,value,error in [('PAIR_BOUND',10,ValueError),('ZERO_PAIR_BUDGET',0,ValueError),('NEGATIVE_PAIR_BUDGET',-1,ValueError),
                             ('FRACTIONAL_PAIR_BUDGET',1.5,TypeError),('BOOLEAN_PAIR_BUDGET',True,TypeError)]:
        reject(name,max_pairs=value,exception=error)
    for name,value,error in [('ZERO_K',0,ValueError),('NEGATIVE_K',-1,ValueError),('FRACTIONAL_K',1.5,TypeError),
                             ('BOOLEAN_K',True,TypeError),('NUMPY_FLOAT_K',np.float64(4),TypeError)]:
        reject(name,k=value,exception=error)
    for label,value in [('NAN',np.nan),('POSITIVE_INF',np.inf),('NEGATIVE_INF',-np.inf)]:
        source=base_source.clone();source[0,0]=value;reject(label+'_SOURCE',source=source)
        query=base_query.clone();query[0,0]=value;reject(label+'_QUERY',query=query)
    source=np.array([[0.,0.],[0.,0.],[1.,0.],[0.,1.],[2.,0.]])
    query=np.array([[0.,0.]])
    x=torch.from_numpy(source);q=torch.from_numpy(query)
    reject('REORDERED_SOURCE_CPU',source=x,query=q,source_cpu=source[::-1].copy(),query_cpu=query)
    reject('STALE_QUERY_CPU',source=x,query=q,source_cpu=source,query_cpu=np.array([[2.,0.]]))
    reject('SOURCE_COPY_SHAPE',source=x,query=q,source_cpu=source[:-1])
    reject('QUERY_COPY_SHAPE',source=x,query=q,query_cpu=np.zeros((1,3),dtype=np.float64))
    reject('SOURCE_COPY_DTYPE',source=x,query=q,source_cpu=source.astype(np.float32),exception=TypeError)
    reject('QUERY_COPY_DTYPE',source=x,query=q,query_cpu=query.astype(np.float32),exception=TypeError)
    bad_source=source.copy();bad_source[0,0]=np.nan
    reject('NONFINITE_SOURCE_COPY',source=x,query=q,source_cpu=bad_source)
    bad_query=query.copy();bad_query[0,0]=np.inf
    reject('NONFINITE_QUERY_COPY',source=x,query=q,query_cpu=bad_query)
    # The consistency contract is unconditional, not only checked once a tie
    # happens to occur. Tiny stale changes must also fail without tolerance.
    stale=base_source.numpy().copy();stale[2,1]=np.nextafter(stale[2,1],np.inf)
    reject('ONE_ULP_STALE_SOURCE_NO_TIE',source_cpu=stale)
    stale=base_query.numpy().copy();stale[2,1]=np.nextafter(stale[2,1],-np.inf)
    reject('ONE_ULP_STALE_QUERY_NO_TIE',query_cpu=stale)
    reject('EMPTY_SOURCE',source=torch.empty((0,3),dtype=torch.float64))
    reject('EMPTY_QUERY',query=torch.empty((0,3),dtype=torch.float64))
    reject('ZERO_DIMENSION',source=torch.empty((2,0),dtype=torch.float64),query=torch.empty((1,0),dtype=torch.float64))
    return probes


def valid_guard_boundaries():
    rng=np.random.default_rng(824);source=rng.normal(size=(11,3));query=rng.normal(size=(5,3));values=rng.normal(size=(11,2))
    x=torch.tensor(source);q=torch.tensor(query);expected,_,_=audit.oracle(source,values,query,4)
    mapping=candidate.build_exact_gpu_knn(x,q,k=np.int64(4),max_pairs=np.int64(len(source)),
        source_cpu=torch.tensor(source),query_cpu=torch.tensor(query))
    np.testing.assert_allclose(mapping.apply(torch.from_numpy(values)).numpy(),expected,rtol=5e-13,atol=2e-12)
    assert mapping.metadata['max_distance_pairs']==len(source) and mapping.metadata['query_block']==1
    assert mapping.metadata['max_distance_pairs']<=mapping.metadata['max_pairs_budget']
    # A valid supplied side copy is snapshotted. Subsequent caller mutation
    # during search cannot change native tie resolution for unchanged tensors.
    s=np.array([[0.,0.],[0.,0.],[1.,0.],[0.,1.],[2.,0.]])
    t=np.array([[0.,0.]]);v=np.array([[1.],[9.],[20.],[30.],[40.]])
    sx=torch.tensor(s);tq=torch.tensor(t);copy_s=s.copy();copy_q=t.copy();original=candidate.torch.cdist
    def changed_side_copy(*args,**kw):
        copy_s[:]=copy_s[::-1].copy();copy_q[:]=2.
        return original(*args,**kw)
    candidate.torch.cdist=changed_side_copy
    try:m=candidate.build_exact_gpu_knn(sx,tq,source_cpu=copy_s,query_cpu=copy_q)
    finally:candidate.torch.cdist=original
    reference,_,_=audit.oracle(s,v,t,4);actual=m.apply(torch.from_numpy(v)).numpy()
    np.testing.assert_array_equal(actual,reference)
    return [dict(id='EXACT_PAIR_BOUND_NUMPY_INTEGERS_TORCH_CPU_COPIES',passed=True,max_pairs=len(source),observed_pairs=mapping.metadata['max_distance_pairs']),
            dict(id='VALIDATED_SIDE_COPY_SNAPSHOT',passed=True,expected=reference.tolist(),actual=actual.tolist())]


def main():
    assert not torch.cuda.is_initialized(),'CPU audit must not use CUDA'
    protected=[ROOT/'code/reconstruct_api.py',ROOT/'code/car_workflow.py',ROOT/'results/gpu_knn_candidate_cpu_audit.py',
               ROOT/'results/gpu_knn_candidate_cpu_audit.json',ROOT/'results/gpu_knn_candidate_cpu_audit.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    old=json.loads((ROOT/'records/gpu_knn_candidate_input_guards/provenance.json').read_text())
    audit.OUT=RUN;audit.probe_failure_inputs=early_rejection_probes
    audit.main()  # Same original12 valid-domain cases and native oracle.
    result=json.loads((RUN/'gpu_knn_candidate_cpu_audit.json').read_text())
    assert result['valid_case_count']==12 and all(r['guard_passed'] and r['cdist_calls']==0 for r in result['issue_probes'])
    result['audit_status']='valid_domain_cpu_checks_and_input_guards_passed'
    result['guard_probes']=result.pop('issue_probes');result['guard_probe_count']=len(result['guard_probes'])
    result['additional_valid_boundaries']=valid_guard_boundaries()
    result['candidate_version']=candidate.CANDIDATE_VERSION
    result['previous_source_sha256']=old['old_candidate_sha256']
    result['original_audit_script_sha256']=hashes['results/gpu_knn_candidate_cpu_audit.py']
    result['runner_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['audit_author_scope']='Implementation follow-up rerun reusing the unchanged independent valid-case suite; no second independent reviewer is claimed'
    result['approval']='CPU input-guard and finite valid-case evidence only; candidate remains outside frozen API v1.1 pending CUDA/end-to-end validation and formal timing'
    assert not torch.cuda.is_initialized()
    for p in protected:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))]
    result['protected_sources_unchanged']=hashes
    (RUN/'gpu_knn_candidate_cpu_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'results/gpu_knn_candidate_guard_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# GPU-kNN candidate input-guard follow-up', '',
        f"Candidate version: `{candidate.CANDIDATE_VERSION}`; SHA256: `{result['source_sha256']}`.",
        f"The unchanged independent valid-domain suite was rerun:12 cases,244 queries,515 output elements; maximum field difference {result['maximum_absolute_field_difference']:.17g}, with0 violations of the declared elementwise tolerance. All neighbor indices still match the native oracle.",
        f"All{result['guard_probe_count']} invalid-input probes are rejected by the builder before any torch.cdist call. Positive NumPy integer arguments, max_pairs exactly equal to the source count, matching Torch CPU side copies, and validated side-copy snapshot behavior also passed.", '',
        '| Fixed defect | New behavior |', '|---|---|',
        '| Pair budget below source count or nonpositive | Reject before search; one query must fit the exhaustive source axis |',
        '| Inconsistent optional CPU copy | Check float64 dtype, exact shape, values and point order against the tensor, then retain a snapshot; mismatches fail even on non-tie queries |',
        '| Invalid k | Require a positive integer; reject zero, negatives, floats and booleans before search |',
        '| Nonfinite source/query | Reject NaN and ±infinity before search, matching the finite native geometry contract |', '',
        'The positive pair-budget guard bounds one distance tile only; retained indices/distances, top-k workspace and other live allocations still add memory. With supplied CPU copies, exact tensor-copy identity checks may transfer coordinates from GPU and synchronize; this work belongs to any future candidate timing and is not hidden as free preparation.', '',
        'The original source and audit are preserved under code/versions/gpu_knn_candidate_before_input_guards.py and records/gpu_knn_candidate_input_guards/original_audit/. The original results/gpu_knn_candidate_cpu_audit.{py,json,md} are unchanged. The new runner is code/audit_gpu_knn_guards.py.', '',
        'This rerun was executed by the implementation follow-up thread using the original independent valid-case suite. It is CPU correctness evidence only. No CUDA validation, peak GPU allocation measurement, end-to-end acceptance or speedup claim is made. The candidate remains separate from frozen API v1.1, and the existing Car controller is unchanged.', '',
        'Reproduce from07_current:', '', '```sh',
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c128-135 python3 linearno_followup_20261114/reconstruction/code/audit_gpu_knn_guards.py", '```']
    (ROOT/'results/gpu_knn_candidate_guard_audit.md').write_text('\n'.join(text)+'\n')
    print('Post-fix audit passed:',result['valid_case_count'],'valid cases,',result['guard_probe_count'],'early rejections;',result['source_sha256'])


if __name__=='__main__':main()
