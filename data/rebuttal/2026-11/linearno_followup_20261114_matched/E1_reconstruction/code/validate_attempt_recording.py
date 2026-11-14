"""Fault-injection check of worker evidence persistence; no experiment data."""
from __future__ import annotations
import hashlib,json,tempfile
from pathlib import Path
from run_attempt import Attempt,atomic_json,preserve_existing,track_status
from timing_helper import measure_paired

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=Path(tempfile.mkdtemp(prefix='attempt_selftest_',dir=ROOT/'runtime'))
    status=root/'status.json';telemetry=root/'gpu_telemetry.json'
    atomic_json(status,{'execution_status':'running','completed_pairs':7})
    original=b'{"fixture":"prior synthetic telemetry; not an experiment"}\n';telemetry.write_bytes(original)
    first=None
    try:
        with Attempt('synthetic_worker',root=root) as attempt:
            first=attempt;track_status(status);preserve_existing(telemetry)
            telemetry.write_text('{"fixture":"new synthetic telemetry"}\n')
            count=0
            def guard():
                nonlocal count
                count+=1
                if count==24:raise RuntimeError('Injected CPU selftest failure after two formal repeat cycles')
            measure_paired({'old':lambda:None,'new':lambda:None},scope='synthetic CPU selftest only',
                warmups=20,repeats=100,synchronize=lambda:None,guard=guard)
    except RuntimeError as exc:assert 'Injected CPU selftest' in str(exc)
    else:raise AssertionError('Missing injected failure')
    receipt=json.loads((first.directory/'attempt.json').read_text());assert receipt['execution_status']=='failed'
    assert json.loads(status.read_text())['execution_status']=='failed'
    assert json.loads(status.read_text())['completed_pairs']==7
    prior=next(r for r in receipt['preserved_artifacts'] if r['source']==str(telemetry))
    assert (first.directory/prior['archive']).read_bytes()==original
    partial=next(r for r in receipt['partial_evidence'] if r['frame']=='timing_helper.py:measure_paired')
    records=json.loads((first.directory/partial['file']).read_text())['raw']
    assert len(records)==44 and sum(r['kind']=='formal' for r in records)==4
    first_hash=hashlib.sha256((first.directory/'attempt.json').read_bytes()).hexdigest()
    with Attempt('synthetic_worker',root=root) as second:
        preserve_existing(telemetry)
    assert json.loads((second.directory/'attempt.json').read_text())['execution_status']=='completed'
    assert hashlib.sha256((first.directory/'attempt.json').read_bytes()).hexdigest()==first_hash
    # A still-running child cannot be duplicated after a controller restart.
    with Attempt('lock_check',root=root):
        try:
            with Attempt('lock_check',root=root):raise AssertionError('Duplicate worker admitted')
        except RuntimeError as exc:assert 'existing lock_check worker' in str(exc)
    refused=[p for p in (root/'runtime/attempts').glob('lock_check_*/attempt.json') if json.loads(p.read_text())['execution_status']=='refused_duplicate_worker']
    assert len(refused)==1
    result=dict(passed=True,evidence_scope='Synthetic CPU fault injection only; no model or experiment results',
        test_directory=str(root),preserved_prior_telemetry=True,partial_warmup_observations=40,partial_formal_observations=4,
        completed_pair_count_preserved=7,failure_traceback_saved=(first.directory/'traceback.txt').exists(),
        distinct_attempt_receipts=True,prior_failed_attempt_unchanged_after_retry=True,duplicate_live_worker_refused=True)
    (ROOT/'results/attempt_recording_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
