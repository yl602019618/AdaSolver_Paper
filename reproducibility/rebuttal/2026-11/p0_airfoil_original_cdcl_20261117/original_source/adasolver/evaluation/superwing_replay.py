"""Reuse previously verified executions with the same protocol and policy."""
import json
from pathlib import Path

from adasolver.inference.superwing_contract import BUDGETS,identity,NFULL,NCOARSE
from adasolver.utils.budget import budget_count


def execution_identity(policy,budget_index=None):
    # These fields select an execution; they never enter sampling or inference.
    if budget_index is not None:
        from adasolver.inference.superwing_density import for_budget
        policy=for_budget(policy,budget_index)
    metadata={'frozen','selection_split','selection_summary_sha256','family','budget_selection_sources'}
    return identity({k:v for k,v in policy.items() if k not in metadata})


def seed_replay(output,runtime,ids,seeds,policies,split,stage,path,done):
    if path is None:return 0
    by_id={identity(p):p for p in policies}
    ids=set(ids);seeds=set(seeds);count=0
    with Path(path).open() as source,(Path(output)/'raw.jsonl').open('a',buffering=1) as handle:
        for line in source:
            r=json.loads(line)
            key=r['config_id'],r['sample_id'],r['seed'],r['budget_fraction']
            if key in done:continue
            if r['config_id'] not in by_id or r['sample_id'] not in ids or r['seed'] not in seeds or r['budget_fraction'] not in BUDGETS:
                raise ValueError('replay record outside requested execution grid')
            if r['protocol_signature']!=runtime.signature or r['split']!=split or r['stage']!=stage:
                raise ValueError('replay protocol/split/stage mismatch')
            if r['operator_sha256']!=runtime.metadata['checkpoint_sha256'] or r['status']!='ok':
                raise ValueError('invalid replay operator or status')
            policy=by_id[r['config_id']]
            if r['method']!=policy['method'] or r['metric_target']!=policy['target'] or r.get('head_sha256')!=policy.get('head_sha256'):
                raise ValueError('replay method, objective or head mismatch')
            if r['selected_count']!=budget_count(NFULL,r['budget_fraction']) or \
                    (r['method'] in ('cpg','ca') and r['coarse_count']!=NCOARSE):
                raise ValueError('replay budget mismatch')
            if not (Path(output)/r['indices_sidecar']).is_file():raise ValueError('replay sidecar is missing')
            if r.get('replay_execution_identity')!=execution_identity(by_id[r['config_id']],BUDGETS.index(r['budget_fraction'])):
                raise ValueError('replay execution policy mismatch')
            handle.write(json.dumps(r,allow_nan=False)+'\n');done.add(key);count+=1
    return count
