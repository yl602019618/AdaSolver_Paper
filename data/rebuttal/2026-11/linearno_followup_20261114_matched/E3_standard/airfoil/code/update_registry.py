"""Separate model identity, training completion and experiment acceptance."""
from common import *
import csv


def main():
    rows=[]
    for arch in ['t3','linearno','linearno_conv']:
        p=OUT/f'checkpoints/{arch}/manifest.json'
        if not p.exists():continue
        m=read(p);completed=m.get('execution_status')=='completed';identity=False;completion=False
        proof_path=OUT/f'results/{arch}_identity_validation.json'
        if completed:
            existing=read(proof_path) if proof_path.exists() else {}
            if existing.get('checkpoint_sha256')==m['checkpoint_sha256'] and existing.get('passed'):
                identity=completion=True
            else:
                assert sha(m['checkpoint'])==m['checkpoint_sha256']
                assert sha(m['normalizer'])==m['normalizer_sha256']
                normalizer=np.load(m['normalizer']);assert normalizer['training_ids'].tolist()==read(SPLITS)['training_ids']
                model=build_model(arch)
                model.load_state_dict(torch.load(m['checkpoint'],map_location='cpu',weights_only=True),strict=True)
                checkpoint=torch.load(m['resume_checkpoint'],map_location='cpu',weights_only=False)
                assert checkpoint['epoch']==999 and checkpoint['steps']==113000
                assert all(int(s['step'].item())==113000 for s in checkpoint['optimizer']['state'].values())
                history=[json.loads(line) for line in (OUT/f'checkpoints/{arch}/history.jsonl').open()]
                assert len(history)==1000 and [r['epoch'] for r in history]==list(range(1000))
                assert [r['global_step'] for r in history]==[113*(epoch+1) for epoch in range(1000)]
                validation=np.array([r['validation_physical_relative_l2'] for r in history]);assert np.isfinite(validation).all()
                assert int(np.argmin(validation))==m['best_epoch'] and float(validation.min())==m['best_validation_relative_l2']
                if arch=='linearno_conv':assert all(r['k_min']==r['k_max']==11271 for r in history)
                else:assert all(4508<=r['k_min']<=r['k_mean']<=r['k_max']<=11271 for r in history)
                assert sha(OUT/'code/common.py')==m['common_code_sha256']
                assert sha(OUT/'config/execution_sources/train.py')==m['train_code_sha256']
                assert sum(t.numel() for t in model.parameters())==m['parameter_count']
                proof=dict(passed=True,checkpoint_sha256=m['checkpoint_sha256'],normalizer_sha256=m['normalizer_sha256'],
                           model_class=type(model.net).__name__,attention_class=type(model.net.blocks[0].Attn).__name__,
                           parameters=m['parameter_count'],epochs_completed=1000,updates_completed=113000,
                           optimizer_parameter_steps=113000,history_epochs_verified=1000,checkpoint_selected_epoch=m['best_epoch'],
                           history_sha256=sha(OUT/f'checkpoints/{arch}/history.jsonl'),verifier_sha256=sha(__file__),
                           scope='strict native state/hash, train-only normalization IDs, actual final resume/optimizer steps, all epoch history and validation-min checkpoint selection; accuracy/latency separate')
                write(proof_path,proof);identity=completion=True
        acceptance=OUT/f'results/{arch}_accuracy_acceptance.json'
        accuracy=read(acceptance).get('passed') is True if acceptance.exists() else 'pending'
        latency_path=OUT/'results/latency_validation.json'
        latency=read(latency_path).get('passed') is True if latency_path.exists() else 'pending'
        head_path=OUT/f'checkpoints/{arch}/head/manifest.json';head=read(head_path) if head_path.exists() else {}
        head_proof_path=OUT/f'results/{arch}_head_acceptance.json'
        head_proof=read(head_proof_path) if head_proof_path.exists() else {}
        rows.append(dict(dataset='airfoil2d',backbone=arch,native_model_class=m.get('native_model_class'),
            native_attention_class=m.get('native_attention_class'),parameters=m.get('parameter_count'),
            training_seed=0,epochs_completed=m.get('epochs_completed',0),updates_completed=m.get('total_updates',0),
            checkpoint_path=m['checkpoint'],checkpoint_sha256=m.get('checkpoint_sha256','pending'),
            normalizer_path=m['normalizer'],normalizer_sha256=m['normalizer_sha256'],normalizer_fit_cases=900,
            identity_verified=identity,training_completed=completion,accuracy_verified=accuracy,latency_verified=latency,
            head_applicability='none_fullgrid_only' if arch=='linearno_conv' else 'own_backbone_pressure_head',
            head_completed=head.get('execution_status')=='completed',head_sha256=head.get('head_sha256',''),
            head_identity_verified='not_applicable' if arch=='linearno_conv' else bool(head_proof.get('passed') and head_proof.get('head_sha256')==head.get('head_sha256')) if head_proof else 'pending',
            identity_verification_scope='strict state/hash/normalizer and final training step; accuracy/latency have separate flags'))
    path=OUT/'results/checkpoint_registry.csv';temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


if __name__=='__main__':main()
