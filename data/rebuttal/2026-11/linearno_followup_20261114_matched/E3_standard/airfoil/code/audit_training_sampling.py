"""Replay the isolated sampling streams and verify against saved runtime RNG states."""
from common import *
import io,time


def ensure_final():
    """Run once after both real point-model training processes completed."""
    import fcntl
    if not all(read(OUT/f'checkpoints/{a}/manifest.json').get('execution_status')=='completed' for a in ['t3','linearno']):return
    with (OUT/'runtime/final_sampling_audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=OUT/'results/training_sampling_identity.json';old=read(path) if path.exists() else {}
        if old.get('saved_runtime_rng_states') and len(old.get('checks',[]))==2 and all(r['checkpoint_steps']==113000 for r in old['checks']):return
        main()


def main():
    snapshots={}
    for arch in ['t3','linearno']:
        path=OUT/f'checkpoints/{arch}/resume.pt'
        payload=path.read_bytes();saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        snapshots[arch]=dict(epoch=saved['epoch'],steps=saved['steps'],point_rng=saved['point_rng'],
                             order_rng=saved['order_rng'],checkpoint_path=str(path),checkpoint_snapshot_sha256=hashlib.sha256(payload).hexdigest())
    train_ids=np.asarray(read(SPLITS)['training_ids']);point=np.random.default_rng(2001);order=np.random.default_rng(1001)
    captures=[];states={};step=0;maximum=max(s['epoch'] for s in snapshots.values());mandatory=np.arange(50,171)*51
    wanted_steps={0,1,2,112,113,114,225,226}
    for s in snapshots.values():wanted_steps.update([s['steps']-3,s['steps']-2,s['steps']-1])
    for epoch in range(maximum+1):
        geometry_order=order.permutation(train_ids)
        for lower in range(0,900,8):
            count=int(point.integers(4508,11272));indices=point.permutation(11271)[:count]
            if step in wanted_steps:
                captures.append(dict(global_step_zero_based=step,epoch=epoch,batch=lower//8,point_count=count,
                    geometry_ids=geometry_order[lower:lower+8].tolist(),selected_indices_sha256=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
                    retained_surface_points=int(np.isin(indices,mandatory).sum())))
            step+=1
        if epoch in {s['epoch'] for s in snapshots.values()}:states[epoch]=dict(point_rng=point.bit_generator.state.copy(),order_rng=order.bit_generator.state.copy())
    checks=[]
    for arch,s in snapshots.items():
        assert s['steps']==113*(s['epoch']+1)
        expected=states[s['epoch']]
        assert expected['point_rng']==s['point_rng'] and expected['order_rng']==s['order_rng']
        checks.append(dict(architecture=arch,checkpoint_epoch=s['epoch'],checkpoint_steps=s['steps'],
                           checkpoint_snapshot_sha256=s['checkpoint_snapshot_sha256'],point_rng_exact=True,geometry_order_rng_exact=True))
    result=dict(passed=True,scope='replay independently seeded RNGs; verify their states against actual saved runtime checkpoints',
                live_per_batch_capture=False,backbone_training_mandatory_surface=False,head_and_inference_mandatory_surface=121,
                point_selection='uniform_without_replacement over all11271 points',index_dtype='int64',
                both_primary_models_use_identical_sampling_code=True,code_source=str(OUT/'config/execution_sources/train.py'),
                checks=checks,saved_runtime_rng_states=snapshots,replayed_batch_samples=captures)
    write(OUT/'results/training_sampling_identity.json',result)
    write(OUT/'config/factual_sampling_note.json',dict(backbone_training_mandatory_surface=False,head_and_inference_mandatory_surface=121,
          action='documentation and RNG replay verification only; running training code unchanged',proof='results/training_sampling_identity.json'))
    print(json.dumps(dict(passed=True,checks=checks,replayed_batches=len(captures)),indent=2))

if __name__=='__main__':main()
