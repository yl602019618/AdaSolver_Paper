"""Matched full1000-epoch training with isolated sampling RNGs and resume state."""
from common import *
import argparse,time,math

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=['t3','linearno','linearno_conv'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False
    protocol=read(OUT/'config/protocol_lock.json');data=Data();device='cuda:0';arch=args.architecture
    clip=.1 if arch=='t3' else 1.
    target=OUT/'checkpoints'/arch;target.mkdir(exist_ok=True)
    resume_path=target/'resume.pt';manifest_path=target/'manifest.json'
    if (target/'last.pt').exists() and read(manifest_path).get('execution_status')=='completed':raise FileExistsError('Completed model exists')
    if resume_path.exists() and not args.resume:raise FileExistsError('Use explicit --resume for existing run')
    model=build_model(arch).to(device)
    coords=torch.from_numpy(np.array(data.pos)).to(device);pressure=torch.from_numpy(np.array(data.pressure)).to(device)
    train_ids=np.array(data.splits['training_ids']);val_ids=data.splits['validation_ids']
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=protocol['learning_rate'],epochs=1000,steps_per_epoch=113,pct_start=.3)
    order_rng=np.random.default_rng(1001);point_rng=np.random.default_rng(2001)
    begin_epoch=0;steps=0;best=float('inf');best_epoch=None;accumulated_seconds=0.
    if args.resume:
        saved=torch.load(resume_path,map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        begin_epoch=saved['epoch']+1;steps=saved['steps'];best=saved['best'];best_epoch=saved['best_epoch']
        order_rng.bit_generator.state=saved['order_rng'];point_rng.bit_generator.state=saved['point_rng'];accumulated_seconds=saved['training_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    manifest=dict(benchmark='airfoil2d',architecture='Transolver3' if arch=='t3' else 'LinearNO',
                  architecture_variant=arch,model_class=type(model).__name__,native_model_class=type(model.net).__name__,
                  native_attention_class=type(model.net.blocks[0].Attn).__name__,parameter_count=sum(p.numel() for p in model.parameters()),
                  execution_status='running',evidence_status='pending',epochs_requested=1000,epochs_completed=begin_epoch,
                  training_samples=900,validation_samples=100,training_seed=0,target_name='pressure',target_channel=3,
                  shape=[221,51],nfull=11271,nmin_count=11271 if arch=='linearno_conv' else 4508,
                  nmin_fraction=1. if arch=='linearno_conv' else .4,batch_size=8,point_schedule=protocol['point_schedule'],
                  normalizer=str(data.normalizer_path),normalizer_sha256=sha(data.normalizer_path),split_manifest_sha256=sha(SPLITS),
                  checkpoint=str(target/'best.pt'),last_checkpoint=str(target/'last.pt'),resume_checkpoint=str(resume_path),
                  protocol=str(OUT/'config/protocol_lock.json'),source_manifest_sha256=sha(OUT/'config/source_manifest.json'),
                  common_code_sha256=sha(OUT/'code/common.py'),max_grad_norm=clip,
                  data_path=str(DATA),data_split=dict(train=data.splits['training_ids'],validation=data.splits['validation_ids'],test=data.splits['airfoil']['primary_test_ids']),
                  train_code_sha256=sha(__file__),torch=torch.__version__,cuda=torch.version.cuda,pid=os.getpid(),
                  device=torch.cuda.get_device_name(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                  validation_criterion=protocol['model_selection'],test_used_for_selection=False,training_seconds=accumulated_seconds)
    # Conv is a separately labelled full-grid efficiency baseline, never variable-point evidence.
    if arch=='linearno_conv':manifest['point_schedule']=dict(minimum=11271,maximum=11271,distribution='full_grid')
    write(manifest_path,manifest);start=time.perf_counter()
    history=target/'history.jsonl'
    with history.open('a' if args.resume else 'x') as log:
        for epoch in range(begin_epoch,1000):
            epoch_start=time.perf_counter();model.train();order=order_rng.permutation(train_ids);total=0.;counts=[]
            for lower in range(0,len(order),8):
                batch=torch.as_tensor(order[lower:lower+8],device=device)
                K=int(point_rng.integers(4508,11272)) if arch!='linearno_conv' else 11271
                selected=torch.as_tensor(point_rng.permutation(11271)[:K],device=device) if arch!='linearno_conv' else torch.arange(11271,device=device)
                pos=coords[batch][:,selected];truth=pressure[batch][:,selected]
                pred=data.decode(model(pos,pos))
                loss=relative_l2(pred,truth).sum()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                optimizer.step();scheduler.step();total+=float(loss.detach());counts.append(K);steps+=1
            model.eval();values=[]
            with torch.no_grad():
                for lower in range(0,100,8):
                    batch=torch.tensor(val_ids[lower:lower+8],device=device)
                    pred=data.decode(model(coords[batch],coords[batch]));values.extend(relative_l2(pred,pressure[batch]).cpu().tolist())
            validation=float(np.mean(values));assert math.isfinite(validation)
            if validation<best:
                best=validation;best_epoch=epoch;temp=target/'best.tmp.pt';torch.save(model.state_dict(),temp);temp.replace(target/'best.pt')
            elapsed=accumulated_seconds+time.perf_counter()-start
            row=dict(epoch=epoch,train_physical_relative_l2=total/900,validation_physical_relative_l2=validation,
                     best_validation=best,best_epoch=best_epoch,k_mean=float(np.mean(counts)),k_min=min(counts),k_max=max(counts),
                     global_step=steps,epoch_seconds=time.perf_counter()-epoch_start,training_seconds=elapsed)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            manifest.update(epochs_completed=epoch+1,total_updates=steps,best_epoch=best_epoch,
                            best_validation_relative_l2=best,training_seconds=elapsed)
            write(manifest_path,manifest)
            if (epoch+1)%10==0 or epoch==999:
                temp=target/'resume.tmp.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,best=best,best_epoch=best_epoch,
                    order_rng=order_rng.bit_generator.state,point_rng=point_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),training_seconds=elapsed),temp);temp.replace(resume_path)
    torch.save(model.state_dict(),target/'last.pt')
    manifest.update(execution_status='completed',epochs_completed=1000,total_updates=steps,
                    checkpoint_sha256=sha(target/'best.pt'),last_checkpoint_sha256=sha(target/'last.pt'))
    write(manifest_path,manifest)

if __name__=='__main__':main()
