"""Train each complete two-block learned acquisition head on its own bank."""
import argparse,time,json,os,shutil
import numpy as np
import torch
from common import OUT,read,write,digest,seeds
from ca_support import ScoreHead


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seeds(0)
    key=args.backbone.lower();bank=OUT/'head_banks'/key;bm=read(bank/'manifest.json');assert bm['execution_status']=='completed'
    folder=OUT/'checkpoints'/key/'head';folder.mkdir(exist_ok=True)
    if (folder/'best.pt').exists() and not args.resume:raise FileExistsError('Existing head run requires explicit --resume')
    features=torch.from_numpy(np.load(bank/'features.npy')).to('cuda:0');target=torch.from_numpy(np.load(bank/'targets.npy')).to('cuda:0')
    model=ScoreHead(args.backbone).to('cuda:0');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    epochs=1000;scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,epochs=epochs,steps_per_epoch=225,pct_start=.3)
    generator=torch.Generator().manual_seed(0);best=float('inf');best_epoch=0;first=0;previous_seconds=0.
    if args.resume:
        saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['generator']);first=saved['epochs_completed'];best=saved['best'];best_epoch=saved['best_epoch']
        previous_seconds=read(folder/'manifest.json').get('training_seconds',0.)
        archived=folder/f'recovery_before_epoch_{first}_{int(time.time())}';archived.mkdir()
        for name in ['best.pt','manifest.json']:shutil.copy2(folder/name,archived/name)
        torch.save(saved['best_model'],folder/'best.pt')
    start=time.perf_counter()
    manifest=dict(execution_status='running',backbone=args.backbone,head_seed=0,epochs=epochs,epochs_completed=0,
                  blocks=2,feature_dim=128,head_architecture='TransolverGradientHead' if args.backbone=='Transolver3' else 'two official LinearNO_temp blocks',
                  parameter_count=sum(p.numel() for p in model.parameters()),bank_manifest=str(bank/'manifest.json'),
                  operator_sha256=bm['operator_sha256'],normalizer_sha256=bm['normalizer_sha256'],
                  target='zscore_log1p_grad_lap',target_mean=bm['target_mean'],target_std=bm['target_std'],
                  training_ids=bm['training_ids'],validation_ids=bm['validation_ids'],test_used_for_selection=False,
                  training_loss='smooth_l1_loss beta1 mean over all coarse values',selection='mean validation per-case vector error norm')
    write(folder/'manifest.json',manifest)
    manifest.update(epochs_completed=first,resumed_from_epoch=first,previous_training_seconds=previous_seconds,
                    source_sha256=digest(__file__),bank_manifest_sha256=digest(bank/'manifest.json'))
    write(folder/'manifest.json',manifest)
    with (OUT/'logs'/f'{key}_head_history.jsonl').open('a') as stream:
        for epoch in range(first,epochs):
            model.train();total=torch.zeros((),device='cuda:0');order=torch.randperm(900,generator=generator).to('cuda:0')
            for offset in range(0,900,4):
                ids=order[offset:offset+4];prediction=model(features[ids]);loss=torch.nn.functional.smooth_l1_loss(prediction,target[ids],beta=1.)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optimizer.step();scheduler.step();total+=loss.detach()*len(ids)
            model.eval();values=[]
            with torch.no_grad():
                for offset in range(900,1000,8):
                    prediction=model(features[offset:offset+8]);values.append(torch.linalg.vector_norm((prediction-target[offset:offset+8]).flatten(1),dim=1))
            validation=float(torch.cat(values).mean());train=float(total/900)
            if not np.isfinite(validation+train):raise RuntimeError('nonfinite head loss')
            if validation<best:
                best=validation;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'best.pt')
            row=dict(epoch=epoch+1,train_loss=train,validation=validation,best_validation=best,best_epoch=best_epoch)
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            manifest.update(epochs_completed=epoch+1,best_validation=best,best_epoch=best_epoch,training_seconds=previous_seconds+time.perf_counter()-start);write(folder/'manifest.json',manifest)
            write(OUT/'runtime'/f'{key}_head_progress.json',dict(execution_status='running',pid=os.getpid(),**row))
            if (epoch+1)%25==0 or epoch+1==epochs:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epochs_completed=epoch+1,generator=generator.get_state(),best=best,best_epoch=best_epoch,best_model=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)),folder/'resume.tmp.pt')
                (folder/'resume.tmp.pt').replace(folder/'resume.pt')
    torch.save(model.state_dict(),folder/'last.pt');manifest.update(execution_status='completed',checkpoint=str(folder/'best.pt'),checkpoint_sha256=digest(folder/'best.pt'),last_checkpoint_sha256=digest(folder/'last.pt'));write(folder/'manifest.json',manifest)
    write(OUT/'runtime'/f'{key}_head_progress.json',dict(execution_status='completed',pid=os.getpid(),epochs_completed=epochs))

if __name__=='__main__':main()
