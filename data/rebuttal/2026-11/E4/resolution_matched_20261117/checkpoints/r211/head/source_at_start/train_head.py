"""Train each native two-block CA head with its own resolution-specific bank."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from head_support import OUT, CODE, ScoreHead, coarse_count, digest, head_paths, head_settings, read, tensor_digest, write


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    for attempt in range(5):
        try:
            torch.save(value, temporary)
            temporary.replace(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(.15 * (attempt + 1))


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def restore_history(folder, epoch):
    path = folder / 'history.jsonl'
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not any(row['epoch'] > epoch for row in rows):
        return
    archive = folder / 'recovery' / f'before_resume_{time.time_ns()}'
    archive.mkdir(parents=True)
    for name in ['history.jsonl', 'manifest.json', 'best.pt']:
        if (folder / name).exists():
            shutil.copyfile(folder / name, archive / name)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['epoch'] <= epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, choices=[85, 141, 211], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--micro-batch', type=int)
    args = parser.parse_args()
    resolution = args.resolution
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    protocol = read(OUT / 'protocol.json')
    settings = head_settings()
    assert world == protocol['head']['world_size'], 'Head process count differs from locked protocol'
    assert settings['epochs'] == protocol['head']['epochs']
    assert settings['effective_batch'] == protocol['head']['effective_batch_size']
    assert settings['effective_batch'] % world == 0
    local_batch = settings['effective_batch'] // world
    micro = args.micro_batch or local_batch
    assert micro > 0 and local_batch % micro == 0
    accumulation = local_batch // micro
    torch.set_num_threads(4)
    torch.manual_seed(settings['head_seed'])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if world > 1:
        dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    bank, folder = head_paths(resolution)
    folder.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (folder / 'training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1:
        dist.barrier()
    try:
        bm = read(bank / 'manifest.json')
        backbone = read(OUT / f'checkpoints/r{resolution}/manifest.json')
        assert bm['execution_status'] == backbone['execution_status'] == 'completed'
        assert bm['resolution'] == resolution and bm['coarse_count'] == coarse_count(resolution)
        assert bm['operator_sha256'] == backbone['checkpoint_sha256']
        assert digest(OUT / f'checkpoints/r{resolution}/best.pt') == bm['operator_sha256']
        assert bm['normalizer_sha256'] == backbone['normalizer_sha256']
        assert digest(OUT / f'data/r{resolution}/normalizer.npz') == bm['normalizer_sha256']
        assert bm['training_ids'] == list(range(900)) and bm['validation_ids'] == list(range(900, 1000))
        assert bm['test_ids_used'] == []
        count = coarse_count(resolution)
        # No physical test input or target is opened by the head trainer.
        features_array = np.load(bank / 'features.npy', mmap_mode='r', allow_pickle=False)
        targets_array = np.load(bank / 'targets.npy', mmap_mode='r', allow_pickle=False)
        assert features_array.shape == (1000, count, 128) and targets_array.shape == (1000, count, 1)
        assert features_array.dtype == targets_array.dtype == np.float32
        features = torch.from_numpy(np.array(features_array, copy=True)).to(device)
        targets = torch.from_numpy(np.array(targets_array, copy=True)).to(device)
        assert bool(torch.isfinite(features).all() and torch.isfinite(targets).all())
        native = ScoreHead().to(device)
        initial_tensor_sha = tensor_digest(native)
        optimizer = torch.optim.AdamW(native.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=settings['learning_rate'],
            epochs=settings['epochs'], steps_per_epoch=settings['steps_per_epoch'], pct_start=settings['pct_start'])
        generator = torch.Generator().manual_seed(settings['head_seed'])
        first = 0; best = float('inf'); best_epoch = 0; prior_seconds = 0.; prior_steps = 0
        manifest_path = folder / 'manifest.json'
        previous = read(manifest_path) if manifest_path.exists() else {}
        resume_path = folder / 'resume.pt'
        if previous.get('execution_status') == 'completed':
            assert previous['operator_sha256'] == bm['operator_sha256']
            assert previous['resolution'] == resolution
            assert digest(folder / 'best.pt') == previous['checkpoint_sha256']
            if rank == 0:
                print(f'r{resolution} head already complete', flush=True)
            return
        if resume_path.exists():
            if not args.resume:
                raise FileExistsError('Head resume checkpoint exists; explicit --resume is required')
            saved = torch.load(resume_path, map_location='cpu', weights_only=False)
            assert saved['operator_sha256'] == bm['operator_sha256']
            assert saved['normalizer_sha256'] == bm['normalizer_sha256']
            assert saved['resolution'] == resolution
            native.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            generator.set_state(saved['generator_state'])
            first = saved['epochs_completed']; best = saved['best_validation']; best_epoch = saved['best_epoch']
            assert saved['logical_steps'] == first * settings['steps_per_epoch']
            prior_seconds = float(previous.get('training_seconds', saved.get('training_seconds', 0.)))
            prior_steps = int(previous.get('executed_optimizer_steps', saved['logical_steps']))
            initial_tensor_sha = saved['initial_tensor_sha256']
            if rank == 0:
                restore_history(folder, first)
                save_torch(folder / 'best.pt', saved['best_model'])
        elif previous or (folder / 'best.pt').exists():
            raise FileExistsError('Partial head files without a resume checkpoint; preserve and recover explicitly')
        elif (folder / 'initial.pt').exists():
            if not args.resume:
                raise FileExistsError('Initial head file exists; use --resume to verify and preserve it')
            initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
            expected = cpu_state(native)
            assert set(initial) == set(expected)
            assert all(torch.equal(initial[name], expected[name]) for name in expected), 'Initial seed0 head tensors changed'
        elif rank == 0:
            save_torch(folder / 'initial.pt', cpu_state(native))
        if world > 1:
            dist.barrier()
        model = DistributedDataParallel(native, device_ids=[local_rank], output_device=local_rank) if world > 1 else native
        start = time.perf_counter()
        state = dict(
            execution_status='running', benchmark='darcy', method='ca', backbone='Transolver3',
            resolution=resolution, nfull=resolution*resolution, coarse_count=count,
            head_seed=0, epochs=settings['epochs'], epochs_completed=first,
            total_optimization_steps=first*settings['steps_per_epoch'], executed_optimizer_steps=prior_steps,
            architecture='TransolverGradientHead', blocks=2, hidden_dim=128, heads=8, slices=64, mlp_ratio=2,
            parameter_count=sum(p.numel() for p in native.parameters()), initial_tensor_sha256=initial_tensor_sha,
            operator_sha256=bm['operator_sha256'], operator_tensor_sha256=bm['operator_tensor_sha256'],
            normalizer_sha256=bm['normalizer_sha256'], bank_manifest=str(bank/'manifest.json'),
            bank_manifest_sha256=digest(bank/'manifest.json'), target='zscore_log1p_grad_lap',
            target_mean=bm['target_mean'], target_std=bm['target_std'],
            training_ids=bm['training_ids'], validation_ids=bm['validation_ids'], test_used_for_selection=False,
            settings=settings, world_size=world, local_batch_size=local_batch, micro_batch_size=micro,
            gradient_accumulation_steps=accumulation,
            ddp_loss_rule='local mean Huber times micro_batch/local_batch; DDP gradient mean gives global mean',
            validation_rule=settings['selection'], best_validation=None if first==0 else best, best_epoch=best_epoch,
            source_sha256={name:digest(CODE/name) for name in ['train_head.py','head_support.py']},
            source_bank_shapes=bm['file_shapes'], test_inputs_opened=[],
            runtime=dict(python=__import__('platform').python_version(), torch=torch.__version__,
                         cuda=torch.version.cuda, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         physical_gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                         dtype='float32', tf32_matmul=False, tf32_cudnn=False,
                         cpu_affinity=sorted(os.sched_getaffinity(0))),
            prior_attempt_training_seconds=prior_seconds, training_seconds=prior_seconds,
            resumed_from_epoch=first, resumed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            actual_training_shapes=list(previous.get('actual_training_shapes', [])))
        if rank == 0:
            sources = folder / 'source_at_start'; sources.mkdir(exist_ok=True)
            for name in state['source_sha256']:
                target = sources/name
                if target.exists() and digest(target) != state['source_sha256'][name]:
                    raise RuntimeError('Head numerical source changed across resume')
                if not target.exists():shutil.copyfile(CODE/name,target)
            write(manifest_path,state)
        if world > 1:
            dist.barrier()
        history = (folder/'history.jsonl').open('a') if rank==0 else None
        try:
            for epoch in range(first, settings['epochs']):
                epoch_start=time.perf_counter(); model.train(); order=torch.randperm(900,generator=generator)
                total=torch.zeros((),dtype=torch.float64,device=device)
                for offset in range(0,900,settings['effective_batch']):
                    global_ids=order[offset:offset+settings['effective_batch']]
                    local_ids=global_ids[rank*local_batch:(rank+1)*local_batch].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    for lower in range(0,local_batch,micro):
                        selected=local_ids[lower:lower+micro]
                        synchronize = lower+micro == local_batch or world==1
                        context = nullcontext() if synchronize else model.no_sync()
                        with context:
                            prediction=model(features[selected])
                            loss=torch.nn.functional.smooth_l1_loss(prediction,targets[selected],beta=1.,reduction='mean')
                            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite head training loss')
                            (loss*(len(selected)/local_batch)).backward()
                        total+=loss.detach().to(torch.float64)*len(selected)
                    torch.nn.utils.clip_grad_norm_(native.parameters(),settings['max_grad_norm'])
                    optimizer.step();scheduler.step()
                    if offset==0 and rank==0 and (epoch==first or epoch==settings['epochs']-1):
                        state['actual_training_shapes'].append(dict(epoch=epoch+1,
                            global_sample_ids=global_ids.tolist(),local_sample_ids=local_ids.cpu().tolist(),
                            features=list(features[local_ids].shape),target=list(targets[local_ids].shape),
                            micro_prediction=list(prediction.shape),native_resolution=[resolution,resolution]))
                native.eval(); validation_rows=torch.arange(900+rank,1000,world,device=device)
                summary=torch.zeros(2,dtype=torch.float64,device=device)
                with torch.no_grad():
                    for offset in range(0,len(validation_rows),8):
                        selected=validation_rows[offset:offset+8]
                        prediction=native(features[selected])
                        errors=torch.linalg.vector_norm((prediction-targets[selected]).flatten(1),dim=1)
                        summary[0]+=errors.to(torch.float64).sum();summary[1]+=len(selected)
                if world>1:
                    dist.all_reduce(summary);dist.all_reduce(total)
                assert int(summary[1].item())==100
                validation=float((summary[0]/summary[1]).item());train=float((total/900).item())
                if not np.isfinite(validation+train):raise RuntimeError('Nonfinite head validation')
                if validation<best:
                    best=validation;best_epoch=epoch+1
                    if rank==0:save_torch(folder/'best.pt',cpu_state(native))
                logical_steps=(epoch+1)*settings['steps_per_epoch']
                row=dict(epoch=epoch+1,global_step=logical_steps,train_loss=train,validation=validation,
                         best_validation=best,best_epoch=best_epoch,
                         learning_rate=scheduler.get_last_lr()[0],epoch_seconds=time.perf_counter()-epoch_start)
                if rank==0:
                    history.write(json.dumps(row)+'\n');history.flush()
                    state.update(epochs_completed=epoch+1,total_optimization_steps=logical_steps,
                        executed_optimizer_steps=prior_steps+(epoch-first+1)*settings['steps_per_epoch'],
                        best_validation=best,best_epoch=best_epoch,
                        training_seconds=prior_seconds+time.perf_counter()-start)
                    write(manifest_path,state)
                    write(OUT/f'runtime/head_r{resolution}.json',dict(execution_status='running',pid=os.getpid(),resolution=resolution,**row))
                    print(json.dumps(dict(resolution=resolution,stage='head',**row)),flush=True)
                    if (epoch+1)%10==0 or epoch+1==settings['epochs']:
                        save_torch(resume_path,dict(model=cpu_state(native),optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),generator_state=generator.get_state(),
                            epochs_completed=epoch+1,logical_steps=logical_steps,best_validation=best,best_epoch=best_epoch,
                            best_model=torch.load(folder/'best.pt',map_location='cpu',weights_only=True),
                            resolution=resolution,operator_sha256=bm['operator_sha256'],normalizer_sha256=bm['normalizer_sha256'],
                            initial_tensor_sha256=initial_tensor_sha,training_seconds=state['training_seconds']))
                if world>1:dist.barrier()
            if rank==0:
                save_torch(folder/'last.pt',cpu_state(native))
                state.update(execution_status='completed',checkpoint=str(folder/'best.pt'),
                    checkpoint_sha256=digest(folder/'best.pt'),last_checkpoint_sha256=digest(folder/'last.pt'),
                    initial_checkpoint_sha256=digest(folder/'initial.pt'),
                    training_seconds=prior_seconds+time.perf_counter()-start)
                state['head_sha256'] = state['checkpoint_sha256']
                write(manifest_path,state)
                write(OUT/f'runtime/head_r{resolution}.json',dict(execution_status='completed',resolution=resolution,
                    epochs_completed=settings['epochs'],total_optimization_steps=settings['total_updates'],manifest=str(manifest_path)))
        finally:
            if history is not None:history.close()
        if world>1:dist.barrier()
    finally:
        if lock is not None:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
