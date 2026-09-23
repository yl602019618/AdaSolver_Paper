"""Full-point Transolver3 training with the original global batch and schedule."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from adasolver.data import Airfoil2DAdapter
from adasolver.metrics import relative_l2
from adasolver.models import Transolver3Adapter
from adasolver.train.airfoil2d import build_manifest, git_revision, sha256_file
from adasolver.train.pipe import variable_point_subset
from adasolver.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.name = 'airfoil2d_transolver3_pressure_fullmesh'
    args.seed, args.batch_size = 0, 8
    args.learning_rate, args.weight_decay = .001, 1e-5
    args.max_grad_norm, args.pct_start = .1, .3
    args.nmin_fraction, args.train_samples, args.validation_samples, args.smoke = 1., 1000, 200, False
    args.command, args.git_revision = ' '.join(os.sys.argv), git_revision()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    assert args.batch_size % world == 0
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device)
    generator = seed_everything(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    data = Airfoil2DAdapter(args.data_path, ntrain=1000, ntest=200).load()
    n = data.train.pos.shape[1]
    train = DataLoader(TensorDataset(data.train.pos, data.train.target), batch_size=8, shuffle=True, generator=shuffle)
    validation = DataLoader(TensorDataset(data.test.pos, data.test.target), batch_size=8)
    module = Transolver3Adapter().to(device)
    model = DDP(module, device_ids=[rank], broadcast_buffers=False)
    data.y_normalizer.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, epochs=args.epochs,
        steps_per_epoch=len(train), pct_start=.3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / (args.name + '_best.pt')
    resume_path = args.output_dir / 'resume.pt'
    start, best = 0, float('inf')
    if args.resume:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        module.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start, best = saved['epoch'] + 1, saved['best']
        generator.set_state(saved['subset_generator'].cpu()); shuffle.set_state(saved['shuffle_generator'].cpu())
    if rank == 0:
        np.savez(args.output_dir / (args.name + '_ynorm.npz'),
            mean=data.y_normalizer.mean.cpu().numpy(), std=data.y_normalizer.std.cpu().numpy())
    overall = time.perf_counter()
    for epoch in range(start, args.epochs):
        begun = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device, dtype=torch.float64)
        for pos, target in train:
            # All ranks advance the same global shuffle and point permutation.
            selected = variable_point_subset(n, n, generator)
            lo, hi = rank * len(pos)//world, (rank+1) * len(pos)//world
            x = pos[lo:hi, selected].to(device)
            y = target[lo:hi, selected].to(device)
            loss = relative_l2(model(x, x), y).sum()
            optimizer.zero_grad(set_to_none=True)
            # DDP averages gradients; this restores the original summed batch-8 loss.
            (loss * world).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step(); scheduler.step(); total += loss.detach()
        dist.all_reduce(total)
        module.eval()
        errors = torch.zeros((), device=device, dtype=torch.float64)
        with torch.no_grad():
            for pos, target in validation:
                lo, hi = rank * len(pos)//world, (rank+1) * len(pos)//world
                x, y = pos[lo:hi].to(device), target[lo:hi].to(device)
                errors += relative_l2(data.y_normalizer.decode(module(x,x)), y).sum()
        dist.all_reduce(errors)
        value = errors.item() / 200
        if value < best:
            best = value
            if rank == 0:
                torch.save(module.state_dict(), best_path)
        seconds = time.perf_counter() - begun
        if rank == 0:
            status = dict(epoch=epoch+1, epochs=args.epochs, train_relative_l2=total.item()/1000,
                validation_relative_l2=value, best_validation_relative_l2=best,
                epoch_seconds=seconds, elapsed_seconds=time.perf_counter()-overall,
                estimated_remaining_seconds=seconds*(args.epochs-epoch-1), world_size=world,
                global_batch_size=8, status='complete' if epoch+1==args.epochs else 'running')
            (args.output_dir.parent / 'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps(status), flush=True)
            if (epoch+1)%10 == 0 or epoch+1 == args.epochs:
                torch.save(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epoch, best=best,
                    subset_generator=generator.get_state(), shuffle_generator=shuffle.get_state()),resume_path)
        dist.barrier()
    if rank == 0:
        torch.save(module.state_dict(), args.output_dir / (args.name + '_last.pt'))
        manifest = build_manifest(args,best,n,n,sha256_file(best_path))
        manifest.update(distributed_world_size=world,global_batch_size=8,per_rank_batch_size=8//world,
            gradient_reduction='DDP mean multiplied by world size to match summed global-batch loss',
            precision='float32, original PyTorch defaults',status='official')
        (args.output_dir / (args.name + '_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
