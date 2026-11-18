"""Independently audit saved matched-resolution training histories and weights.

This script never opens evaluation arrays and never changes training files.
Use --partial for an explicitly provisional audit while runs are active.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
from pathlib import Path

import torch

from common import (CODE, OUT, ResolutionData, configure_precision, digest,
                    load_model, read, relative_error, tensor_state_digest, write)


def expected_lr(step):
    """Closed-form default cosine OneCycleLR; includes its final extra step."""
    initial = .001 / 25
    end_first = .3 * 225000 - 1
    if step <= end_first:
        start, end, fraction = initial, .001, step / end_first
    else:
        start, end = .001, initial / 10000
        fraction = (step - end_first) / (225000 - 1 - end_first)
    return end + (start - end) / 2 * (math.cos(math.pi * fraction) + 1)


def reload_validation(resolution, device):
    """Recompute the selected checkpoint's recorded validation mean only."""
    configure_precision(0, 4)
    data = ResolutionData(resolution)
    features, target = data.load_split('validation')
    model, manifest = load_model(resolution, device)
    features, target = features.to(device), target.to(device)
    positions = data.positions.to(device)
    mean, std = data.y_mean.to(device), data.y_std.to(device)
    totals = []
    per_case = []
    shapes = set()
    with torch.no_grad():
        # Match the original rank0/rank1 shards and final short batch exactly.
        for start, stop in [(0, 50), (50, 100)]:
            total = torch.zeros((), dtype=torch.float64, device=device)
            for lo in range(start, stop, 4):
                hi = min(stop, lo + 4)
                fx = features[lo:hi]
                prediction = model(positions[None].expand(hi-lo, -1, -1), fx) * std + mean
                shapes.add(tuple(prediction.shape))
                errors = relative_error(prediction, target[lo:hi])
                total += errors.to(torch.float64).sum()
                per_case.extend(dict(case_id=900+lo+i, relative_l2=float(value))
                                for i, value in enumerate(errors.cpu().tolist()))
            totals.append(total)
        observed = float(((totals[0] + totals[1]) / 100).item())
    expected = manifest['best_validation_relative_l2']
    # Fixed before evaluation: the same FP32 graph and shard reduction are reused.
    tolerance = 1e-12 + 1e-7 * abs(expected)
    passed = abs(observed - expected) <= tolerance
    passed = passed and all(not Path(p).name.startswith('test_') for p in data.accessed_files)
    payload = dict(passed=passed, resolution=resolution,
                   evaluated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   checkpoint_sha256=manifest['checkpoint_sha256'],
                   normalizer_sha256=data.normalizer_sha256,
                   selected_epoch=manifest['best_epoch'],
                   expected_validation_relative_l2=expected,
                   observed_validation_relative_l2=observed,
                   absolute_difference=abs(observed-expected),
                   tolerance=tolerance, tolerance_rule='1e-12 + 1e-7 * abs(recorded validation mean)',
                   identical_float_value=observed == expected,
                   validation_ids=list(range(900, 1000)),
                   actual_prediction_shapes=[list(shape) for shape in sorted(shapes)],
                   data_access=data.accessed_files, device=str(device),
                   precision='FP32; TF32 matmul/cuDNN disabled; same validation batch and shard layout',
                   per_case=per_case, verifier_sha256=digest(Path(__file__)))
    write(OUT / f'results/selected_checkpoint_validation_r{resolution}.json', payload)
    return payload


def audit_resolution(resolution, partial, reload_selected=False, device='cpu'):
    folder = OUT / 'checkpoints' / f'r{resolution}'
    manifest = read(folder / 'manifest.json')
    completed = manifest['execution_status'] == 'completed'
    committed_epochs = int(manifest['epochs_completed'])
    rows = [read(folder / 'history' / f'epoch_{epoch:04d}.json')
            for epoch in range(1, committed_epochs + 1)]
    failures = []

    def check(condition, reason):
        if not condition:
            failures.append(reason)

    check(partial or completed, 'Training is not completed')
    check(partial or committed_epochs == 1000, 'Training does not contain 1000 epochs')
    check(manifest['total_optimization_steps'] == committed_epochs * 225,
          'Manifest optimizer update count differs from 225 per epoch')
    check(manifest['training_ids'] == list(range(900)), 'Training IDs differ')
    check(manifest['validation_ids'] == list(range(900, 1000)), 'Validation IDs differ')
    check(manifest['trained_from_scratch'] is True, 'Backbone is not marked scratch-trained')
    check(manifest['training_seed'] == 0, 'Training seed differs')
    check(manifest['effective_batch_size'] == 4, 'Effective batch differs')
    check(manifest['world_size'] == 2 and manifest['local_batch_size'] == 2,
          'DDP distribution differs')
    check(manifest['epochs'] == 1000 and manifest['updates_per_epoch'] == 225,
          'Training schedule differs')
    check(manifest['test_used_for_selection'] is False and
          manifest['test_opened_by_training'] is False, 'Test use declared')
    check(all(not Path(p).name.startswith('test_') for p in manifest['input_data_access']),
          'Training data access contains test arrays')
    check(manifest['train_coefficient_shape'] == [900, resolution**2, 1],
          'Native training coefficient shape differs')
    check(manifest['validation_coefficient_shape'] == [100, resolution**2, 1],
          'Native validation coefficient shape differs')
    check(manifest['positions_shape'] == [resolution**2, 2],
          'Native position shape differs')
    check(manifest['model_config'] == dict(hidden_dim=128, layers=8, heads=8,
          slices=64, mlp_ratio=2, space_dim=2, fun_dim=1, out_dim=1, dropout=0.),
          'Architecture differs')
    check(manifest['optimizer'] == dict(name='AdamW', lr=.001,
          weight_decay=1e-5, max_grad_norm=.1), 'Optimizer differs')
    check(manifest['scheduler'] == dict(name='OneCycleLR', pct_start=.3,
          steps_per_epoch=225, total_steps=225000), 'Scheduler differs')
    check(manifest['normalizer_sha256'] == digest(OUT / 'data' / f'r{resolution}' / 'normalizer.npz'),
          'Normalizer changed after training started')
    check(manifest['protocol_sha256'] == digest(OUT / 'protocol.json'), 'Protocol changed')
    check(manifest['split_manifest_sha256'] == digest(OUT / 'split_manifest.json'),
          'Split manifest changed')
    for name, expected_hash in manifest['source_sha256'].items():
        check(digest(CODE / name) == expected_hash, f'Live source changed: {name}')
        check(digest(folder / 'source_at_start' / name) == expected_hash,
              f'Source snapshot changed: {name}')
    for name, expected_hash in manifest['native_model_source_sha256'].items():
        check(digest(CODE / name) == expected_hash, f'Native model source changed: {name}')

    initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
    initial_tensor_hash = tensor_state_digest(initial)
    check(digest(folder / 'initial.pt') == manifest['initial_checkpoint_sha256'],
          'Initial checkpoint changed')
    check(initial_tensor_hash == manifest['initial_state_tensor_sha256'],
          'Initial tensor state differs')

    generator = torch.Generator().manual_seed(0)
    best, best_epoch = math.inf, 0
    for epoch, row in enumerate(rows, 1):
        prefix = f'epoch{epoch}: '
        check(row['epoch'] == epoch and row['global_step'] == epoch * 225,
              prefix + 'epoch/update sequence differs')
        check(row['actual_native_grid'] == [resolution, resolution] and
              row['actual_prediction_shape'] == [manifest['microbatch_size'], resolution**2, 1],
              prefix + 'actual full-grid prediction shape differs')
        for key in ('train_loss', 'train_field_l2', 'train_derivative_loss',
                    'validation_field_l2', 'last_unclipped_gradient_norm',
                    'epoch_seconds', 'training_wall_seconds'):
            check(math.isfinite(row[key]), prefix + f'{key} is nonfinite')
        check(math.isclose(row['learning_rate'], expected_lr(epoch * 225),
                           rel_tol=1e-13, abs_tol=1e-16),
              prefix + 'learning rate differs from locked OneCycle schedule')
        order = torch.randperm(900, generator=generator)
        order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        check(row['global_data_order_sha256'] == order_hash and
              row['first_effective_batch_ids'] == order[:4].tolist(),
              prefix + 'global batch ordering differs from seed0 schedule')
        if row['validation_field_l2'] < best:
            best, best_epoch = row['validation_field_l2'], epoch
        check(row['best_validation'] == best and row['best_epoch'] == best_epoch,
              prefix + 'checkpoint selection differs from validation minimum')

    if rows:
        check(manifest['best_validation_relative_l2'] == best and
              manifest['best_epoch'] == best_epoch, 'Manifest validation selection differs')
    checkpoint_hash = None
    validation_receipt = None
    if completed:
        check(len(list((folder / 'history').glob('epoch_*.json'))) == 1000,
              'Completed training has extra or missing history records')
        for name, field in [('best.pt', 'checkpoint_sha256'),
                            ('last.pt', 'last_checkpoint_sha256'),
                            ('resume.pt', 'resume_checkpoint_sha256')]:
            check(digest(folder / name) == manifest[field], f'{name} hash differs')
        checkpoint_hash = manifest['checkpoint_sha256']
        saved = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
        best_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        last_state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        check(saved['epochs_completed'] == 1000, 'Resume epoch is incomplete')
        check(saved['scheduler']['last_epoch'] == 225000, 'Resume scheduler update count differs')
        check(all(int(value['step']) == 225000 for value in saved['optimizer']['state'].values()),
              'AdamW parameter update count differs')
        check(saved['best_epoch'] == best_epoch and saved['best_validation'] == best,
              'Resume selection differs from validation history')
        check(tensor_state_digest(saved['best_model']) == tensor_state_digest(best_state) ==
              manifest['best_state_tensor_sha256'], 'Best tensor state differs between saves')
        check(tensor_state_digest(saved['model']) == tensor_state_digest(last_state),
              'Last tensor state differs from final resume state')
        if not partial:
            receipt_path = OUT / f'results/selected_checkpoint_validation_r{resolution}.json'
            if reload_selected:
                validation_receipt = reload_validation(resolution, device)
            elif receipt_path.exists():
                validation_receipt = read(receipt_path)
            check(validation_receipt is not None, 'Selected checkpoint has not been reloaded on validation')
            if validation_receipt is not None:
                check(validation_receipt['passed'], 'Selected checkpoint validation metric differs')
                check(validation_receipt['checkpoint_sha256'] == checkpoint_hash and
                      validation_receipt['normalizer_sha256'] == manifest['normalizer_sha256'],
                      'Validation reload receipt belongs to different weights/normalizer')
                check(validation_receipt['expected_validation_relative_l2'] == best and
                      validation_receipt['selected_epoch'] == best_epoch,
                      'Validation reload receipt differs from selected history')
                check(validation_receipt['validation_ids'] == list(range(900, 1000)),
                      'Validation reload population differs')

    return dict(resolution=resolution, execution_status=manifest['execution_status'],
                epochs_audited=len(rows), optimizer_updates_audited=225*len(rows),
                passed=not failures, failures=failures,
                best_validation_relative_l2=None if not rows else best,
                best_epoch=best_epoch, checkpoint_sha256=checkpoint_hash,
                initial_state_tensor_sha256=initial_tensor_hash,
                parameter_count=manifest['parameter_count'],
                manifest_sha256=digest(folder / 'manifest.json'),
                normalizer_sha256=manifest['normalizer_sha256'],
                source_sha256=manifest['source_sha256'],
                training_wall_seconds=manifest.get('training_wall_seconds'),
                selected_checkpoint_validation_passed=(None if validation_receipt is None else validation_receipt['passed']),
                selected_checkpoint_validation_difference=(None if validation_receipt is None else validation_receipt['absolute_difference']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--reload-validation', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[85, 141, 211],
                        choices=[85, 141, 211])
    args = parser.parse_args()
    if args.partial and args.reload_validation:
        parser.error('--reload-validation requires completed training')
    torch.set_num_threads(1)
    rows = [audit_resolution(r, args.partial, args.reload_validation, args.device) for r in args.resolutions]
    cross = dict(initial_tensor_identity=len({r['initial_state_tensor_sha256'] for r in rows}) == 1,
                 parameter_count_identity=len({r['parameter_count'] for r in rows}) == 1,
                 training_source_identity=len({tuple(sorted(r['source_sha256'].items())) for r in rows}) == 1)
    payload = dict(audited_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   status='provisional' if args.partial else 'final',
                   passed=all(r['passed'] for r in rows) and all(cross.values()),
                   all_requested_training_complete=all(r['execution_status']=='completed' for r in rows),
                   resolutions=rows, cross_resolution_checks=cross,
                   scope='saved histories, checkpoint states, native shapes, data access, fixed split, source and normalizer identity; selected checkpoint reload on validation only; no evaluation-population access or new selection',
                   verifier_sha256=digest(Path(__file__)))
    suffix = '_partial' if args.partial else ''
    if len(args.resolutions) == 1:
        suffix += f'_r{args.resolutions[0]}'
    destination = OUT / 'results' / f'training_verification{suffix}.json'
    write(destination, payload)
    print(destination)
    for row in rows:
        print(f"r{row['resolution']}: {row['epochs_audited']} epochs, passed={row['passed']}, failures={row['failures']}")
    if not payload['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
