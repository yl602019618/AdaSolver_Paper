"""Resolution-specific CA heads over each model's frozen first-block features."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1]
CODE = OUT / 'code'
VENDOR = CODE / 'vendor'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def read(path):
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(.1 * (attempt + 1))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    for attempt in range(5):
        try:
            temp.write_text(payload)
            temp.replace(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(.15 * (attempt + 1))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_digest(model):
    """Stable state identity, independent of torch.save container metadata."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode() + b'\0')
        h.update(str(array.dtype).encode() + b'\0')
        h.update(json.dumps(list(array.shape)).encode() + b'\0')
        h.update(array.tobytes())
    return h.hexdigest()


def coarse_count(resolution):
    return max(8, int(round(resolution * resolution * .1)))


def head_paths(resolution):
    return OUT / f'head_banks/r{resolution}', OUT / f'checkpoints/r{resolution}/head'


def _require_vendor(module):
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(VENDOR.resolve()):
        raise RuntimeError(f'Expected frozen task-local AdaSolver source, loaded {origin}')


class ScoreHead(nn.Module):
    def __init__(self):
        super().__init__()
        import adasolver.inference.ca.head as native
        _require_vendor(native)
        self.body = native.TransolverGradientHead(
            hidden_dim=128, heads=8, mlp_ratio=2, slices=64, blocks=2)

    def forward(self, hidden):
        return self.body(hidden)


def model_core(operator):
    return operator.net if hasattr(operator, 'net') else operator


@torch.no_grad()
def frozen_features(operator, pos, fx):
    if pos.ndim != 3 or fx.ndim != 3 or pos.shape[:2] != fx.shape[:2]:
        raise ValueError('Coordinates and coefficient features must align as batch × points')
    core = model_core(operator)
    hidden = core.preprocess(torch.cat((pos, fx), dim=-1))
    hidden = hidden + core.placeholder[None, None, :]
    return core.blocks[0](hidden)


def coarse_grad_lap_score(reference, coarse_indices, prediction):
    """Original E4 raw grad-plus-index-Laplacian teacher, at the supplied grid.

    The gradient score uses physical-coordinate squared distances. The native
    nearest-grid Laplacian deliberately retains index increments (no spacing
    argument). This experiment does not introduce a new acquisition rule.
    """
    from scipy.interpolate import griddata
    import adasolver.inference.acquisition as acquisition
    _require_vendor(acquisition)
    reference = np.asarray(reference, dtype=np.float64)
    coarse_indices = np.asarray(coarse_indices, dtype=np.int64)
    coords = reference[coarse_indices]
    values = np.asarray(prediction, dtype=np.float64).reshape(len(coords), -1)
    gradient = acquisition.grad_score(coords, values, k=8)
    axis, ay = np.unique(reference[:, 0]), np.unique(reference[:, 1])
    if len(axis) != len(ay) or len(axis) * len(ay) != len(reference):
        raise ValueError('Teacher reference must be the complete declared square grid')
    gx, gy = np.meshgrid(axis, ay, indexing='xy')
    coarse = griddata(coords, values[:, 0], np.column_stack((gx.ravel(), gy.ravel())), method='nearest')
    field = coarse.reshape(len(ay), len(axis))
    lap = np.abs(np.gradient(np.gradient(field, axis=0), axis=0)
                 + np.gradient(np.gradient(field, axis=1), axis=1)).reshape(-1)
    xi = np.searchsorted(axis, coords[:, 0])
    yi = np.searchsorted(ay, coords[:, 1])
    return np.maximum(gradient + lap[yi * len(axis) + xi], 0.0)


def decode_score(prediction, mean, std):
    log_score = np.asarray(prediction, dtype=np.float64) * std + mean
    return np.expm1(np.clip(log_score, 0., None))


class HeadAcquisition(nn.Module):
    def __init__(self, operator, head, metadata):
        super().__init__()
        self.operator = operator.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()
        self.metadata = metadata

    def forward(self, pos, fx):
        return self.head(frozen_features(self.operator, pos, fx))

    def decode_score(self, prediction):
        return decode_score(prediction, self.metadata['target_mean'], self.metadata['target_std'])


def load_head(resolution, operator, device='cpu', *, require_verified=True):
    bank, folder = head_paths(resolution)
    metadata = read(folder / 'manifest.json')
    backbone = read(OUT / f'checkpoints/r{resolution}/manifest.json')
    normalizer = OUT / f'data/r{resolution}/normalizer.npz'
    if metadata.get('execution_status') != 'completed':
        raise RuntimeError('Corresponding head is incomplete')
    if metadata['resolution'] != resolution or metadata['coarse_count'] != coarse_count(resolution):
        raise ValueError('Head resolution/context mismatch')
    if metadata['operator_sha256'] != backbone['checkpoint_sha256']:
        raise ValueError('Head belongs to another backbone checkpoint')
    if metadata['normalizer_sha256'] != backbone['normalizer_sha256'] or digest(normalizer) != metadata['normalizer_sha256']:
        raise ValueError('Head/backbone/data normalizer mismatch')
    if digest(folder / 'best.pt') != metadata['checkpoint_sha256']:
        raise ValueError('Head weight file changed')
    if tensor_digest(operator) != metadata['operator_tensor_sha256']:
        raise ValueError('Actually loaded operator differs from the head feature source')
    if require_verified:
        acceptance = read(OUT / f'results/head_r{resolution}_validation.json')
        if not acceptance.get('passed') or acceptance['head_sha256'] != metadata['checkpoint_sha256']:
            raise ValueError('Head lacks the matching final acceptance receipt')
        if acceptance['operator_sha256'] != metadata['operator_sha256']:
            raise ValueError('Head acceptance belongs to another operator')
    head = ScoreHead().to(device)
    head.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True), strict=True)
    return HeadAcquisition(operator, head, metadata).to(device).eval(), metadata


def head_settings():
    return dict(epochs=1000, effective_batch=4, steps_per_epoch=225, total_updates=225000,
                learning_rate=.001, weight_decay=1e-5, max_grad_norm=.1,
                optimizer='AdamW', scheduler='OneCycleLR', pct_start=.3, head_seed=0,
                training_loss='smooth_l1_loss beta1 mean over global batch and all coarse values',
                selection='minimum validation mean per-case vector error norm',
                hidden_dim=128, layers=2, heads=8, slices=64, mlp_ratio=2)
