"""Use the locked Car sampling/metrics with an explicitly identified LinearNO.

Only the runtime loader and the architecture-specific compensation insertion
are replaced. The geometry, sampling policies, reconstruction and metrics
come from the task-local frozen AdaSolver snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from adasolver.data.shapenet_car_runtime import CarRuntime
from adasolver.data.shapenet_car import _load_normalizers
from adasolver.models.checkpoint import sha256_file
from linearno_adapter import LinearNOSpec, install_linear_compensation


class LinearNORuntime(CarRuntime):
    def __init__(self, manifest, data_path=None, *, require_variable=True):
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.metadata = json.loads(self.manifest_path.read_text())
        meta = self.metadata
        if meta.get('benchmark') != 'shapenet_car' or meta.get('architecture') != 'LinearNO':
            raise ValueError('expected a ShapeNet-Car LinearNO manifest')
        if require_variable and meta.get('nmin_fraction') != .4:
            raise ValueError('this matched protocol requires nmin_fraction=.4')
        self.spec = LinearNOSpec(**meta['model'])
        self.checkpoint = Path(meta['checkpoint'])
        self.normalizer_path = Path(meta['normalizer'])
        for key, path in [('checkpoint', self.checkpoint), ('normalizer', self.normalizer_path)]:
            if sha256_file(path) != meta[key+'_sha256']:
                raise ValueError(key+' identity mismatch')
        self.data_path = Path(data_path or meta['data_path'])
        self.input_normalizer, self.output_normalizer = _load_normalizers(self.normalizer_path)

    def load_operator(self, device='cpu'):
        model = self.spec.build().to(device)
        model.load_state_dict(torch.load(self.checkpoint, map_location=device, weights_only=True))
        return model.requires_grad_(False).eval()


def main():
    from adasolver.evaluation import shapenet_car as evaluator
    evaluator.CarRuntime = LinearNORuntime
    evaluator.install_compensation = install_linear_compensation
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output', type=Path, required=True)
    args, _ = p.parse_known_args()
    evaluator.main()
    wrapper_files = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ['evaluate_linearno.py', 'linearno_adapter.py']
    }
    for path in args.output.glob('*/run_manifest.json'):
        data = json.loads(path.read_text())
        data.update(backbone='LinearNO', official_commit='3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
                    backbone_adapter_sources=wrapper_files,
                    correction_insertion='log(c) added to key logits before point-axis softmax',
                    input_interface='official seven-channel ShapeNetCar graph input packed as tensors')
        path.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
