"""Small single-stress heads on frozen, sparse first-block features."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from adasolver.models.checkpoint import sha256_file
from adasolver.layers.first_block import FirstBlockCache, first_block_features, _prefix_forward


@dataclass(frozen=True)
class ElasticityFeatureCache:
    pointwise: FirstBlockCache
    coordinates: np.ndarray


@torch.no_grad()
def sparse_stress_features(operator, coordinates):
    """Capture the exact host inputs along with their pointwise GPU cache."""
    coordinates = np.array(coordinates, dtype=np.float32, copy=True)
    if coordinates.ndim != 3 or coordinates.shape[2] != 2 or not 2 <= coordinates.shape[1] <= 97:
        raise ValueError("Elasticity acquisition must use (B, 2..97, 2) coordinates")
    device = next(operator.parameters()).device
    features, pointwise = first_block_features(operator, torch.as_tensor(coordinates, device=device), capture=True)
    coordinates.setflags(write=False)
    return features, ElasticityFeatureCache(pointwise, coordinates)


@torch.no_grad()
def cached_stress_forward(operator, coordinates, cache, compensation):
    """Verify the coarse prefix on the host before transferring final inputs.

    Elasticity refinement constructs Sfinal as S0 followed by new points.
    Checking the captured host coordinates avoids the generic mapping path's
    GPU equality checks and index transfer, while retaining input/weight identity.
    The mathematical first-block implementation is shared unchanged.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    pointwise = cache.pointwise
    n0 = cache.coordinates.shape[1]
    if (coordinates.ndim != 3 or coordinates.shape[0] != cache.coordinates.shape[0]
            or coordinates.shape[2] != 2 or not n0 <= coordinates.shape[1] <= 972
            or not np.array_equal(coordinates[:, :n0], cache.coordinates)):
        raise ValueError("final coordinates do not contain the captured coarse prefix")
    if pointwise.operator_id != id(operator):
        raise ValueError("cache belongs to another operator")
    if pointwise.parameter_versions != tuple(p._version for p in operator.parameters()):
        raise ValueError("operator parameters changed after acquisition")
    device = next(operator.parameters()).device
    if pointwise.inputs.device != device or pointwise.inputs.dtype != torch.float32:
        raise ValueError("cached input dtype/device differs")
    pos = torch.as_tensor(coordinates, device=device)
    c = None if compensation is None else torch.as_tensor(compensation, device=device, dtype=pos.dtype)
    if c is not None:
        if c.ndim == 1:
            c = c[None].expand(pos.shape[0], -1)
        if c.shape != pos.shape[:2]:
            raise ValueError("final compensation does not align with coordinates")
    return _prefix_forward(operator, pos, pointwise, c, 131072)


class ElasticityStressHead(nn.Module):
    def __init__(self, bottleneck=4):
        super().__init__()
        if bottleneck not in (0, 1, 2, 4, 8, 16):
            raise ValueError("unsupported Elasticity bottleneck")
        self.bottleneck = bottleneck
        self.register_buffer("feature_mean", torch.zeros(128))
        self.register_buffer("feature_std", torch.ones(128))
        self.layers = (nn.Linear(128, 1) if bottleneck == 0 else
                       nn.Sequential(nn.Linear(128, bottleneck), nn.GELU(), nn.Linear(bottleneck, 1)))

    def forward(self, features):
        return self.layers((features - self.feature_mean) / self.feature_std)


def load_stress_head(checkpoint, manifest, *, operator_sha256, normalizer_sha256, device="cpu"):
    m = json.loads(Path(manifest).read_text())
    for key, expected in {"benchmark": "elasticity", "target": "zscore_stress",
                          "feature_tap": "frozen_block0_on_S0", "coarse_cap_count": 97,
                          "operator_sha256": operator_sha256,
                          "normalizer_sha256": normalizer_sha256}.items():
        if m.get(key) != expected:
            raise ValueError(f"stress head manifest mismatch: {key}")
    if not np.isfinite(m["target_std"]) or m["target_std"] <= 0:
        raise ValueError("invalid head target normalization")
    if sha256_file(checkpoint) != m["head_sha256"]:
        raise ValueError("stress head checkpoint does not match manifest")
    head = ElasticityStressHead(m["bottleneck_dim"]).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    head.eval()
    return head, m
