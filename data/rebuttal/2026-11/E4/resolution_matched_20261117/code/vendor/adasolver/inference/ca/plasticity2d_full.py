"""Small full20 field heads over frozen Transolver3 block-0 features."""
from pathlib import Path
import json
import torch
from torch import nn
from adasolver.models.checkpoint import sha256_file


class Full20FieldHead(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=128, out_dim=40):
        super().__init__()
        if feature_dim not in (1, 2, 4, 8, 16, 32, 128):
            raise ValueError("unsupported full20 feature bottleneck")
        self.feature_dim = feature_dim
        if out_dim not in (1, 40):
            raise ValueError("full20 heads predict either the 40 fields or one acquisition score")
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(hidden_dim, feature_dim), nn.GELU(),
                                 nn.Linear(feature_dim, 32), nn.GELU(), nn.Linear(32, out_dim))

    def forward(self, features):
        return self.net(features)


def load_full20_head(manifest, operator_metadata, device="cpu"):
    metadata = json.loads(Path(manifest).read_text())
    targets = {"raw_full20_displacement": 40, "zscore_log1p_full20_gradient": 1}
    if metadata.get("task_version") != "full20_v1" or metadata.get("target") not in targets:
        raise ValueError("not a supported full20 head")
    if metadata["operator_sha256"] != operator_metadata["checkpoint_sha256"]:
        raise ValueError("head belongs to another operator")
    if metadata["input_normalizer_sha256"] != operator_metadata["input_normalizer_sha256"]:
        raise ValueError("head normalizer differs from the operator")
    if metadata.get("coarse_count") != 313:
        raise ValueError("head must be trained on actual 313-point subsets")
    checkpoint = Path(metadata["checkpoint"])
    if sha256_file(checkpoint) != metadata["head_sha256"]:
        raise ValueError("head checksum mismatch")
    out_dim = targets[metadata["target"]]
    if metadata.get("out_dim", out_dim) != out_dim:
        raise ValueError("head target and output dimension differ")
    head = Full20FieldHead(metadata["feature_dim"], operator_metadata["model"]["hidden_dim"], out_dim)
    head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return head.to(device).eval(), metadata
