from __future__ import annotations

import torch


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return one full-field relative L2 value per leading sample."""
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim < 2:
        raise ValueError("relative_l2 expects a batch/sample dimension")
    flat_pred = prediction.reshape(prediction.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(flat_pred - flat_target, dim=1)
    denominator = torch.linalg.vector_norm(flat_target, dim=1).clamp_min(eps)
    return numerator / denominator
