from __future__ import annotations

import torch

from .field import relative_l2


def reconstruct_native(values: torch.Tensor, shape: tuple[int, int] = (85, 85)) -> torch.Tensor:
    """Reshape flattened Darcy fields using the native C-order index lattice."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3 or values.shape[1] != shape[0] * shape[1]:
        raise ValueError(
            f"expected (samples, {shape[0] * shape[1]}, channels), got {tuple(values.shape)}"
        )
    return values.reshape(values.shape[0], shape[0], shape[1], values.shape[2])


def darcy_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: tuple[int, int] = (85, 85),
) -> torch.Tensor:
    """Compute one full-field relative L2 error per Darcy sample."""
    pred = reconstruct_native(prediction, shape)
    truth = reconstruct_native(target, shape)
    return relative_l2(pred, truth)
