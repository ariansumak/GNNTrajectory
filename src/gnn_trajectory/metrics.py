"""
Collection of motion forecasting metrics.
"""
from __future__ import annotations

import torch


def _apply_mask(error: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return error.mean()
    weighted = (error * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return weighted / denom


def average_displacement_error(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes Average Displacement Error (ADE) over the prediction horizon.
    """

    error = torch.linalg.norm(pred - target, dim=-1)
    return _apply_mask(error, mask)


def final_displacement_error(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes Final Displacement Error (FDE) at the last predicted frame.
    """

    final_error = torch.linalg.norm(pred[..., -1, :] - target[..., -1, :], dim=-1)
    if mask is not None:
        mask = mask[..., -1]
    return _apply_mask(final_error, mask)


__all__ = ["average_displacement_error", "final_displacement_error"]
