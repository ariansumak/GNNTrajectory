"""
Common training losses.
"""
from __future__ import annotations

import torch


def masked_l2_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean L2 displacement error with optional temporal mask.
    """

    error = (preds - targets) ** 2
    error = error.sum(dim=-1)  # (B, A, T)
    if mask is not None:
        error = error * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(error.numel(), device=error.device, dtype=error.dtype)
    return error.sum() / denom


__all__ = ["masked_l2_loss"]
