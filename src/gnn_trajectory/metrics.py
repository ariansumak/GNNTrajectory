"""
Collection of motion forecasting metrics.
"""
from __future__ import annotations

import torch


def average_displacement_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Average Displacement Error (ADE) over the prediction horizon.
    """

    return torch.linalg.norm(pred - target, dim=-1).mean()


def final_displacement_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Final Displacement Error (FDE) at the last predicted frame.
    """

    return torch.linalg.norm(pred[:, -1] - target[:, -1], dim=-1).mean()


__all__ = ["average_displacement_error", "final_displacement_error"]
