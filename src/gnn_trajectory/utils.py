"""
Utility helpers shared across modules.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Set global seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["seed_everything", "ensure_dir"]
