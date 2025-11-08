"""
Top-level package for the GNN-based trajectory prediction project.

The modules herein provide composable building blocks for:
1. data ingestion / preprocessing,
2. model definitions (LSTM baselines, GNN prototypes), and
3. training + evaluation utilities.

Actual model experiments should extend these foundational components.
"""

__all__ = [
    "config",
    "data",
    "models",
    "metrics",
    "utils",
]
