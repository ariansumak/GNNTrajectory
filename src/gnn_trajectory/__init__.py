"""
Top-level package for the GNN-based trajectory prediction project.

The modules herein provide composable building blocks for:
1. data ingestion and graph construction,
2. model definitions leveraging PyTorch Geometric, and
3. training and evaluation utilities.

Actual model experiments should extend these foundational components.
"""

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "metrics",
    "utils",
]
