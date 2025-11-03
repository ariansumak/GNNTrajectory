"""
Central configuration schemas for experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .training.trainer import TrainingConfig


@dataclass
class DataConfig:
    root: Path = Path("data")
    history_steps: int = 20
    future_steps: int = 30
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    history_steps: int
    future_steps: int
    embedding_dim: int = 64


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def materialize_model_config(self) -> ModelConfig:
        if self.model is None:
            self.model = ModelConfig(
                history_steps=self.data.history_steps,
                future_steps=self.data.future_steps,
            )
        return self.model
