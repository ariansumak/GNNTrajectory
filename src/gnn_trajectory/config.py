"""
Central configuration schemas for experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


@dataclass
class DataConfig:
    root: Path = DEFAULT_DATA_ROOT
    split: str = "train"
    obs_seconds: float = 7.0
    fut_seconds: float = 4.0
    frequency_hz: float = 10.0
    max_agents: int = 128
    max_lanes: int = 256
    max_lane_points: int = 40
    agent_radius: float = 30.0
    lane_knn: int = 3
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class ModelConfig:
    history_steps: int
    future_steps: int
    embedding_dim: int = 64


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    grad_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50
    checkpoint_dir: Path = Path("checkpoints")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def materialize_model_config(self) -> ModelConfig:
        history_steps = int(self.data.obs_seconds * self.data.frequency_hz)
        future_steps = int(self.data.fut_seconds * self.data.frequency_hz)
        if self.model is None:
            self.model = ModelConfig(
                history_steps=history_steps,
                future_steps=future_steps,
            )
        else:
            self.model.history_steps = history_steps
            self.model.future_steps = future_steps
        return self.model
