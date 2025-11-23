"""
Central configuration schemas for experiments plus helper loaders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/dtu/blackhole/07/224071/dataset")
#DEFAULT_DATA_ROOT = "/home/arian-sumak/Documents/DTU/Deep Learning"

def _to_path(value: Any) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(value).expanduser()


@dataclass
class DataConfig:
    root: Path = DEFAULT_DATA_ROOT
    split: str = "train"
    val_split: str | None = None
    obs_seconds: float = 7.0
    fut_seconds: float = 4.0
    frequency_hz: float = 10.0
    max_agents: int = 128
    max_lanes: int = 256
    max_lane_points: int = 40
    agent_radius: float = 50.0
    lane_knn: int = 3
    batch_size: int = 2
    num_workers: int = 0


@dataclass
class ModelConfig:
    history_steps: int = 0
    future_steps: int = 0
    encoder: str = "gcn_v2"
    decoder: str = "mlp"
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRSchedulerConfig:
    type: str = "plateau"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-3
    cooldown: int = 0
    min_lr: float = 0.0
    verbose: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    grad_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path | None = Path("runs")
    lr_scheduler: LRSchedulerConfig | None = None
    val_every_steps: int | None = 1000
    val_max_scenarios: int | None = None
    train_max_scenarios: int | None = None


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

    @staticmethod
    def _coerce_data_config(data_cfg: Dict[str, Any] | None) -> DataConfig:
        cfg = dict(data_cfg or {})
        if "root" in cfg:
            cfg["root"] = _to_path(cfg["root"]) or DEFAULT_DATA_ROOT
        return DataConfig(**cfg)

    @staticmethod
    def _coerce_model_config(model_cfg: Dict[str, Any] | None) -> ModelConfig | None:
        if not model_cfg:
            return None
        return ModelConfig(**model_cfg)

    @staticmethod
    def _coerce_training_config(training_cfg: Dict[str, Any] | None) -> TrainingConfig:
        cfg = dict(training_cfg or {})
        if "checkpoint_dir" in cfg:
            cfg["checkpoint_dir"] = _to_path(cfg["checkpoint_dir"]) or Path("checkpoints")
        if "log_dir" in cfg:
            cfg["log_dir"] = _to_path(cfg["log_dir"])
        if "lr_scheduler" in cfg and cfg["lr_scheduler"] is not None:
            cfg["lr_scheduler"] = LRSchedulerConfig(**cfg["lr_scheduler"])
        else:
            cfg["lr_scheduler"] = None
        return TrainingConfig(**cfg)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> ExperimentConfig:
        data = cls._coerce_data_config(cfg.get("data"))
        model = cls._coerce_model_config(cfg.get("model"))
        training = cls._coerce_training_config(cfg.get("training"))
        return cls(data=data, model=model, training=training)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    def to_dict(self) -> Dict[str, Any]:
        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj

        data_dict = self.data.__dict__.copy()
        data_dict["root"] = str(self.data.root)
        training_dict = self.training.__dict__.copy()
        training_dict["checkpoint_dir"] = str(self.training.checkpoint_dir)
        training_dict["log_dir"] = str(self.training.log_dir) if self.training.log_dir else None
        training_dict["lr_scheduler"] = (
            asdict(self.training.lr_scheduler) if self.training.lr_scheduler else None
        )
        model_dict = self.model.__dict__.copy() if self.model else None
        if model_dict:
            model_dict = {k: _serialize(v) for k, v in model_dict.items()}
        return {
            "data": {k: _serialize(v) for k, v in data_dict.items()},
            "model": model_dict,
            "training": {k: _serialize(v) for k, v in training_dict.items()},
        }
