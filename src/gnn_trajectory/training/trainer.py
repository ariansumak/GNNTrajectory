"""
Scaffold for the training loop and evaluation routines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    grad_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50
    checkpoint_dir: Path = Path("checkpoints")


class Trainer:
    """
    Minimal trainer handling optimisation, logging, and evaluation hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.config = config or TrainingConfig()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.global_step = 0
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> None:
        device = torch.device(self.config.device)
        self.model.to(device)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            for batch in self.train_loader:
                batch = self._move_batch_to_device(batch, device)
                preds = self.model(batch)
                targets = batch["future"]

                loss = self.loss_fn(preds, targets)
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.optimizer.step()
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    metrics = self._compute_metrics(preds, targets)
                    self._log_metrics(epoch, loss.item(), metrics)

            if self.val_loader is not None:
                self.evaluate(epoch, device)

    def evaluate(self, epoch: int, device: torch.device) -> None:
        self.model.eval()
        metric_accumulators = {name: [] for name in self.metrics}
        with torch.no_grad():
            for batch in self.val_loader or []:
                batch = self._move_batch_to_device(batch, device)
                preds = self.model(batch)
                targets = batch["future"]
                for name, fn in self.metrics.items():
                    metric_accumulators[name].append(fn(preds, targets).item())

        averaged = {name: sum(values) / len(values) for name, values in metric_accumulators.items() if values}
        self._log_validation(epoch, averaged)

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        return {name: fn(predictions, targets).item() for name, fn in self.metrics.items()}

    def _log_metrics(self, epoch: int, loss: float, metrics: Dict[str, float]) -> None:
        print(f"[train] epoch={epoch} step={self.global_step} loss={loss:.4f} metrics={metrics}")

    def _log_validation(self, epoch: int, metrics: Dict[str, float]) -> None:
        print(f"[val] epoch={epoch} metrics={metrics}")

    def save_checkpoint(self, name: str) -> Path:
        checkpoint_path = self.config.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.global_step,
            },
            checkpoint_path,
        )
        return checkpoint_path
