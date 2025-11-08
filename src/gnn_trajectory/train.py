"""
Command-line entry point for training runs.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import AV2GNNForecastingDataset
from .metrics import average_displacement_error, final_displacement_error
from .models import AgentForecastingModel
from .utils import seed_everything


def _build_dataloader(cfg: ExperimentConfig) -> DataLoader:
    dataset = AV2GNNForecastingDataset(
        root=cfg.data.root,
        split=cfg.data.split,
        obs_sec=cfg.data.obs_seconds,
        fut_sec=cfg.data.fut_seconds,
        hz=cfg.data.frequency_hz,
        max_agents=cfg.data.max_agents,
        max_lanes=cfg.data.max_lanes,
        max_pts=cfg.data.max_lane_points,
        agent_radius=cfg.data.agent_radius,
        knn_lanes=cfg.data.lane_knn,
    )
    if cfg.data.batch_size != 1:
        raise ValueError(
            "Only batch_size=1 is currently supported because scenarios have variable edge counts."
        )
    collate_fn = lambda batch: batch[0]
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )


def masked_l2_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    error = (preds - targets) ** 2
    error = error.sum(dim=-1)  # (B, A, T)
    if mask is not None:
        error = error * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(error.numel(), device=error.device, dtype=error.dtype)
    return error.sum() / denom


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _extract_targets(
    batch: Dict[str, torch.Tensor],
    preds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    targets = batch.get("future") or batch.get("fut_traj")
    mask = batch.get("future_mask") or batch.get("fut_mask")
    if targets is None:
        raise KeyError("Batch must contain `future` or `fut_traj` for supervision.")
    return _align_targets(preds, targets, mask)


def _align_targets(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if targets.dim() == preds.dim() - 1:
        targets = targets.unsqueeze(0)
    if mask is not None and mask.dim() == preds.dim() - 2:
        mask = mask.unsqueeze(0)
    return targets, mask


def _compute_metrics(
    metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]],
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> Dict[str, float]:
    return {name: fn(preds, targets, mask).item() for name, fn in metric_fns.items()}


def _log_metrics(epoch: int, step: int, loss: float, metrics: Dict[str, float]) -> None:
    print(f"[train] epoch={epoch} step={step} loss={loss:.4f} metrics={metrics}")


def run_experiment(config: ExperimentConfig | None = None) -> None:
    """
    Assemble data loaders, instantiate the model, and launch training.
    """

    cfg = config or ExperimentConfig()
    model_cfg = cfg.materialize_model_config()

    seed_everything(42)

    train_loader = _build_dataloader(cfg)
    device = torch.device(cfg.training.device)

    model = AgentForecastingModel(
        history_dim=5,
        future_steps=model_cfg.future_steps,
        hidden_dim=model_cfg.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "ADE": average_displacement_error,
        "FDE": final_displacement_error,
    }

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            preds = model(batch)
            targets, mask = _extract_targets(batch, preds)

            loss = masked_l2_loss(preds, targets, mask)
            optimizer.zero_grad()
            loss.backward()
            if cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            global_step += 1

            if global_step % cfg.training.log_every == 0:
                metric_values = _compute_metrics(metrics, preds, targets, mask)
                _log_metrics(epoch, global_step, loss.item(), metric_values)


if __name__ == "__main__":
    run_experiment()
