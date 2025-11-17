"""
Command-line entry point for training runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from gnn_trajectory.config import ExperimentConfig
from gnn_trajectory.data import AV2GNNForecastingDataset
from gnn_trajectory.losses import masked_l2_loss
from gnn_trajectory.metrics import (
    average_displacement_error,
    final_displacement_error,
    make_hit_rate_metric,
)
from gnn_trajectory.models.motion_estimation_model import (
    MotionForecastModel,
    collate_fn as motion_collate_fn,
)
from gnn_trajectory.utils import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNNTrajectory models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON file containing data/model/training hyperparameters.",
    )
    return parser.parse_args()


def _build_dataloader(cfg: ExperimentConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = AV2GNNForecastingDataset(
        root=cfg.data.root,
        split=split,
        obs_sec=cfg.data.obs_seconds,
        fut_sec=cfg.data.fut_seconds,
        hz=cfg.data.frequency_hz,
        max_agents=cfg.data.max_agents,
        max_lanes=cfg.data.max_lanes,
        max_pts=cfg.data.max_lane_points,
        agent_radius=cfg.data.agent_radius,
        knn_lanes=cfg.data.lane_knn,
    )
    print(f"[data] split={split} root={cfg.data.root} samples={len(dataset)} shuffle={shuffle}")
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=shuffle,
        collate_fn=motion_collate_fn,
    )


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
    targets = _select_focal(batch, targets)
    mask = _select_focal(batch, mask) if mask is not None else None
    return _align_targets(preds, targets, mask)


def _select_focal(batch: Dict[str, torch.Tensor], tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    focal_indices = batch.get("focal_indices")
    if focal_indices is None:
        return tensor
    if tensor.size(0) == focal_indices.shape[0]:
        return tensor
    return tensor[focal_indices]


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


def _log_metrics(stage: str, epoch: int, step: int, loss: float, metrics: Dict[str, float]) -> None:
    print(f"[{stage}] epoch={epoch} step={step} loss={loss:.4f} metrics={metrics}")


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_fns: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]],
) -> tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    aggregated: Dict[str, list[float]] = {name: [] for name in metric_fns.keys()}
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            preds, _ = model(batch)
            targets, mask = _extract_targets(batch, preds)
            loss = masked_l2_loss(preds, targets, mask)
            losses.append(loss.item())
            metric_values = _compute_metrics(metric_fns, preds, targets, mask)
            for name, value in metric_values.items():
                aggregated[name].append(value)
    mean_loss = float(torch.tensor(losses).mean()) if losses else 0.0
    mean_metrics = {name: (sum(values) / len(values) if values else 0.0) for name, values in aggregated.items()}
    return mean_loss, mean_metrics


def _maybe_create_writer(cfg: ExperimentConfig) -> SummaryWriter | None:
    log_dir = cfg.training.log_dir
    if log_dir is None or SummaryWriter is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def run_experiment(config: ExperimentConfig | None = None) -> None:
    """
    Assemble data loaders, instantiate the model, and launch training.
    """

    cfg = config or ExperimentConfig()
    model_cfg = cfg.materialize_model_config()

    seed_everything(42)

    train_loader = _build_dataloader(cfg, cfg.data.split, shuffle=True)
    val_loader = None
    if cfg.data.val_split:
        val_loader = _build_dataloader(cfg, cfg.data.val_split, shuffle=False)

    device = torch.device(cfg.training.device)

    encoder_cfg = dict(model_cfg.encoder_kwargs)
    decoder_cfg = dict(model_cfg.decoder_kwargs)
    decoder_cfg.setdefault("pred_len", model_cfg.future_steps)
    model = MotionForecastModel(
        encoder_name=model_cfg.encoder,
        decoder_name=model_cfg.decoder,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        future_steps=model_cfg.future_steps,
    ).to(device)
    print(
        "[model] encoder=%s decoder=%s device=%s"
        % (model_cfg.encoder, model_cfg.decoder, cfg.training.device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "ADE": average_displacement_error,
        "FDE": final_displacement_error,
        "HitRate@2m": make_hit_rate_metric(2.0),
    }
    writer = _maybe_create_writer(cfg)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        print(f"[epoch] starting epoch {epoch}/{cfg.training.epochs}")
        model.train()
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            preds, _ = model(batch)
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
                _log_metrics("train", epoch, global_step, loss.item(), metric_values)
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    for name, value in metric_values.items():
                        writer.add_scalar(f"train/{name}", value, global_step)

        if val_loader is not None:
            print(f"[epoch] running validation after epoch {epoch}")
            val_loss, val_metrics = _evaluate(model, val_loader, device, metrics)
            _log_metrics("val", epoch, global_step, val_loss, val_metrics)
            if writer:
                writer.add_scalar("val/loss", val_loss, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"val/{name}", value, epoch)

    if writer:
        writer.close()

if __name__ == "__main__":
    args = _parse_args()
    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    run_experiment(cfg)
