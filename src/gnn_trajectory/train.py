"""
Command-line entry point for training runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Callable, Dict

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from gnn_trajectory.config import ExperimentConfig, LRSchedulerConfig, ModelConfig
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


def _build_dataloader(
    cfg: ExperimentConfig,
    split: str,
    shuffle: bool,
    limit: int | None = None,
) -> DataLoader:
    _ensure_split_directory(cfg.data.root, split)
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
    if limit is not None and limit > 0:
        capped = min(limit, len(dataset))
        dataset = Subset(dataset, list(range(capped)))
        print(f"[data] limiting split={split} to first {capped} samples")
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
    total_samples = 0
    loss_sum = 0.0
    metric_sums: Dict[str, float] = {name: 0.0 for name in metric_fns.keys()}
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            preds, _ = model(batch)
            targets, mask = _extract_targets(batch, preds)
            loss = masked_l2_loss(preds, targets, mask)
            batch_size = preds.shape[0]
            total_samples += batch_size
            loss_sum += loss.item() * batch_size
            metric_values = _compute_metrics(metric_fns, preds, targets, mask)
            for name, value in metric_values.items():
                metric_sums[name] += value * batch_size
    if total_samples == 0:
        zero_metrics = {name: 0.0 for name in metric_fns.keys()}
        return 0.0, zero_metrics
    mean_loss = loss_sum / total_samples
    mean_metrics = {name: metric_sum / total_samples for name, metric_sum in metric_sums.items()}
    return mean_loss, mean_metrics


def _prepare_run_outputs(
    cfg: ExperimentConfig, model_cfg: ModelConfig
) -> tuple[SummaryWriter | None, Path, Path, Path]:
    log_root = Path(cfg.training.log_dir).expanduser() if cfg.training.log_dir else None
    ckpt_root = Path(cfg.training.checkpoint_dir).expanduser() if cfg.training.checkpoint_dir else None
    base_dir = log_root or ckpt_root or Path("outputs")
    split_name = cfg.data.split or "train"
    run_name = f"{split_name}_{model_cfg.encoder}-{model_cfg.decoder}"
    run_dir = base_dir / run_name
    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    artifact_dir = run_dir / "artifacts"
    for directory in (tb_dir, ckpt_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    tb_run_dir = tb_dir / run_stamp
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer: SummaryWriter | None = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        print(f"[tensorboard] logging run data to {tb_run_dir}")
    else:  # pragma: no cover - tensorboard optional dependency
        print(f"[tensorboard] SummaryWriter unavailable. Outputs in {run_dir}")
    return writer, run_dir, ckpt_dir, artifact_dir


def _create_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_cfg: LRSchedulerConfig | None
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_cfg is None:
        return None
    if scheduler_cfg.type != "plateau":
        raise ValueError(f"Unsupported scheduler type '{scheduler_cfg.type}'")
    print(
        f"[scheduler] ReduceLROnPlateau factor={scheduler_cfg.factor} "
        f"patience={scheduler_cfg.patience} min_lr={scheduler_cfg.min_lr}"
    )
    kwargs = dict(
        mode="min",
        factor=scheduler_cfg.factor,
        patience=scheduler_cfg.patience,
        threshold=scheduler_cfg.threshold,
        cooldown=scheduler_cfg.cooldown,
        min_lr=scheduler_cfg.min_lr,
    )
    try:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=scheduler_cfg.verbose,
            **kwargs,
        )
    except TypeError:
        if scheduler_cfg.verbose:
            print("[scheduler] ReduceLROnPlateau verbose flag not supported by this torch version.")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


def _ensure_split_directory(root: Path | str, split: str | None) -> None:
    if split is None:
        return
    root_path = Path(root)
    candidate = root_path / split
    if candidate.exists():
        return
    if root_path.exists() and root_path.name == split:
        return
    raise FileNotFoundError(f"Expected split directory '{split}' under '{root_path}', but it was not found.")


def run_experiment(config: ExperimentConfig | None = None) -> None:
    """
    Assemble data loaders, instantiate the model, and launch training.
    """

    cfg = config or ExperimentConfig()
    model_cfg = cfg.materialize_model_config()

    seed_everything(42)

    writer, run_dir, ckpt_dir, artifact_dir = _prepare_run_outputs(cfg, model_cfg)
    cfg.training.checkpoint_dir = ckpt_dir

    cfg_dict = cfg.to_dict()
    print("[config] Experiment hyperparameters:")
    for section, values in cfg_dict.items():
        if values is None:
            continue
        print(f"  {section}:")
        for key, value in values.items():
            print(f"    {key}: {value}")

    with open(artifact_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    with open(artifact_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg_dict, indent=2))

    if writer:
        writer.add_text("config/full", json.dumps(cfg_dict, indent=2))

    train_limit = cfg.training.train_max_scenarios
    train_loader = _build_dataloader(cfg, cfg.data.split, shuffle=True, limit=train_limit)
    val_loader = None
    val_limit = cfg.training.val_max_scenarios
    if cfg.data.val_split:
        val_loader = _build_dataloader(cfg, cfg.data.val_split, shuffle=False, limit=val_limit)
    elif cfg.training.val_every_steps and cfg.training.val_every_steps > 0:
        raise ValueError(
            "Validation was requested (training.val_every_steps set) but data.val_split is missing."
        )

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
    scheduler = _create_scheduler(optimizer, cfg.training.lr_scheduler)

    metrics = {
        "ADE": average_displacement_error,
        "FDE": final_displacement_error,
        "HitRate@2m": make_hit_rate_metric(2.0),
    }
    val_interval = cfg.training.val_every_steps
    use_step_validation = val_interval is not None and val_interval > 0
    global_step = 0
    last_val_step = 0

    current_epoch = 0

    def _run_validation(trigger: str) -> None:
        nonlocal last_val_step
        if val_loader is None or global_step == 0:
            return
        print(f"[val] running validation ({trigger}) at step {global_step}")
        val_loss, val_metrics = _evaluate(
            model,
            val_loader,
            device,
            metrics,
        )
        _log_metrics("val", current_epoch, global_step, val_loss, val_metrics)
        if writer:
            writer.add_scalar("val/loss", val_loss, global_step)
            for name, value in val_metrics.items():
                writer.add_scalar(f"val/{name}", value, global_step)
        if scheduler is not None:
            scheduler.step(val_loss)
        ckpt = {
            "epoch": current_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg_dict,
        }
        ckpt_path = ckpt_dir / f"step_{global_step}_epoch_{current_epoch}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[ckpt] saved checkpoint to {ckpt_path}")
        last_val_step = global_step

    for epoch in range(1, cfg.training.epochs + 1):
        current_epoch = epoch
        print(f"[epoch] starting epoch {epoch}/{cfg.training.epochs}")
        model.train()
        epoch_loss_total = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()
        first_batch_reported = False
        for batch in train_loader:
            if not first_batch_reported:
                print(
                    f"[epoch] first batch ready after {time.time() - epoch_start_time:.2f}s"
                )
                first_batch_reported = True
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
            epoch_loss_total += loss.item()
            epoch_steps += 1

            if global_step % cfg.training.log_every == 0:
                metric_values = _compute_metrics(metrics, preds, targets, mask)
                _log_metrics("train", epoch, global_step, loss.item(), metric_values)
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    for name, value in metric_values.items():
                        writer.add_scalar(f"train/{name}", value, global_step)

            if use_step_validation and val_loader is not None and global_step % val_interval == 0:
                _run_validation("interval")

        if not use_step_validation and val_loader is not None:
            _run_validation("epoch_end")
        elif scheduler is not None and epoch_steps > 0 and val_loader is None:
            avg_train_loss = epoch_loss_total / epoch_steps
            scheduler.step(avg_train_loss)

    if use_step_validation and val_loader is not None and global_step > 0 and last_val_step != global_step:
        _run_validation("final")

    if writer:
        writer.close()

if __name__ == "__main__":
    args = _parse_args()
    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    run_experiment(cfg)
