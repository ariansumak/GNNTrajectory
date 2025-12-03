# src/gnn_trajectory/visualize_predictions.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_trajectory.config import ExperimentConfig, DEFAULT_DATA_ROOT
from gnn_trajectory.data import AV2GNNForecastingDataset
from gnn_trajectory.models.motion_estimation_model import (
    MotionForecastModel,
    collate_fn as motion_collate_fn,
)


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Argoverse 2 trajectories with GNN predictions."
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config (same style as configs/cluster_train.json).",
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Override dataset root path (directory containing train/val/test).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to use.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Scenario index to visualize. If omitted, will iterate from 0.",
    )
    p.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of scenarios to visualize. If --index is set, starts there.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda', 'cpu'). "
             "Defaults to training.device from config.",
    )
    return p.parse_args()


# ---------- Model loading ----------

def build_model(cfg: ExperimentConfig, device: torch.device) -> MotionForecastModel:
    model_cfg = cfg.materialize_model_config()

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
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        # Try common keys first; fall back to assuming it's a raw state_dict.
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format not understood (expected a state_dict or dict containing one).")

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[checkpoint] loaded from {ckpt_path}")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")


# ---------- Visualization helpers ----------

def visualize_with_prediction(sample: Dict[str, Any], pred_traj: np.ndarray) -> None:
    """
    sample: single-scenario dict from AV2GNNForecastingDataset.__getitem__
    pred_traj: (T, 2) or (K, T, 2) numpy array with predicted futures for the focal agent
    """
    preds = np.asarray(pred_traj)
    if preds.ndim == 2:
        preds = preds[None, ...]  # normalize to (K=1, T, 2)
    elif preds.ndim != 3 or preds.shape[-1] != 2:
        raise ValueError(f"pred_traj must have shape (T,2) or (K,T,2); got {preds.shape}")

    agent_hist = sample["agent_hist"].numpy()
    fut_traj = sample["fut_traj"].numpy()
    fut_mask = sample["fut_mask"].numpy()
    lane_nodes = sample["lane_nodes"].numpy()
    agent_types = sample["agent_types"].numpy()

    A = int(sample["num_agents"])
    L = int(sample["num_lanes"])
    agent_hist = agent_hist[:A]
    fut_traj = fut_traj[:A]
    fut_mask = fut_mask[:A]
    agent_types = agent_types[:A]
    lane_nodes = lane_nodes[:L]

    # color map for agent classes (same as test_argoverse2.py)
    colors = {
        0: "blue",      # VEHICLE
        1: "orange",    # PEDESTRIAN
        2: "purple",    # MOTORCYCLIST
        3: "green",     # CYCLIST
        4: "red",       # BUS
        5: "gray",      # STATIC
        6: "brown",     # BACKGROUND
        7: "pink",      # CONSTRUCTION
        8: "cyan",      # RIDERLESS_BICYCLE
        9: "white",     # UNKNOWN
    }

    plt.figure(figsize=(8, 8))

    # Lanes
    for ln in lane_nodes:
        valid = (ln != 0).any(axis=1)
        pts = ln[valid]
        if len(pts) > 1:
            plt.plot(pts[:, 0], pts[:, 1], color="lightgray", lw=0.6, alpha=0.5)

    # All agents: history + GT future
    for ah, fut, mask, cls in zip(agent_hist, fut_traj, fut_mask, agent_types):
        color = colors.get(int(cls), "black")

        valid_obs = np.isfinite(ah[:, 0]) & np.isfinite(ah[:, 1])
        valid_obs &= (np.abs(ah[:, 0]) < 200) & (np.abs(ah[:, 1]) < 200)
        obs_pts = ah[valid_obs]
        if len(obs_pts) > 1:
            plt.plot(obs_pts[:, 0], obs_pts[:, 1], ".-", lw=1, color=color, alpha=0.8)

        valid_fut = mask > 0
        fut_pts = fut[valid_fut]
        fut_pts = fut_pts[(np.abs(fut_pts[:, 0]) < 200) & (np.abs(fut_pts[:, 1]) < 200)]
        if len(fut_pts) > 0:
            plt.plot(fut_pts[:, 0], fut_pts[:, 1], "--", lw=1, color=color, alpha=0.5)

    # Focal agent (closest to origin at last observed step)
    focal_idx = np.argmin(np.linalg.norm(agent_hist[:, -1, :2], axis=1))

    plt.plot(
        agent_hist[focal_idx, :, 0],
        agent_hist[focal_idx, :, 1],
        "k-",
        lw=3,
        label="Focal history",
    )

    # Ground-truth future of focal agent
    focal_gt_mask = fut_mask[focal_idx] > 0
    focal_gt = fut_traj[focal_idx][focal_gt_mask]
    if len(focal_gt) > 0:
        plt.plot(
            focal_gt[:, 0],
            focal_gt[:, 1],
            color="orange",
            linestyle="--",
            lw=2,
            label="Focal GT future",
        )

    # Predicted future for focal agent
    for i, pred_pts in enumerate(preds):
        pred_pts = pred_pts[(np.abs(pred_pts[:, 0]) < 200) & (np.abs(pred_pts[:, 1]) < 200)]
        if len(pred_pts) > 0:
            plt.plot(
                pred_pts[:, 0],
                pred_pts[:, 1],
                color="red",
                linestyle="-",
                lw=2,
                label=f"Prediction {i + 1}" if len(preds) > 1 else "Prediction",
            )

    scenario_id = sample["scenario_id"]
    plt.title(f"Scenario {scenario_id} (focal agent GT vs prediction)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.axis("equal")
    plt.show()


# ---------- Main loop ----------

def main():
    args = parse_args()

    # Load config (or defaults)
    if args.config:
        cfg = ExperimentConfig.from_file(args.config)
    else:
        cfg = ExperimentConfig()

    # Override dataset root / split if needed
    cfg.data.root = Path(args.root).expanduser()
    cfg.data.split = args.split

    device_str = args.device or cfg.training.device
    device = torch.device(device_str)
    print(f"[device] using {device}")

    # Dataset
    ds = AV2GNNForecastingDataset(
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
    print(f"[data] split={cfg.data.split} root={cfg.data.root} samples={len(ds)}")

    # Model + checkpoint
    model = build_model(cfg, device)
    load_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    def run_on_index(idx: int):
        sample = ds[idx]
        batch = motion_collate_fn([sample])

        # move tensors to device
        batch_device = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            pred_traj, focal_idx = model(batch_device)  # pred_traj: (B=1, T, 2)

        pred = pred_traj[0].cpu().numpy()
        num_preds = pred.shape[0] if pred.ndim == 3 else 1
        print(f"[sample] idx={idx} scenario_id={sample['scenario_id']} "
              f"pred_traj shape={pred.shape} num_preds={num_preds} focal_idx={int(focal_idx[0])}")
        visualize_with_prediction(sample, pred)

    start_idx = args.index if args.index is not None else 0
    if start_idx < 0 or start_idx >= len(ds):
        raise IndexError(f"Index {start_idx} out of range (dataset has {len(ds)} samples)")

    total = min(args.num, len(ds) - start_idx)
    for i in range(total):
        run_on_index(start_idx + i)


if __name__ == "__main__":
    main()
