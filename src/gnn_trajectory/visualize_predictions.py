# src/gnn_trajectory/visualize_predictions.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import matplotlib
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
        nargs="+",
        required=True,
        help="One or more trained model checkpoints (.pt). Pass multiple to overlay predictions.",
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
        "--separate",
        action="store_true",
        help="Plot each checkpoint in its own row to avoid overlapping trajectories.",
    )
    p.add_argument(
        "--focal-only",
        action="store_true",
        help="Only draw the focal agent (hide other agents) for a cleaner view.",
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
        for key in ["state_dict", "model_state_dict", "model", "model_state"]:
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


def load_config_from_checkpoint(ckpt_path: Path) -> ExperimentConfig | None:
    """
    Attempt to read ExperimentConfig (as dict) stored inside a training checkpoint.
    Returns None if not present or on failure.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint] could not read {ckpt_path} to get config: {exc}")
        return None

    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        try:
            return ExperimentConfig.from_dict(ckpt["config"])
        except Exception as exc:
            print(f"[checkpoint] config in {ckpt_path} could not be parsed: {exc}")
            return None
    return None


# ---------- Visualization helpers ----------

def visualize_with_prediction(
    sample: Dict[str, Any],
    pred_traj: np.ndarray | Dict[str, np.ndarray],
    separate: bool = False,
    focal_only: bool = False,
    hz: float | None = None,
) -> None:
    """
    sample: single-scenario dict from AV2GNNForecastingDataset.__getitem__
    pred_traj: (T, 2) or (K, T, 2) numpy array with predicted futures for the focal agent,
               or a dict mapping labels -> such arrays (to overlay multiple checkpoints)
    separate: if True, plot one row per checkpoint to avoid overlap
    focal_only: if True, hide all non-focal agents
    """
    if isinstance(pred_traj, dict):
        pred_items = [(label, np.asarray(arr)) for label, arr in pred_traj.items()]
    else:
        pred_items = [("Prediction", np.asarray(pred_traj))]

    normalized_preds = []
    for label, arr in pred_items:
        if arr.ndim == 2:
            arr = arr[None, ...]  # normalize to (K=1, T, 2)
        elif arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(f"pred_traj must have shape (T,2) or (K,T,2); got {arr.shape}")
        normalized_preds.append((label, arr))

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

    # Focal agent (closest to origin at last observed step)
    focal_idx = np.argmin(np.linalg.norm(agent_hist[:, -1, :2], axis=1))
    focal_hist = agent_hist[focal_idx]
    focal_gt_mask = fut_mask[focal_idx] > 0
    focal_gt = fut_traj[focal_idx][focal_gt_mask]

    # Axis limits: zoom into last 5 seconds around focal agent
    window_steps = int(round(5 * hz)) if hz else None

    def _xy(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        return arr[..., :2]

    def _window_tail(arr: np.ndarray) -> np.ndarray:
        if window_steps is None:
            return _xy(arr)
        return _xy(arr[-window_steps:])

    def _window_head(arr: np.ndarray) -> np.ndarray:
        if window_steps is None:
            return _xy(arr)
        return _xy(arr[:window_steps])

    def _compute_axis_limits(preds: list[tuple[str, np.ndarray]]) -> tuple[float, float, float, float] | None:
        pts_list = []
        fh = _window_tail(focal_hist)
        pts_list.append(fh)
        if len(focal_gt) > 0:
            pts_list.append(_xy(focal_gt))  # full GT
        for _, arr in preds:
            for pred_pts in arr:
                pts_list.append(_xy(pred_pts))  # full predictions
        pts = np.concatenate([p for p in pts_list if len(p) > 0], axis=0) if pts_list else None
        if pts is None or len(pts) == 0:
            return None
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) == 0:
            return None
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        dx, dy = xmax - xmin, ymax - ymin
        pad = max(dx, dy) * 0.15 + 2.0
        return xmin - pad, xmax + pad, ymin - pad, ymax + pad

    axis_limits = _compute_axis_limits(normalized_preds)

    def _draw_scene(ax, label: str, pred_arr: np.ndarray, color, linestyle="-", marker="o"):
        ax.set_facecolor("#f9f9f9")

        # Lanes
        for ln in lane_nodes:
            valid = (ln != 0).any(axis=1)
            pts = ln[valid]
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], color="#d9d9d9", lw=0.8, alpha=0.6)

        # Other agents (optional)
        if not focal_only:
            for ah, fut, mask, cls in zip(agent_hist, fut_traj, fut_mask, agent_types):
                if np.allclose(ah, focal_hist):
                    continue
                c = colors.get(int(cls), "black")
                valid_obs = np.isfinite(ah[:, 0]) & np.isfinite(ah[:, 1])
                valid_obs &= (np.abs(ah[:, 0]) < 200) & (np.abs(ah[:, 1]) < 200)
                obs_pts = ah[valid_obs]
                if len(obs_pts) > 1:
                    ax.plot(obs_pts[:, 0], obs_pts[:, 1], ".-", lw=1, color=c, alpha=0.2)

        # Focal history
        ax.plot(
            focal_hist[:, 0],
            focal_hist[:, 1],
            "k-",
            lw=3,
            label="Focal history",
        )

        # Focal GT
        if len(focal_gt) > 0:
            ax.plot(
                focal_gt[:, 0],
                focal_gt[:, 1],
                color="#c62828",  # red
                linestyle="--",
                lw=2,
                label="Focal GT future",
            )

        # Predictions
        for k, pred_pts in enumerate(pred_arr):
            pred_pts = pred_pts[(np.abs(pred_pts[:, 0]) < 200) & (np.abs(pred_pts[:, 1]) < 200)]
            if len(pred_pts) > 0:
                ax.plot(
                    pred_pts[:, 0],
                    pred_pts[:, 1],
                    color=color,
                    linestyle=linestyle,
                    lw=2.2,
                    marker=marker,
                    markevery=4,
                    label=f"{label} ({k + 1})" if pred_arr.shape[0] > 1 else label,
                )
                ax.scatter(
                    pred_pts[-1, 0],
                    pred_pts[-1, 1],
                    color=color,
                    edgecolors="black",
                    s=36,
                    zorder=5,
                )

        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])

    palette = [
        "#1e88e5",  # blue
        "#00838f",  # teal
        "#3949ab",  # indigo
        "#00acc1",  # cyan
        "#5e35b1",  # purple
        "#1976d2",  # steel blue
    ]
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "d", "^"]

    if separate and len(normalized_preds) > 1:
        rows = (len(normalized_preds) + 1) // 2
        cols = 2
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(12, 2.5 * rows),
            sharex=False,
            sharey=False,
            gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        )
        axes = np.array(axes).reshape(rows, cols)
        for idx, ax in enumerate(axes.flat):
            if idx >= len(normalized_preds):
                ax.axis("off")
                continue
            label, pred_arr = normalized_preds[idx]
            color = palette[idx % len(palette)]
            linestyle = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            _draw_scene(ax, label, pred_arr, color=color, linestyle=linestyle, marker=marker)
            ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        for pred_group_idx, (label, pred_arr) in enumerate(normalized_preds):
            color = palette[pred_group_idx % len(palette)]
            linestyle = line_styles[pred_group_idx % len(line_styles)]
            marker = markers[pred_group_idx % len(markers)]
            _draw_scene(ax, label, pred_arr, color=color, linestyle=linestyle, marker=marker)

        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()


# ---------- Main loop ----------

def main():
    args = parse_args()

    # Load base config (CLI file or defaults)
    ckpt_paths = [Path(p) for p in args.checkpoint]
    if args.config:
        cfg = ExperimentConfig.from_file(args.config)
        cfg_source = f"file {args.config}"
    else:
        cfg = ExperimentConfig()
        cfg_source = "default ExperimentConfig"

    # Override dataset root / split
    cfg.data.root = Path(args.root).expanduser()
    cfg.data.split = args.split
    print(f"[config] base {cfg_source} (root={cfg.data.root}, split={cfg.data.split})")

    device_str = args.device or cfg.training.device
    device = torch.device(device_str)
    print(f"[device] using {device}")

    # Load configs from checkpoints (for model construction), track data horizons
    ckpt_cfgs = []
    max_fut_sec = cfg.data.fut_seconds
    for ckpt_path in ckpt_paths:
        ckpt_cfg = load_config_from_checkpoint(ckpt_path)
        ckpt_cfgs.append(ckpt_cfg)
        if ckpt_cfg is not None:
            max_fut_sec = max(max_fut_sec, ckpt_cfg.data.fut_seconds)

    # Use base/user obs_seconds; extend fut_seconds to the longest across ckpts
    cfg.data.fut_seconds = max_fut_sec
    cfg.materialize_model_config()  # refresh history/future steps on cfg (base)

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
    print(f"[data] split={cfg.data.split} root={cfg.data.root} obs_sec={cfg.data.obs_seconds} fut_sec={cfg.data.fut_seconds} samples={len(ds)}")

    # Models + checkpoints (each can carry its own config)
    models = []
    for ckpt_path, ckpt_cfg in zip(ckpt_paths, ckpt_cfgs):
        if ckpt_cfg is None:
            ckpt_cfg = ExperimentConfig.from_dict(cfg.to_dict())
            cfg_note = "(using base config)"
        else:
            cfg_note = "(config from checkpoint)"
        # warn if data shapes differ from dataset config (obs/fut/hz etc.)
        mismatches = []
        for field in ["obs_seconds", "fut_seconds", "frequency_hz", "max_agents", "max_lanes", "max_lane_points"]:
            base_val = getattr(cfg.data, field)
            ckpt_val = getattr(ckpt_cfg.data, field)
            if base_val != ckpt_val:
                mismatches.append(f"{field}: dataset={base_val} ckpt={ckpt_val}")
        if mismatches:
            print(f"[warn] data config mismatch for {ckpt_path.name} vs dataset: " + "; ".join(mismatches))

        model_cfg = ckpt_cfg.materialize_model_config()
        model = build_model(ckpt_cfg, device)
        load_checkpoint(model, ckpt_path, device)
        model.eval()
        models.append((ckpt_path.stem, model))
        print(f"[model] {ckpt_path.name} {cfg_note} encoder={model_cfg.encoder} decoder={model_cfg.decoder}")

    def run_on_index(idx: int):
        sample = ds[idx]
        batch = motion_collate_fn([sample])

        # move tensors to device
        batch_device = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        preds_by_label: Dict[str, np.ndarray] = {}

        for label, model in models:
            with torch.no_grad():
                pred_traj, focal_idx = model(batch_device)  # pred_traj: (B=1, T, 2)

            pred = pred_traj[0].cpu().numpy()
            num_preds = pred.shape[0] if pred.ndim == 3 else 1
            print(
                f"[sample] idx={idx} scenario_id={sample['scenario_id']} "
                f"ckpt={label} pred_traj shape={pred.shape} num_preds={num_preds} focal_idx={int(focal_idx[0])}"
            )
            preds_by_label[label] = pred

        visualize_with_prediction(
            sample,
            preds_by_label,
            separate=args.separate,
            focal_only=args.focal_only,
            hz=cfg.data.frequency_hz,
        )

    start_idx = args.index if args.index is not None else 0
    if start_idx < 0 or start_idx >= len(ds):
        raise IndexError(f"Index {start_idx} out of range (dataset has {len(ds)} samples)")

    total = min(args.num, len(ds) - start_idx)
    for i in range(total):
        run_on_index(start_idx + i)


if __name__ == "__main__":
    main()
