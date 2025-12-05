"""
Test script for AV2GNNForecastingDataset using local per-scenario maps.

Example:
    python -m gnn_trajectory.test_argoverse2 --split train --index 21363 --vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from gnn_trajectory.config import DEFAULT_DATA_ROOT
from gnn_trajectory.data import AV2GNNForecastingDataset


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a specific AV2 scenario sample.")
    p.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to Argoverse 2 motion-forecasting root (defaults to repo data/).",
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Dataset index to inspect (e.g., 21363). If omitted, iterate sequentially.",
    )
    p.add_argument("--num", type=int, default=1, help="Number of samples to inspect when --index is not set.")
    p.add_argument("--vis", action="store_true", help="Visualize the selected scenario.")
    return p.parse_args()


def describe_sample(sample, idx: int):
    print(f"=== Scenario idx {idx} ===")
    print("Scenario ID:", sample["scenario_id"])
    print("agent_hist:", tuple(sample["agent_hist"].shape))
    print("fut_traj:", tuple(sample["fut_traj"].shape))
    print("edge_index_aa:", tuple(sample["edge_index_aa"].shape))
    print("edge_index_al:", tuple(sample["edge_index_al"].shape))
    print("lane_nodes:", tuple(sample["lane_nodes"].shape))
    print("lane_topology:", tuple(sample["lane_topology"].shape))
    print()


def main():
    args = parse_args()

    root = Path(args.root)
    print(f"\nLoading AV2 dataset from: {root}")
    ds = AV2GNNForecastingDataset(root=root, split=args.split)
    print(f"Found {len(ds)} scenarios\n")

    if args.index is not None:
        if args.index < 0 or args.index >= len(ds):
            raise IndexError(f"Index {args.index} out of range (dataset has {len(ds)} samples)")
        sample = ds[args.index]
        describe_sample(sample, args.index)
        if args.vis:
            visualize(sample)
        return

    for i in range(min(args.num, len(ds))):
        sample = ds[i]
        describe_sample(sample, i)
        if args.vis:
            visualize(sample)
            break


def visualize(batch):
    import numpy as np

    agent_hist = batch["agent_hist"].numpy()
    fut_traj = batch["fut_traj"].numpy()
    fut_mask = batch["fut_mask"].numpy()
    lane_nodes = batch["lane_nodes"].numpy()
    agent_types = batch["agent_types"].numpy()

    A = int(batch["num_agents"])
    L = int(batch["num_lanes"])
    agent_hist = agent_hist[:A]
    fut_traj = fut_traj[:A]
    fut_mask = fut_mask[:A]
    agent_types = agent_types[:A]
    lane_nodes = lane_nodes[:L]

    # color map for agent classes
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
    # draw lanes
    for ln in lane_nodes:
        valid = (ln != 0).any(axis=1)
        pts = ln[valid]
        if len(pts) > 1:
            plt.plot(pts[:, 0], pts[:, 1], color="lightgray", lw=0.6, alpha=0.5)

    for i, (ah, fut, mask, cls) in enumerate(zip(agent_hist, fut_traj, fut_mask, agent_types)):
        color = colors.get(int(cls), "black")

        # Filter out padded timesteps and obvious outliers
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


    # highlight focal agent (usually index 0)
    focal_idx = np.argmin(np.linalg.norm(agent_hist[:, -1, :2], axis=1))
    plt.plot(agent_hist[focal_idx, :, 0],
             agent_hist[focal_idx, :, 1],
             "k-", lw=3, label="Focal agent")

    scenario_id = batch["scenario_id"] if isinstance(batch["scenario_id"], str) else batch["scenario_id"][0]
    plt.title(f"Scenario {scenario_id}  (agents colored by class)")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
