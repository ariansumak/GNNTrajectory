"""
Test script for AV2GNNForecastingDataset using local per-scenario maps.
Run:
    python test_av2_localmaps.py --root /path/to/av2 --split train --vis
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.argoverse2_dataset import AV2GNNForecastingDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to Argoverse2 root directory")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--num", type=int, default=1, help="Number of samples to inspect")
    p.add_argument("--vis", action="store_true", help="Visualize one scenario")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading AV2 dataset from: {args.root}/motion-forecasting/{args.split}")
    ds = AV2GNNForecastingDataset(root=args.root, split=args.split)
    print(f"Found {len(ds)} scenarios\n")

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(dl):
        print(f"=== Scenario {i+1} ===")
        print("Scenario ID:", batch["scenario_id"])
        print("agent_hist:", tuple(batch["agent_hist"].shape))
        print("fut_traj:", tuple(batch["fut_traj"].shape))
        print("edge_index_aa:", tuple(batch["edge_index_aa"].shape))
        print("edge_index_al:", tuple(batch["edge_index_al"].shape))
        print("lane_nodes:", tuple(batch["lane_nodes"].shape))
        print("lane_topology:", tuple(batch["lane_topology"].shape))
        print()

        if args.vis:
            visualize(batch)
            break

        if i + 1 >= args.num:
            break
def visualize(batch):
    import numpy as np

    agent_hist = batch["agent_hist"][0].numpy()      # [A, To, 5]
    fut_traj = batch["fut_traj"][0].numpy()          # [A, Tf, 2]
    fut_mask = batch["fut_mask"][0].numpy()          # [A, Tf]
    lane_nodes = batch["lane_nodes"][0].numpy()      # [L, P, 2]
    agent_types = batch["agent_types"][0].numpy()    # [A]
    focal_id = batch.get("focal_track_id", [None])[0]

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
    focal_idx = 0
    plt.plot(agent_hist[focal_idx, :, 0],
             agent_hist[focal_idx, :, 1],
             "k-", lw=3, label="Focal agent")

    plt.title(f"Scenario {batch['scenario_id'][0]}  (agents colored by class)")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
