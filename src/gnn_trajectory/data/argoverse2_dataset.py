# av2_gnn_dataset_localmaps.py (with local coordinate transform)
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap


AV2_DYNAMIC_CLASSES = {
    "vehicle": 0, "pedestrian": 1, "motorcyclist": 2, "cyclist": 3, "bus": 4,
    "static": 5, "background": 6, "construction": 7, "riderless_bycicle": 8, "unknown": 9
}


def _angle_wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def _pad(arr, t, axis=0): return np.pad(arr, [(0, max(0, t - arr.shape[axis]))] + [(0, 0)] * (arr.ndim - 1))
def _radius_graph(x, r):
    N = len(x)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if np.linalg.norm(x[i] - x[j]) < r]
    edges_length = [np.linalg.norm(x[i] - x[j]) for i in range(N) for j in range(i + 1, N) if np.linalg.norm(x[i] - x[j]) < r]
    edges_length = np.array(edges_length, dtype=float)
    if not edges: 
        return np.zeros((2, 0), np.int64)
    e = np.array(edges)
    return np.concatenate([e, e[:, ::-1]]).T, np.concatenate([edges_length, edges_length], axis=0)

def _knn(a, b, k):
    if not len(a) or not len(b): return np.zeros((2, 0), np.int64)
    d = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    idx = np.argsort(d, axis=1)[:, :min(k, len(b))]
    return np.stack([np.repeat(np.arange(len(a)), idx.shape[1]), idx.reshape(-1)], 0)

def _point_to_polyline_distance(P, polyline):
    """
    P: (2,) point
    polyline: (M,2) array of lane centerline points
    Returns shortest Euclidean distance from P to any line segment.
    """
    if len(polyline) < 2:
        return np.linalg.norm(P - polyline[0])

    dmin = float("inf")
    for i in range(len(polyline) - 1):
        A = polyline[i]
        B = polyline[i+1]
        AB = B - A
        AP = P - A
        t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9)
        t = np.clip(t, 0.0, 1.0)
        proj = A + t * AB
        d = np.linalg.norm(P - proj)
        if d < dmin:
            dmin = d
    return dmin


class AV2GNNForecastingDataset(Dataset):
    """Dataset using local maps and converting to the focal agent’s local coordinate frame."""
    def __init__(self, root: str | Path, split="train",
                 obs_sec=7, fut_sec=4, hz=10.0,
                 max_agents=128, max_lanes=256, max_pts=40,
                 agent_radius=50, knn_lanes=3):
        self.root = Path(root)
        self.split = split
        candidate = self.root / split
        if candidate.exists():
            self.split_dir = candidate
        elif self.root.exists():
            # Allow pointing directly at a split directory (e.g., only `train` downloaded).
            self.split_dir = self.root
        else:
            raise FileNotFoundError(f"Could not find dataset directory at {self.root}")
        self.obs_len = int(obs_sec * hz)
        self.pred_len = int(fut_sec * hz)
        self.dt = 1.0 / hz
        self.max_agents, self.max_lanes, self.max_pts = max_agents, max_lanes, max_pts
        self.agent_radius, self.knn_lanes = agent_radius, knn_lanes

        self.scene_dirs = sorted(self.split_dir.iterdir())
        self.scene_dirs = [d for d in self.scene_dirs if d.is_dir()]
        if not self.scene_dirs:
            raise FileNotFoundError(f"No scenario folders found in {split}")

    def __len__(self): return len(self.scene_dirs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scenedir = self.scene_dirs[i]
        parquet = next(scenedir.glob("*.parquet"))
        map_json = next(scenedir.glob("log_map_archive_*.json"))

        scenario = load_argoverse_scenario_parquet(parquet)
        amap = ArgoverseStaticMap.from_json(map_json)

        # ---------- Agents ----------
        agents, futs, fmask, types = [], [], [], []
        focal_track_id = scenario.focal_track_id

        # gather reference (ego) pose
        ref_pos, ref_heading = None, 0.0

        for tr in scenario.tracks:
            states = sorted(tr.object_states, key=lambda s: s.timestep)
            if not states: 
                continue
            if tr.object_type.value == "UNKNOWN":
                continue
            xs, ys = np.array([s.position[0] for s in states]), np.array([s.position[1] for s in states])
            vx, vy = np.array([s.velocity[0] for s in states]), np.array([s.velocity[1] for s in states])
            hd = _angle_wrap(np.array([s.heading for s in states]))
            seq = np.stack([xs, ys, vx, vy, hd], 1)

            if tr.track_id == focal_track_id:
                ref_pos = seq[min(len(seq), self.obs_len) - 1, :2].copy()
                ref_heading = seq[min(len(seq), self.obs_len) - 1, 4].copy()

            obs, fut = seq[:self.obs_len], seq[self.obs_len:self.obs_len+self.pred_len]
            mask = np.zeros(self.pred_len, np.float32); mask[:len(fut)] = 1
            agents.append(_pad(obs, self.obs_len))
            futs.append(_pad(fut[:, :2], self.pred_len))
            fmask.append(mask)
            types.append(AV2_DYNAMIC_CLASSES.get(tr.object_type.value, 9))

        # if no focal found, just use mean position as fallback
        if ref_pos is None:
            ref_pos = np.mean([a[-1, :2] for a in agents], axis=0)
            ref_heading = 0.0

        # ---------- Apply local coordinate transform ----------
        c, s = np.cos(-ref_heading), np.sin(-ref_heading)
        R = np.array([[c, -s], [s, c]])

        def transform_xy(xy): return (xy - ref_pos) @ R.T

        def transform_vel(v): return v @ R.T

        for a in agents:
            a[:, :2] = transform_xy(a[:, :2])
            a[:, 2:4] = transform_vel(a[:, 2:4])
            a[:, 4] = _angle_wrap(a[:, 4] - ref_heading)

        for f in futs:
            f[:, :2] = transform_xy(f[:, :2])

        agent_pos_T_all = np.array([a[-1, :2] for a in agents])
        dists = np.linalg.norm(agent_pos_T_all, axis=1)        # distance from focal (now at origin)
        keep = dists < self.agent_radius                       # within radius in local frame

        agents = [agents[i] for i in range(len(agents)) if keep[i]]
        futs    = [futs[i]   for i in range(len(futs))   if keep[i]]
        fmask   = [fmask[i]  for i in range(len(fmask))  if keep[i]]
        types   = [types[i]  for i in range(len(types))  if keep[i]]

        A = min(len(agents), self.max_agents)
        if A == 0:
            raise ValueError(f"No agents within {self.agent_radius} m in scenario {scenario.scenario_id}")

        agent_hist = np.pad(np.stack(agents[:A]), ((0, self.max_agents - A), (0, 0), (0, 0)))
        agent_pos_T = agent_hist[:, -1, :2]
        edge_aa, edges_length = _radius_graph(agent_pos_T[:A], self.agent_radius)

        # ---------- Map (lanes in local frame) ----------
        lane_ids = amap.get_scenario_lane_segment_ids()[:self.max_lanes]
        lane_nodes, lane_centroids, topo, lane_lengths  = [], [], [], []
        for lid in lane_ids:
            cl = amap.get_lane_segment_centerline(lid)[:, :2]
            cl = transform_xy(cl)  # transform to local frame

            lane_len = len(cl)

            if lane_len > self.max_pts:
                cl = cl[np.linspace(0, lane_len - 1, self.max_pts, dtype=int)]
                lane_len = self.max_pts  # cap to max_pts

            lane_lengths.append(lane_len)
            lane_nodes.append(_pad(cl, self.max_pts))
            lane_centroids.append(cl.mean(0))
            for s in amap.get_lane_segment_successor_ids(lid) or []:
                if s in lane_ids:
                    topo.append((lane_ids.index(lid), lane_ids.index(s)))
        L = len(lane_nodes)
        lane_nodes = np.pad(np.stack(lane_nodes), ((0, self.max_lanes - L), (0, 0), (0, 0)))
        lane_centroids = np.pad(np.stack(lane_centroids), ((0, self.max_lanes - L), (0, 0)))
        edge_al = _knn(agent_pos_T[:A], lane_centroids[:L], self.knn_lanes)

        # edge agent distances
        # ---- Compute true geometric distance agent -> lane polyline ----
        edges_length_al = []

        # lane_nodes shape: (L, max_pts, 2)
        lane_nodes_np = lane_nodes[:L]  # only real lanes

        for ai, li in edge_al.T:  # iterate over edges
            agent_xy = agent_pos_T[ai]
            polyline = lane_nodes_np[li][:lane_lengths[li]]  # truncate padded polyline
            d = _point_to_polyline_distance(agent_xy, polyline)
            edges_length_al.append(d)

        edges_length_al = np.array(edges_length_al, dtype=float)



        lane_topo = np.array(topo, int).T if topo else np.zeros((2, 0), int)

        # ---------- Output ----------
        return {
            "scenario_id": scenario.scenario_id,
            "agent_hist": torch.tensor(agent_hist, dtype=torch.float32),
            "agent_types": torch.tensor(types + [0]*(self.max_agents - len(types)), dtype=torch.int64),
            "agent_pos_T": torch.tensor(agent_pos_T, dtype=torch.float32),
            "edge_index_aa": torch.tensor(edge_aa, dtype=torch.int64),
            "edges_length": torch.tensor(edges_length, dtype=torch.float32),
            "edges_length_al": torch.tensor(edges_length_al, dtype=torch.float32),
            "lane_nodes": torch.tensor(lane_nodes, dtype=torch.float32),
            "lane_topology": torch.tensor(lane_topo, dtype=torch.int64),
            "edge_index_al": torch.tensor(edge_al, dtype=torch.int64),
            "lane_lengths": torch.tensor(lane_lengths, dtype=torch.int64),
            "fut_traj": torch.tensor(np.pad(np.stack(futs[:A]), ((0, self.max_agents - A), (0, 0), (0, 0))), dtype=torch.float32),
            "fut_mask": torch.tensor(np.pad(np.stack(fmask[:A]), ((0, self.max_agents - A), (0, 0))), dtype=torch.float32),
            "num_agents": torch.tensor([A]),
            "num_lanes": torch.tensor([L]),
        }


def _parse_args():
    parser = argparse.ArgumentParser(description="Inspect Argoverse 2 motion-forecasting samples.")
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the Argoverse 2 motion-forecasting directory (the one containing train/val/test).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to read from.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Scenario index to inspect.",
    )
    return parser.parse_args()


def _print_summary(sample: Dict[str, torch.Tensor]) -> None:
    print(f"scenario_id: {sample['scenario_id']}")
    print(f"agent_hist: {tuple(sample['agent_hist'].shape)}")
    print(f"fut_traj: {tuple(sample['fut_traj'].shape)}")
    print(f"fut_mask: {tuple(sample['fut_mask'].shape)}")
    print(f"edge_index_aa: {tuple(sample['edge_index_aa'].shape)}")
    print(f"edge_index_al: {tuple(sample['edge_index_al'].shape)}")
    print(f"lane_nodes: {tuple(sample['lane_nodes'].shape)}")
    print(f"lane_topology: {tuple(sample['lane_topology'].shape)}")


if __name__ == "__main__":
    args = _parse_args()
    dataset = AV2GNNForecastingDataset(root=args.root, split=args.split)
    print(f"Loaded {len(dataset)} scenarios from {args.root}/{args.split}")
    sample = dataset[args.index]
    _print_summary(sample)
