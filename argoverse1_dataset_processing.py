
"""
Argoverse v1 Motion Forecasting → LSTM-ready per-agent sequences (+ optional lane context)
-----------------------------------------------------------------------------------------

Outputs per SCENE (CSV):
  • x_seqs:   [N_agents, T_obs, F_in]  — features per agent per timestep (agent-centric)
  • mask:     [N_agents, T_obs]        — 1 where that agent has a valid observation at timestep t
  • roles:    [N_agents, 3]            — one-hot for [AGENT, AV, OTHERS]
  • agent_idx: int                     — index of the AGENT in the N_agents dimension
  • meta: dict with timestamps, city, and SE(2) transform to map back to world

Designed to be fed directly into an LSTM encoder (batch_first=True). A collate_fn is provided to batch scenes.

Per-timestep feature layout (F_in), in AGENT-centric normalized frame:
  0:  x_t          (meters; origin at AGENT last observed; x-axis aligned to AGENT heading)
  1:  y_t
  2:  dx_t         (meters; displacement from t-1 → t)
  3:  dy_t
  4:  speed_t      (m/s)
  5:  sin(yaw_t)   (yaw from displacement direction)
  6:  cos(yaw_t)
  7:  yaw_rate_t   (Δyaw / Δt)
  8:  t_rel        (seconds from obs start)
  9:  sin(0.1·t_rel)
 10:  cos(0.1·t_rel)
 11:  is_AGENT
 12:  is_AV
 13:  is_OTHERS
 Optional lane-relative context (if map is available within radius):
 14:  lat_offset          (signed lateral distance to nearest lane centerline, meters)
 15:  head_diff_sin       (sin of heading difference actor vs lane tangent)
 16:  head_diff_cos       (cos of heading difference)
 17:  has_left_neighbor   (0/1)
 18:  has_right_neighbor  (0/1)
 19:  is_intersection     (0/1)
 20:  turn_left           (0/1)
 21:  turn_right          (0/1)
 22:  turn_none           (0/1)

Dependencies: pandas, numpy, torch, scipy (KDTree), argoverse>=1.x (for maps)
"""
from __future__ import annotations
import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

try:
    from argoverse.map_representation.map_api import ArgoverseMap
except Exception:
    ArgoverseMap = None

# ----------------- SE(2) helpers -----------------

def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

@dataclass
class SE2:
    R: np.ndarray  # 2x2 rotation (world→agent frame)
    t: np.ndarray  # 2,   translation (world→agent frame)

    @staticmethod
    def from_pose(origin_xy: np.ndarray, heading: float) -> "SE2":
        R = _rot(-heading).astype(np.float32)
        t = (-R @ origin_xy.reshape(2,1)).reshape(2).astype(np.float32)
        return SE2(R, t)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return (pts @ self.R.T + self.t).astype(np.float32)

# -------------- map utilities --------------------
def densify_polyline(xy: np.ndarray, step: float = 1.0) -> np.ndarray:
    if len(xy) < 2:
        return xy.copy()
    out = [xy[0]]
    for i in range(1, len(xy)):
        p0, p1 = xy[i-1], xy[i]
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        n = max(1, int(math.floor(L / step)))
        for k in range(1, n+1):
            out.append(p0 + seg * (k / n))
    return np.stack(out, axis=0)

@dataclass
class LaneCloud:
    pts: np.ndarray         # [M,2] lane centerline points (densified), in WORLD frame
    tangents: np.ndarray    # [M,2] unit vector along centerline at each point (approx)
    attrs: np.ndarray       # [M, 6] (has_left, has_right, is_intersection, turn_left, turn_right, turn_none)
    tree: Optional[KDTree]  # KDTree over pts

def build_lane_cloud(amap, city, origin_world_xy, search_radius_m=80.0, densify_step=1.0):
    lane_ids = amap.get_lane_ids_in_xy_bbox(origin_world_xy[0], origin_world_xy[1], city, search_radius_m)
    all_pts, all_tan, all_attr = [], [], []
    for lid in lane_ids:
        ctr_full = np.asarray(amap.get_lane_segment_centerline(lid, city))
        if ctr_full.ndim != 2 or ctr_full.shape[0] < 2:
            continue
        # *** force 2D XY ***
        ctr_xy = ctr_full[:, :2].astype(np.float32)

        # densify in 2D
        ctr_d = densify_polyline(ctr_xy, step=densify_step)
        if ctr_d.shape[0] < 2:
            continue

        # Tangent via finite diff in 2D
        v = np.zeros_like(ctr_d)
        v[1:-1] = ctr_d[2:] - ctr_d[:-2]
        v[0] = ctr_d[1] - ctr_d[0]
        v[-1] = ctr_d[-1] - ctr_d[-2]
        nrm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        tan = v / nrm

        info = amap.city_lane_centerlines_dict[city][lid]

        # robust neighbor/flags (works on older API builds)
        def _has_neighbor(flag_name, id_name):
            vflag = getattr(info, flag_name, None)
            if vflag is not None:
                return bool(vflag)
            nid = getattr(info, id_name, None)
            if isinstance(nid, (list, tuple)):
                return len(nid) > 0 and nid[0] is not None
            return nid is not None and nid != "" and nid != -1

        has_left  = _has_neighbor("has_left_neighbor",  "left_neighbor_id")
        has_right = _has_neighbor("has_right_neighbor", "right_neighbor_id")

        is_intersection = bool(getattr(info, "is_intersection", False))
        turn_dir = str(getattr(info, "turn_direction", "NONE")).upper()
        turn_left  = 1.0 if turn_dir == "LEFT"  else 0.0
        turn_right = 1.0 if turn_dir == "RIGHT" else 0.0
        turn_none  = 1.0 if turn_dir == "NONE"  else 0.0

        attr = np.array([
            float(has_left),
            float(has_right),
            float(is_intersection),
            turn_left,
            turn_right,
            turn_none,
        ], dtype=np.float32)

        all_pts.append(ctr_d)                         # 2D
        all_tan.append(tan)                           # 2D
        all_attr.append(np.repeat(attr[None, :], ctr_d.shape[0], axis=0))

    if len(all_pts) == 0:
        return LaneCloud(np.zeros((0,2), np.float32), np.zeros((0,2), np.float32), np.zeros((0,6), np.float32), None)

    pts = np.concatenate(all_pts, axis=0)            # shape (M, 2)
    tangents = np.concatenate(all_tan, axis=0)       # shape (M, 2)
    attrs = np.concatenate(all_attr, axis=0)         # shape (M, 6)
    tree = KDTree(pts) if KDTree is not None and pts.shape[0] > 0 else None
    return LaneCloud(pts, tangents, attrs, tree)

# -------------- core dataset ---------------------

ROLE2OH = {"AGENT": np.array([1,0,0], np.float32),
           "AV":    np.array([0,1,0], np.float32),
           "OTHERS":np.array([0,0,1], np.float32)}

class ArgoverseLSTMDataset(Dataset):
    def __init__(self,
                 root_csv_dir: str,
                 split: str = "train",
                 map_dir: Optional[str] = None,
                 use_map: bool = True,
                 obs_horizon: int = 20,   # 2s @ 10Hz
                 lane_search_radius: float = 80.0,
                 lane_densify_step: float = 1.0,
                 lane_attach_radius: float = 6.0):
        """
        root_csv_dir: folder that contains split subfolders (train/val/test) with scenario CSVs
        map_dir: path to Argoverse map assets (required if use_map)
        """
        super().__init__()
        self.split = split
        self.use_map = use_map and (ArgoverseMap is not None)
        self.obs_horizon = obs_horizon
        self.lane_attach_radius = lane_attach_radius
        self.lane_search_radius = lane_search_radius
        self.lane_densify_step = lane_densify_step

        split_dir = os.path.join(root_csv_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split dir not found: {split_dir}")
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.csv")))
        if len(self.files) == 0:
            raise ValueError(f"No CSV files found under {split_dir}")

        self.amap = ArgoverseMap() if self.use_map else None

    def __len__(self):
        return len(self.files)

    def _load_scene_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        need = ["TIMESTAMP", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "CITY_NAME"]
        miss = [n for n in need if n not in df.columns]
        if miss:
            raise ValueError(f"Missing columns {miss} in {path}")
        df = df.astype({"TIMESTAMP": float, "TRACK_ID": str, "OBJECT_TYPE": str, "X": float, "Y": float, "CITY_NAME": str})
        df = df.sort_values(["TIMESTAMP", "TRACK_ID"]).reset_index(drop=True)
        return df

    @staticmethod
    def _infer_obs_ts(df: pd.DataFrame, obs_horizon: int) -> np.ndarray:
        ts = np.sort(df["TIMESTAMP"].unique())
        if ts.shape[0] < 1:
            raise ValueError("Empty scene")
        return ts[:min(obs_horizon, ts.shape[0])]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        df = self._load_scene_df(path)
        city = df["CITY_NAME"].iloc[0]
        ts_grid = self._infer_obs_ts(df, self.obs_horizon)

        # Build AGENT-centric SE(2)
        df_agent = df[df["OBJECT_TYPE"] == "AGENT"]
        if df_agent.empty:
            raise ValueError(f"No AGENT in scene {path}")
        df_agent = df_agent[df_agent["TIMESTAMP"].isin(ts_grid)].sort_values("TIMESTAMP")
        if df_agent.shape[0] == 0:
            raise ValueError(f"AGENT has no observations in first {len(ts_grid)} frames: {path}")
        agent_xy = df_agent[["X", "Y"]].to_numpy(dtype=np.float32)
        if agent_xy.shape[0] >= 2:
            v = agent_xy[-1] - agent_xy[-2]
            if np.linalg.norm(v) < 1e-6 and agent_xy.shape[0] >= 3:
                v = agent_xy[-1] - agent_xy[-3]
            heading = float(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0
        else:
            heading = 0.0
        se2 = SE2.from_pose(agent_xy[-1], heading)

        # Optional: lane cloud around agent
        lane_cloud = None
        if self.amap is not None:
            agent_origin_world = se2.R.T @ (-se2.t)  # inverse transform → world
            lane_cloud = build_lane_cloud(self.amap, city, agent_origin_world,
                                          search_radius_m=self.lane_search_radius,
                                          densify_step=self.lane_densify_step)

        # Construct time grid index
        ts_to_idx = {t: i for i, t in enumerate(ts_grid)}
        T = len(ts_grid)

        # Collect per-track sequences
        tracks = []
        roles = []
        masks = []

        for tid, g in df.groupby("TRACK_ID"):
            role = g["OBJECT_TYPE"].iloc[0]
            oh = ROLE2OH.get(role, np.array([0,0,1], np.float32))
            # Align to time grid
            g = g[g["TIMESTAMP"].isin(ts_grid)].sort_values("TIMESTAMP")
            if g.shape[0] == 0:
                continue
            xy_world = g[["X", "Y"]].to_numpy(dtype=np.float32)
            t_idx = np.array([ts_to_idx[t] for t in g["TIMESTAMP"].to_numpy()], dtype=np.int64)
            # Normalize positions to agent frame
            xy = se2.apply(xy_world)
            # Prepare arrays for full grid
            x_full = np.full((T, 2), np.nan, dtype=np.float32)
            x_full[t_idx] = xy
            mask = ~np.isnan(x_full[:, 0])

            # Kinematics
            dx = np.zeros((T, 2), np.float32)
            speed = np.zeros((T,), np.float32)
            yaw_sin = np.zeros((T,), np.float32)
            yaw_cos = np.ones((T,), np.float32)
            yaw_rate = np.zeros((T,), np.float32)
            prev_xy = None
            prev_yaw = None
            prev_t = None
            for t in range(T):
                if not mask[t]:
                    continue
                cur_xy = x_full[t]
                if prev_xy is not None:
                    dt = float(ts_grid[t] - prev_t)
                    dxy = cur_xy - prev_xy
                    dx[t] = dxy
                    sp = float(np.linalg.norm(dxy) / max(dt, 1e-3))
                    speed[t] = sp
                    yaw = math.atan2(dxy[1], dxy[0]) if np.linalg.norm(dxy) > 1e-6 else (prev_yaw if prev_yaw is not None else 0.0)
                    yaw_sin[t] = math.sin(yaw)
                    yaw_cos[t] = math.cos(yaw)
                    if prev_yaw is not None:
                        dyaw = (yaw - prev_yaw + math.pi) % (2*math.pi) - math.pi
                        yaw_rate[t] = dyaw / max(dt, 1e-3)
                    prev_yaw = yaw
                prev_xy = cur_xy
                prev_t = ts_grid[t]

            # Time encodings
            t_rel = (ts_grid - ts_grid[0]).astype(np.float32)
            t_sin = np.sin(0.1 * t_rel)
            t_cos = np.cos(0.1 * t_rel)
            role_tile = np.repeat(oh[None, :], T, axis=0)

            # Lane-relative features per timestep
            # shape (T, 9): [lat, sin dθ, cos dθ] + 6 lane flags
            lane_feats = np.zeros((T, 9), np.float32)
            if lane_cloud is not None and lane_cloud.tree is not None and lane_cloud.pts.shape[0] > 0:
                for t in range(T):
                    if not mask[t]:
                        continue
                    # Back to world coords for KD-tree
                    xy_w = (se2.R.T @ (x_full[t] - se2.t))
                    dists, idxs = lane_cloud.tree.query(xy_w, k=1, distance_upper_bound=self.lane_attach_radius)
                    if np.isinf(dists):
                        continue
                    j = int(idxs)
                    p_lane_w = lane_cloud.pts[j]
                    tan_w = lane_cloud.tangents[j]
                    attr = lane_cloud.attrs[j]  # [6]
                    # Signed lateral offset (left-hand normal)
                    n_w = np.array([-tan_w[1], tan_w[0]], dtype=np.float32)
                    vec_w = xy_w - p_lane_w
                    lat = float(np.dot(vec_w, n_w))
                    # Heading difference actor vs lane tangent
                    ys, yc = yaw_sin[t], yaw_cos[t]
                    yaw = math.atan2(ys, yc) if (ys != 0 or yc != 1) else 0.0
                    tan_a = tan_w @ se2.R.T
                    tan_a = tan_a / (np.linalg.norm(tan_a) + 1e-9)
                    lane_yaw = math.atan2(tan_a[1], tan_a[0])
                    d = (yaw - lane_yaw + math.pi) % (2*math.pi) - math.pi
                    lane_feats[t, 0] = lat
                    lane_feats[t, 1] = math.sin(d)
                    lane_feats[t, 2] = math.cos(d)
                    lane_feats[t, 3:9] = attr  # 6 flags

            # Assemble feature vector per t
            x_t = np.concatenate([
                x_full,            # (T,2)  pos
                dx,                # (T,2)  delta
                speed[:,None],     # (T,1)
                yaw_sin[:,None],   # (T,1)
                yaw_cos[:,None],   # (T,1)
                yaw_rate[:,None],  # (T,1)
                t_rel[:,None],     # (T,1)
                t_sin[:,None],     # (T,1)
                t_cos[:,None],     # (T,1)
                role_tile,         # (T,3)
                lane_feats         # (T,9)
            ], axis=1)
            x_t = np.nan_to_num(x_t, nan=0.0)

            tracks.append(x_t)
            roles.append(oh)
            masks.append(mask.astype(np.bool_))

        if len(tracks) == 0:
            raise ValueError(f"No tracks found for scene {path}")

        x_seqs = torch.from_numpy(np.stack(tracks, axis=0))  # [N, T, F]
        mask = torch.from_numpy(np.stack(masks, axis=0))     # [N, T]
        roles = torch.from_numpy(np.stack(roles, axis=0))    # [N, 3]
        # Find AGENT index
        agent_idx = None
        for i, r in enumerate(roles.numpy()):
            if r[0] == 1: # is_AGENT
                agent_idx = i
                break
        if agent_idx is None:
            agent_idx = int(0)

        meta = {
            "scene_path": path,
            "city": city,
            "timestamps": ts_grid.astype(np.float32),
            "se2_R": se2.R,
            "se2_t": se2.t,
            "feature_dim": x_seqs.shape[-1],
        }
        return {"x_seqs": x_seqs, "mask": mask, "roles": roles, "agent_idx": agent_idx, "meta": meta}

# -------------- batching utilities ----------------
def collate_scenes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # All scenes share same T,F by construction (first 20 frames), but N varies -> pad
    B = len(batch)
    T = batch[0]["x_seqs"].shape[1]
    F = batch[0]["x_seqs"].shape[2]
    maxN = max(b["x_seqs"].shape[0] for b in batch)

    x_pad = torch.zeros((B, maxN, T, F), dtype=batch[0]["x_seqs"].dtype)
    m_pad = torch.zeros((B, maxN, T), dtype=torch.bool)
    r_pad = torch.zeros((B, maxN, 3), dtype=batch[0]["roles"].dtype)
    agent_idx = torch.zeros((B,), dtype=torch.long)

    metas = []
    for i, b in enumerate(batch):
        n = b["x_seqs"].shape[0]
        x_pad[i, :n] = b["x_seqs"]
        m_pad[i, :n] = b["mask"]
        r_pad[i, :n] = b["roles"]
        agent_idx[i] = b["agent_idx"]
        metas.append(b["meta"])
    return {"x_seqs": x_pad, "mask": m_pad, "roles": r_pad, "agent_idx": agent_idx, "metas": metas}

# -------------- example usage ---------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_csv_dir", type=str, required=True,
                        help="Folder containing train/val/test subfolders with scenario CSVs")
    parser.add_argument("--use_map", action="store_true")
    parser.add_argument("--map_dir", type=str, default=None, help="Path to argoverse-api 'map_files' directory")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    ds = ArgoverseLSTMDataset(root_csv_dir=args.root_csv_dir, split="data", map_dir=args.map_dir, use_map=args.use_map)
    sample = ds[42]

    x, mask, roles, agent_idx = sample["x_seqs"], sample["mask"], sample["roles"], sample["agent_idx"]
    print("One scene →", f"agents={x.shape[0]}", f"T={x.shape[1]}", f"F_in={x.shape[2]}", f"agent_idx={agent_idx}")

    # DataLoader with collate
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_scenes)
    batch = next(iter(dl))
    x_b, m_b = batch["x_seqs"], batch["mask"]  # [B, N, T, F], [B, N, T]
    print("Batch shapes:", x_b.shape, m_b.shape)

    # How to feed into an LSTM encoder (per agent):
    B, N, T, F = x_b.shape
    x_bn = x_b.view(B * N, T, F)  # [B*N, T, F]
    m_bn = m_b.view(B * N, T)  # [B*N, T]

    lengths = m_bn.sum(dim=1)  # valid steps per agent
    valid = lengths > 0  # keep only real agents (not padded rows)

    x_valid = x_bn[valid]
    len_valid = lengths[valid].cpu()

    lstm = torch.nn.LSTM(input_size=F, hidden_size=128, num_layers=1, batch_first=True)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        x_valid, len_valid, batch_first=True, enforce_sorted=False
    )
    _, (h, _) = lstm(packed)  # h: [1, Nv, 128], Nv = valid.sum()
    emb_valid = h[-1]  # [Nv, 128]

    # Scatter back into [B, N, 128] (zeros for padded agents)
    H = emb_valid.size(1)
    emb = torch.zeros(B * N, H, device=x_b.device, dtype=x_b.dtype)
    emb[valid] = emb_valid
    emb = emb.view(B, N, H)

    print("Embeddings shape:", emb.shape)

