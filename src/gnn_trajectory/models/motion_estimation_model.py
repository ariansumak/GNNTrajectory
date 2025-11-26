import torch
import torch.nn as nn
from pathlib import Path

from gnn_trajectory.models.decoder import MotionDecoder
from gnn_trajectory.models.decoder_mlp import MLPDisplacementDecoder
from gnn_trajectory.models.encoder_gat import MotionEncoder
from gnn_trajectory.models.encoder_gcn import MotionEncoderGCN
from gnn_trajectory.models.encoder_gat_v2 import MotionEncoder as MotionEncoderGATv2
from gnn_trajectory.models.encoder_gcn_v2 import MotionEncoderGCN as MotionEncoderGCNv2

from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset

ENCODER_REGISTRY = {
    "gat": MotionEncoderGATv2,
    "gcn_v2": MotionEncoderGCNv2,
}

DECODER_REGISTRY = {
    "lstm": MotionDecoder,
    "mlp": MLPDisplacementDecoder,
}

def collate_fn(batch):
    """
    Batch list of scenario dictionaries into one batched dict.
    """

    agent_offset = 0
    lane_offset = 0

    out = {
        "agent_hist": [],
        "agent_pos_T": [],
        "edge_index_aa": [],
        "edges_length": [],
        "lane_nodes": [],
        "edge_index_al": [],
        "batch_agent": [],
        "batch_lane": [],
        "fut_traj": [],
        "fut_mask": [],
        "focal_indices": [],
    }

    for scenario_idx, data in enumerate(batch):
        # Ensure ints
        A = int(data["num_agents"])
        L = int(data["num_lanes"])

        # slice to real agents / lanes (drop padded)
        agent_hist = data["agent_hist"][:A]
        agent_pos_T = data["agent_pos_T"][:A]
        fut_traj    = data["fut_traj"][:A]
        fut_mask    = data["fut_mask"][:A]
        lane_nodes  = data["lane_nodes"][:L]

        # agents
        out["agent_hist"].append(agent_hist)
        out["agent_pos_T"].append(agent_pos_T)
        out["fut_traj"].append(fut_traj)
        out["fut_mask"].append(fut_mask)

        # batch vectors
        out["batch_agent"].append(torch.full((A,), scenario_idx, dtype=torch.long))
        out["batch_lane"].append(torch.full((L,), scenario_idx, dtype=torch.long))

        # edges agent-agent (offset nodes within global agent space)
        ei_aa = data["edge_index_aa"]
        el_aa = data["edges_length"]
        if ei_aa.numel() > 0:
            if (ei_aa[0] >= A).any() or (ei_aa[1] >= A).any():
                max_a0 = int(torch.max(ei_aa[0]).item())
                max_a1 = int(torch.max(ei_aa[1]).item())
                scen_id = data.get("scenario_id", "unknown")
                raise IndexError(
                    f"Agent-agent edge index exceeds agent count (scenario_idx={scenario_idx}, id={scen_id}): "
                    f"max src={max_a0}, max dst={max_a1}, A={A}"
                )
            aa_mask = (ei_aa[0] < A) & (ei_aa[1] < A)
            ei_aa = ei_aa[:, aa_mask]
            el_aa = el_aa[aa_mask]
        ei_aa = ei_aa + agent_offset
        out["edge_index_aa"].append(ei_aa)
        out["edges_length"].append(el_aa)

        # edges agent-lane (offset both sides into global agent/lane spaces)
        ei_al = data["edge_index_al"].clone()  # (2, E)
        if ei_al.numel() > 0:
            if (ei_al[0] >= A).any() or (ei_al[1] >= L).any():
                max_agent = int(torch.max(ei_al[0]).item())
                max_lane = int(torch.max(ei_al[1]).item())
                scen_id = data.get("scenario_id", "unknown")
                raise IndexError(
                    f"Agent-lane edge index exceeds counts (scenario_idx={scenario_idx}, id={scen_id}): "
                    f"agent max {max_agent} vs A={A}, lane max {max_lane} vs L={L}"
                )
            valid_mask = (ei_al[0] < A) & (ei_al[1] < L)
            ei_al = ei_al[:, valid_mask]
        ei_al[0] += agent_offset   # agents
        ei_al[1] += lane_offset    # lanes
        out["edge_index_al"].append(ei_al)

        # lane nodes
        out["lane_nodes"].append(lane_nodes)

        # record focal agent index (global)
        focal_idx_local = torch.argmin(torch.norm(agent_pos_T, dim=1)).item()
        out["focal_indices"].append(focal_idx_local + agent_offset)

        # update offsets by REAL counts
        agent_offset += A
        lane_offset  += L

    # concatenate all lists
    for k in ["agent_hist", "agent_pos_T", "fut_traj", "fut_mask"]:
        out[k] = torch.cat(out[k], dim=0)

    out["edge_index_aa"] = torch.cat(out["edge_index_aa"], dim=1)
    out["edges_length"]  = torch.cat(out["edges_length"], dim=0)
    out["edge_index_al"] = torch.cat(out["edge_index_al"], dim=1)
    out["lane_nodes"]    = torch.cat(out["lane_nodes"], dim=0)
    out["batch_agent"]   = torch.cat(out["batch_agent"], dim=0)
    out["batch_lane"]    = torch.cat(out["batch_lane"], dim=0)

    # focal_indices: one per scenario
    out["focal_indices"] = torch.tensor(out["focal_indices"], dtype=torch.long)

    return out

class MotionForecastModel(nn.Module):
    def __init__(self, encoder_name="gcn_v2", decoder_name="mlp", encoder_cfg=None, decoder_cfg=None, future_steps=None):
        super().__init__()
        encoder_cls = ENCODER_REGISTRY.get(encoder_name)
        if encoder_cls is None:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Options: {list(ENCODER_REGISTRY.keys())}")
        decoder_cls = DECODER_REGISTRY.get(decoder_name)
        if decoder_cls is None:
            raise ValueError(f"Unknown decoder '{decoder_name}'. Options: {list(DECODER_REGISTRY.keys())}")

        encoder_cfg = dict(encoder_cfg or {})
        decoder_cfg = dict(decoder_cfg or {})
        if future_steps is not None and "pred_len" not in decoder_cfg:
            decoder_cfg["pred_len"] = future_steps

        self.encoder = encoder_cls(**encoder_cfg)
        self.decoder = decoder_cls(**decoder_cfg)

    def forward(self, batch):
        # Encode the scene
        enc_out = self.encoder(batch)
        agent_map = enc_out["agent_map"]
        agent_pos_T = batch["agent_pos_T"]

        # Find focal agent (closest to origin)
        focal_indices = batch["focal_indices"]   # shape: (B,)
        focal_feat = agent_map[focal_indices]    # (B, H)
        start_pos = agent_pos_T[focal_indices]   # (B, 2)

        # Decode trajectory
        pred_traj = self.decoder(focal_feat, start_pos=start_pos)
        return pred_traj, focal_indices

# ------------------------------------------------------------
# Main: test run
# ------------------------------------------------------------
def main():
    root = Path("/home/arian-sumak/Documents/DTU/Deep Learning")
    dataset = AV2GNNForecastingDataset(root=root, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    sample = next(iter(dataloader))
    #print(f"Loaded scenario: {sample['scenario_id']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}

    model = MotionForecastModel().to(device)
    model.eval()

    with torch.no_grad():
        pred_traj, focal_idx = model(sample)

    print(f"Predicted trajectory shape: {pred_traj.shape}")
    print(f"Focal agent index: {focal_idx}")
    print(f"First few predicted coords:\n{pred_traj[0, :5].cpu().numpy()}")

if __name__ == "__main__":
    main()
