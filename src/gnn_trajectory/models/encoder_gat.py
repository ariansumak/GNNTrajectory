#!/usr/bin/env python3
"""
Run one Argoverse2 scenario through the MotionEncoder
and print the encoded feature vector of the focal agent.
"""

import torch
from pathlib import Path
from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset
from torch_geometric.nn import GAT
import torch.nn as nn


# ------------------------------------------------------------
# Model definition (LSTM + two GATs)
# ------------------------------------------------------------
class MotionEncoder(nn.Module):
    def __init__(self,
                 agent_in_dim=5, lane_in_dim=2,
                 lstm_hidden=64, gat_hidden=64,
                 gat_heads=4, gat_layers=2):
        super().__init__()

        # Temporal encoders
        self.agent_lstm = nn.LSTM(agent_in_dim, lstm_hidden, batch_first=True)
        self.lane_lstm = nn.LSTM(lane_in_dim, lstm_hidden, batch_first=True)

        # Agent–Agent GAT
        self.gat_aa = GAT(
            in_channels=lstm_hidden,
            hidden_channels=gat_hidden,
            num_layers=gat_layers,
            out_channels=lstm_hidden,
            heads=gat_heads,
            dropout=0.1,
            act="elu",
            norm="batchnorm",
            residual=True,
        )

        # Agent–Lane GAT (bipartite)
        self.gat_al = GAT(
            in_channels=(lstm_hidden, lstm_hidden),
            hidden_channels=gat_hidden,
            num_layers=gat_layers,
            out_channels=lstm_hidden,
            heads=2,
            dropout=0.1,
            act="relu",
            norm="batchnorm",
            residual=True,
        )

    def forward(self, batch):
        agent_hist = batch["agent_hist"]         # (A, T, 5)
        lane_nodes = batch["lane_nodes"]         # (L, P, 2)
        edge_index_aa = batch["edge_index_aa"]   # (2, E_aa)
        edge_index_al = batch["edge_index_al"]   # (2, E_al)

        # 1. Encode agents temporally
        agent_emb, _ = self.agent_lstm(agent_hist)
        agent_emb = agent_emb[:, -1]             # last timestep

        # 2. Encode lanes (mean pool centerline points)
        lane_emb, _ = self.lane_lstm(lane_nodes)
        lane_emb = lane_emb.mean(1)

        total_agents = agent_emb.size(0)
        total_lanes = lane_emb.size(0)

        # Final safety mask on agent-agent edges
        if edge_index_aa.numel() > 0:
            aa_mask = (edge_index_aa[0] < total_agents) & (edge_index_aa[1] < total_agents)
            edge_index_aa = edge_index_aa[:, aa_mask]

        # Final safety mask on agent-lane edges before flip
        if edge_index_al.numel() > 0:
            valid_mask = (edge_index_al[0] < total_agents) & (edge_index_al[1] < total_lanes)
            edge_index_al = edge_index_al[:, valid_mask]
        else:
            edge_index_al = edge_index_al.new_zeros((2, 0), dtype=edge_index_al.dtype)
        edge_index_al_flipped = edge_index_al[[1, 0]]

        # 3. Agent–Agent social interaction
        agent_social = self.gat_aa(agent_emb, edge_index_aa)

        # 4. Agent–Lane interaction (sequential)
        if edge_index_al_flipped.numel() > 0:
            valid_mask = (edge_index_al_flipped[0] < lane_emb.size(0)) & (
                edge_index_al_flipped[1] < agent_social.size(0)
            )
            edge_index_al_flipped = edge_index_al_flipped[:, valid_mask]
        agent_map = self.gat_al((lane_emb, agent_social), edge_index_al_flipped)

        return {
            "agent_emb": agent_emb,
            "agent_social": agent_social,
            "agent_map": agent_map,
            "lane_emb": lane_emb,
        }


# ------------------------------------------------------------
# Main: run a single example
# ------------------------------------------------------------
def main():
    root = Path("/home/silviu/Documents/Workspace/DeepLearning/Project/av2-api")  # 🔁 <-- change this!
    dataset = AV2GNNForecastingDataset(root=root, split="val")

    # Load one sample
    sample = dataset[0]
    print(f"Loaded scenario: {sample['scenario_id']}")

    # Move tensors to a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}

    # Initialize the model
    model = MotionEncoder().to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(sample)

    agent_map = outputs["agent_map"]  # (A, H)
    agent_pos_T = sample["agent_pos_T"]  # (A, 2)

    # Find the focal agent (closest to origin)
    dists = torch.norm(agent_pos_T, dim=1)
    focal_idx = torch.argmin(dists).item()

    focal_feat = agent_map[focal_idx].cpu().numpy()

    print(f"\nFocal agent index: {focal_idx}")
    print(f"Focal agent local position: {agent_pos_T[focal_idx].tolist()}")
    print(f"Encoded feature vector (shape={focal_feat.shape}):\n{focal_feat}\n")


if __name__ == "__main__":
    main()
