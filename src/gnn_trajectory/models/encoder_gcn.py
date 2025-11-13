#!/usr/bin/env python3
"""
GCN-based MotionEncoder variant:
- Uses GCN for agent–agent interaction.
- Uses edge_length as edge_weight (via a small embedding MLP).
"""

import torch
import torch.nn as nn
from pathlib import Path
from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset
from torch_geometric.nn import GCN, GAT


class MotionEncoderGCN(nn.Module):
    def __init__(self,
                 agent_in_dim=5,
                 lane_in_dim=2,
                 lstm_hidden=64,
                 gcn_hidden=64,
                 gcn_layers=2,
                 gat_hidden=64,
                 gat_heads=4,
                 gat_layers=2):
        super().__init__()

        # -------- Temporal encoders --------
        self.agent_lstm = nn.LSTM(agent_in_dim, lstm_hidden, batch_first=True)
        self.lane_lstm = nn.LSTM(lane_in_dim, lstm_hidden, batch_first=True)

        # -------- Agent–Agent GCN (with edge lengths) --------
        # GCN here replaces the GAT_aa block.
        self.gcn_aa = GCN(
            in_channels=lstm_hidden,
            hidden_channels=gcn_hidden,
            num_layers=gcn_layers,
            out_channels=lstm_hidden,
            dropout=0.1,
            act="relu",
            norm="batchnorm",
            jk=None,
        )

        # Map scalar edge length → scalar edge weight in (0, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        # -------- Agent–Lane interaction (still GAT, bipartite) --------
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
        edges_length = batch["edges_length"]     # (E_aa,) or (E_aa, 1)

        # 1. Encode agents temporally
        agent_emb, _ = self.agent_lstm(agent_hist)   # (A, T, H)
        agent_emb = agent_emb[:, -1]                 # (A, H) last timestep

        # 2. Encode lanes (mean pool centerline points)
        lane_emb, _ = self.lane_lstm(lane_nodes)     # (L, P, H)
        lane_emb = lane_emb.mean(1)                  # (L, H)

        # 3. Agent–Agent social interaction via GCN + edge_length
        #    edges_length: one scalar per edge in edge_index_aa
        edge_w = edges_length.view(-1, 1)            # (E_aa, 1)
        edge_w = self.edge_mlp(edge_w).view(-1)      # (E_aa,) in (0, 1)

        agent_social = self.gcn_aa(
            x=agent_emb,             # (A, H)
            edge_index=edge_index_aa,
            edge_weight=edge_w,
        )                            # (A, H)

        # 4. Agent–Lane interaction (bipartite GAT, same as before)
        edge_index_al_flipped = edge_index_al[[1, 0]]
        agent_map = self.gat_al((lane_emb, agent_social), edge_index_al_flipped)

        return {
            "agent_emb": agent_emb,
            "agent_social": agent_social,
            "agent_map": agent_map,
            "lane_emb": lane_emb,
        }
