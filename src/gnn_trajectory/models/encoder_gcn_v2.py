#!/usr/bin/env python3

import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.nn import GCN

from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset


class MotionEncoderGCN(nn.Module):
    """
    Unified GCN encoder:
    - LSTM over agent histories and lane polylines.
    - Single graph over (agents + lanes).
    - Uses edges_length as edge_weight for agent-agent edges.
    - Agent-lane edges get weight 1.0.
    """

    def __init__(self,
                 agent_in_dim=5,
                 lane_in_dim=2,
                 lstm_hidden=64,
                 gcn_hidden=64,
                 gcn_layers=2,
                 with_edge_weights=True):
        super().__init__()

        self.lstm_hidden = lstm_hidden
        self.with_edge_weights = with_edge_weights
        # Temporal encoders
        self.agent_lstm = nn.LSTM(agent_in_dim, lstm_hidden, batch_first=True)
        self.lane_lstm  = nn.LSTM(lane_in_dim, lstm_hidden, batch_first=True)

        # Unified GCN
        self.gnn = GCN(
            in_channels=lstm_hidden,
            hidden_channels=gcn_hidden,
            num_layers=gcn_layers,
            out_channels=lstm_hidden,
            dropout=0.1,
            act="relu",
            norm="batchnorm",
            jk=None,
        )

        # Map scalar edge length → scalar edge weight (0,1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        agent_hist = batch["agent_hist"]         # (A_tot, T, 5)
        lane_nodes = batch["lane_nodes"]         # (L_tot, P, 2)
        edge_index_aa = batch["edge_index_aa"]   # (2, E_aa)
        edges_length  = batch["edges_length"]    # (E_aa,)
        edge_index_al = batch["edge_index_al"]   # (2, E_al)
        edges_length_al = batch["edges_length_al"]  # (E_al,)
        A_tot = agent_hist.size(0)
        L_tot = lane_nodes.size(0)

        # 1. Temporal encoding
        agent_emb, _ = self.agent_lstm(agent_hist)
        agent_emb = agent_emb[:, -1]             # (A_tot, H)

        lane_emb, _ = self.lane_lstm(lane_nodes)
        lane_emb = lane_emb.mean(1)              # (L_tot, H)

        # 2. Unified node features
        x = torch.cat([agent_emb, lane_emb], dim=0)  # (A_tot + L_tot, H)

        # 3. Build edges + weights

        # Agent-agent edges
        ei_aa = edge_index_aa                     # (2, E_aa)
        w_aa = self.edge_mlp(edges_length.view(-1, 1)).view(-1)  # (E_aa,)
        # Compute weights for agent-lane edges and duplicate for both directions
        w_al_single = self.edge_mlp(edges_length_al.view(-1, 1)).view(-1)  # (E_al,)
        w_al = torch.cat([w_al_single, w_al_single], dim=0)  # (2*E_al,)

        # Agent-lane edges (bidirectional, weight=1)
        agents_al = edge_index_al[0]
        lanes_al  = edge_index_al[1] + A_tot

        ei_al_src = torch.cat([agents_al, lanes_al], dim=0)
        ei_al_dst = torch.cat([lanes_al, agents_al], dim=0)
        ei_al = torch.stack([ei_al_src, ei_al_dst], dim=0)    # (2, 2*E_al)
        #w_al  = torch.ones(ei_al.size(1), device=x.device)    # (2*E_al,)

        # Combine
        edge_index_full = torch.cat([ei_aa, ei_al], dim=1)    # (2, E_full)
        edge_weight_full = torch.cat([w_aa, w_al], dim=0)     # (E_full,)

        # 4. GCN forward
        if self.with_edge_weights:
            x_out = self.gnn(x, edge_index_full, edge_weight=edge_weight_full)
        else:
            x_out = self.gnn(x, edge_index_full)

        # 5. Split back
        agent_map = x_out[:A_tot]
        lane_out  = x_out[A_tot:]

        return {
            "agent_emb": agent_emb,
            "agent_map": agent_map,
            "lane_emb": lane_emb,
            "lane_out": lane_out,
        }
