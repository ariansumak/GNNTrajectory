import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.nn import GAT

from gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset


class MotionEncoder(nn.Module):
    """
    Unified GAT encoder:
    - LSTM over agent histories and lane polylines.
    - Builds a single graph over (agents + lanes).
    - Runs homogeneous GAT over this graph.
    - Returns agent embeddings after message passing as 'agent_map'.
    """

    def __init__(self,
                 agent_in_dim=5,
                 lane_in_dim=2,
                 lstm_hidden=64,
                 gat_hidden=64,
                 gat_heads=4,
                 gat_layers=2):
        super().__init__()

        self.lstm_hidden = lstm_hidden

        # Temporal encoders
        self.agent_lstm = nn.LSTM(agent_in_dim, lstm_hidden, batch_first=True)
        self.lane_lstm  = nn.LSTM(lane_in_dim, lstm_hidden, batch_first=True)

        # Unified GAT over agents + lanes
        self.gnn = GAT(
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

    def forward(self, batch):
        """
        batch keys (after collate_fn):
          - agent_hist: (SumA, T, 5)
          - lane_nodes: (SumL, P, 2)
          - edge_index_aa: (2, E_aa), indices in [0, SumA)
          - edge_index_al: (2, E_al), agents in [0, SumA), lanes in [0, SumL)
        """
        agent_hist = batch["agent_hist"]         # (A_tot, T, 5)
        lane_nodes = batch["lane_nodes"]         # (L_tot, P, 2)
        edge_index_aa = batch["edge_index_aa"]   # (2, E_aa)
        edge_index_al = batch["edge_index_al"]   # (2, E_al)

        A_tot = agent_hist.size(0)
        L_tot = lane_nodes.size(0)

        # 1. Temporal encoding
        agent_emb, _ = self.agent_lstm(agent_hist)   # (A_tot, T, H)
        agent_emb = agent_emb[:, -1]                 # (A_tot, H)

        lane_emb, _ = self.lane_lstm(lane_nodes)     # (L_tot, P, H)
        lane_emb = lane_emb.mean(1)                  # (L_tot, H)

        # 2. Build unified node feature matrix
        x = torch.cat([agent_emb, lane_emb], dim=0)  # (A_tot + L_tot, H)

        # 3. Build unified edge_index

        # Agent-agent edges: indices already in [0, A_tot)
        ei_aa = edge_index_aa

        # Agent-lane edges: agents [0, A_tot), lanes [0, L_tot) → shift lanes by A_tot
        agents_al = edge_index_al[0]                 # (E_al,)
        lanes_al  = edge_index_al[1] + A_tot         # (E_al,)

        # Make edges bidirectional
        ei_al_src = torch.cat([agents_al, lanes_al], dim=0)
        ei_al_dst = torch.cat([lanes_al, agents_al], dim=0)
        ei_al = torch.stack([ei_al_src, ei_al_dst], dim=0)  # (2, 2*E_al)

        # Combine AA + AL
        edge_index_full = torch.cat([ei_aa, ei_al], dim=1)  # (2, E_full)

        # 4. Run GAT on unified graph
        x_out = self.gnn(x, edge_index_full)                # (A_tot + L_tot, H)

        # 5. Split back
        agent_map = x_out[:A_tot]
        lane_out  = x_out[A_tot:]

        return {
            "agent_emb": agent_emb,
            "agent_map": agent_map,
            "lane_emb": lane_emb,
            "lane_out": lane_out,
        }