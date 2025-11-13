import torch
import torch.nn as nn

class MLPDisplacementDecoder(nn.Module):
    """
    Predicts Δx, Δy at each timestep using an MLP + integrates them.
    """

    def __init__(self, hidden_dim=64, pred_len=50, time_emb_dim=16, output_dim=2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.time_emb = nn.Embedding(pred_len, time_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)   # Δx, Δy
        )

    def forward(self, agent_feat, start_pos, pred_len=None):
        B, H = agent_feat.shape
        T = pred_len or self.pred_len

        time_ids = torch.arange(T, device=agent_feat.device)
        t_emb = self.time_emb(time_ids)
        z_rep = agent_feat.unsqueeze(1).expand(B, T, H)

        inp = torch.cat([z_rep, t_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        delta = self.mlp(inp)   # (B, T, 2)

        # integrate to get absolute positions
        traj = start_pos.unsqueeze(1) + torch.cumsum(delta, dim=1)
        return traj