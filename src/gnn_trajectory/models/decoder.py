import torch
import torch.nn as nn

class MotionDecoder(nn.Module):
    """
    One-to-many LSTM decoder that predicts a future trajectory
    from the encoded feature of the focal agent.
    """

    def __init__(self, hidden_dim=64, pred_len=50, output_dim=2, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.num_layers = num_layers

        # LSTM decoder: takes the previous output (pos) or hidden state as input
        self.lstm = nn.LSTM(
            input_size=output_dim,  # we feed previous position or zero at t=0
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Projection from hidden state to coordinate output
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, agent_feat, start_pos=None, pred_len=None):
        """
        Args:
            agent_feat: (B, H) latent feature of focal agent
            start_pos: (B, 2) last observed position (optional)
            pred_len: int, number of predicted timesteps (default=self.pred_len)
        Returns:
            traj_pred: (B, T, 2)
        """
        B = agent_feat.size(0)
        pred_len = pred_len or self.pred_len

        # Initialize hidden/cell with agent feature
        h0 = agent_feat.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, H)
        c0 = torch.zeros_like(h0)

        # Initialize first input (e.g., last observed position or zeros)
        if start_pos is None:
            inp = torch.zeros(B, 1, self.output_dim, device=agent_feat.device)
        else:
            inp = start_pos.unsqueeze(1)  # (B, 1, 2)

        preds = []
        h, c = h0, c0
        for _ in range(pred_len):
            out, (h, c) = self.lstm(inp, (h, c))      # out: (B, 1, H)
            pos = self.fc_out(out)                    # (B, 1, 2)
            preds.append(pos)
            inp = pos                                 # next step input is previous output

        traj_pred = torch.cat(preds, dim=1)           # (B, T, 2)
        return traj_pred
