"""
Baseline agent-level trajectory forecaster powered by an LSTM encoder.
"""
from __future__ import annotations

import torch
from torch import nn

from .lstm_encoder import DummyLSTMEncoder, LSTMEncoderConfig


class AgentForecastingModel(nn.Module):
    """
    Minimal encoder + linear head that predicts future xy positions for each agent.

    The model treats every agent independently: it encodes the historical states
    of each agent with an LSTM and decodes the embedding with a lightweight MLP.
    """

    def __init__(
        self,
        history_dim: int,
        future_steps: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = DummyLSTMEncoder(
            LSTMEncoderConfig(feature_dim=history_dim, hidden_dim=hidden_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary with key `agent_hist` shaped
                   (batch_size?, max_agents, history_steps, history_dim).
        Returns:
            Tensor shaped like the dataset futures (B, max_agents, future_steps, 2).
        """

        histories = batch["agent_hist"]
        if histories.dim() == 3:
            histories = histories.unsqueeze(0)
        bsz, max_agents, history_steps, feature_dim = histories.shape
        flat_histories = histories.reshape(bsz * max_agents, history_steps, feature_dim)

        embeddings = self.encoder(dummy_inputs=flat_histories)
        preds = self.regressor(embeddings)
        preds = preds.view(bsz, max_agents, self.future_steps, 2)
        return preds
