"""
LSTM-based trajectory encoder that can ingest AV2 dataset batches or dummy tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class LSTMEncoderConfig:
    """Configuration controlling the LSTM encoder behaviour."""

    feature_dim: int = 5  # x, y, vx, vy, heading coming from AV2 dataset.
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    use_velocity: bool = True  # toggle for stripping velocity/heading if desired.


class DummyLSTMEncoder(nn.Module):
    """
    Minimal LSTM encoder that works with data coming from `AV2GNNForecastingDataset`
    as well as arbitrary dummy tensors supplied for quick experimentation.

    Usage examples
    --------------
    >>> encoder = DummyLSTMEncoder()
    >>> av2_batch = next(iter(dataloader))
    >>> embeddings = encoder(av2_batch)        # encode AV2 agent histories
    >>> dummy = torch.randn(4, 70, 5)           # (num_agents, obs_len, feature_dim)
    >>> embeddings = encoder(dummy_inputs=dummy)  # toggle into dummy mode
    """

    def __init__(self, config: LSTMEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LSTMEncoderConfig()

        lstm_input_dim = self._compute_effective_input_dim()
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        *,
        dummy_inputs: Optional[torch.Tensor] = None,
        agent_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch: Dictionary produced by `AV2GNNForecastingDataset.__getitem__`.
            dummy_inputs: Optional tensor of shape (N, T, F) for quick tests.
            agent_indices: Optional 1-D tensor selecting which agents to encode.
        Returns:
            Tensor of shape (N, hidden_dim) containing agent embeddings.
        """

        sequences = self._select_sequences(batch, dummy_inputs, agent_indices)
        outputs, (h_n, _) = self.lstm(sequences)
        return h_n[-1]  # last layer hidden state

    # --------------------------------------------------------------------- #
    def _select_sequences(
        self,
        batch: Optional[Dict[str, torch.Tensor]],
        dummy_inputs: Optional[torch.Tensor],
        agent_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if dummy_inputs is not None:
            return self._maybe_drop_features(dummy_inputs)

        if batch is None:
            raise ValueError("Provide either `batch` or `dummy_inputs`.")

        histories = batch["agent_hist"]  # (max_agents, obs_len, feat_dim)
        device = histories.device

        if agent_indices is None:
            # Heuristic: treat agents with any non-zero future mask as valid.
            fut_mask = batch.get("fut_mask")
            if fut_mask is not None:
                valid = fut_mask.max(dim=-1).values > 0
            else:
                valid = histories.abs().sum(dim=(1, 2)) > 0
            agent_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if agent_indices.numel() == 0:
                agent_indices = torch.arange(histories.size(0), device=device)

        sequences = histories[agent_indices]
        return self._maybe_drop_features(sequences)

    def _maybe_drop_features(self, sequences: torch.Tensor) -> torch.Tensor:
        if self.config.use_velocity:
            return sequences[..., : self.config.feature_dim]
        # Keep only xy positions (first two dims) when velocity info is disabled.
        return sequences[..., :2]

    def _compute_effective_input_dim(self) -> int:
        return self.config.feature_dim if self.config.use_velocity else 2
