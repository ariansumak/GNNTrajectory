"""
Neural network modules backing the trajectory prediction pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv, global_add_pool
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    GATv2Conv = None  # type: ignore[assignment]
    global_add_pool = None  # type: ignore[assignment]


class TrajectoryEncoder(nn.Module):
    """
    Encodes historical agent trajectories into latent embeddings.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectories: Tensor of shape (batch, time, 2).
        """

        outputs, (h_n, _) = self.lstm(trajectories)
        # Use the final hidden state as the agent embedding.
        return h_n[-1]


@dataclass
class InteractionConfig:
    hidden_dim: int = 64
    heads: int = 4
    conv_layers: int = 2


class InteractionNetwork(nn.Module):
    """
    Relational reasoning over agent embeddings via graph message passing.

    Uses multi-head attention convolutions (GATv2) by default. Replace with
    other PyG layers (e.g., GCN, GraphTransformer) as experiments dictate.
    """

    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        if GATv2Conv is None:
            raise ModuleNotFoundError(
                "torch-geometric is required for InteractionNetwork."
            )

        layers = []
        for i in range(config.conv_layers):
            layers.append(
                GATv2Conv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim // config.heads,
                    heads=config.heads,
                    concat=True,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = node_embeddings
        for layer in self.layers:
            x = layer(x, edge_index).relu()

        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if global_add_pool is None:
            raise ModuleNotFoundError("torch-geometric is required for pooling.")

        graph_embedding = global_add_pool(x, batch_index)
        return x, graph_embedding


class TrajectoryDecoder(nn.Module):
    """
    Predicts future coordinates conditioned on agent and global embeddings.
    """

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, agent_embedding: torch.Tensor, scene_embedding: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([agent_embedding, scene_embedding], dim=-1)
        coords = self.mlp(fused)
        return coords.view(-1, self.horizon, 2)


class TrajectoryPredictor(nn.Module):
    """
    End-to-end wrapper combining encoder, interaction network, and decoder.
    """

    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        embedding_dim: int = 64,
        interaction_cfg: Optional[InteractionConfig] = None,
    ) -> None:
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim=2, hidden_dim=embedding_dim)
        self.interaction = InteractionNetwork(
            interaction_cfg or InteractionConfig(hidden_dim=embedding_dim)
        )
        decoder_input = embedding_dim * 2  # agent and scene embeddings
        self.decoder = TrajectoryDecoder(input_dim=decoder_input, horizon=future_steps)

        self.history_steps = history_steps
        self.future_steps = future_steps

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing keys `history`, `edge_index`, `batch`.
        """

        agent_history = batch["history"]
        agent_embedding = self.encoder(agent_history)

        node_embeddings, scene_embedding = self.interaction(
            node_embeddings=agent_embedding,
            edge_index=batch["edge_index"],
            batch_index=batch.get("batch"),
        )

        focal_agent_index = batch.get("focal_agent_index", None)
        if focal_agent_index is not None:
            agent_embedding = node_embeddings[focal_agent_index]

        return self.decoder(agent_embedding, scene_embedding)
