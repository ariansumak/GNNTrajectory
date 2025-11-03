"""
Utilities to convert trajectory scenes into PyTorch Geometric graphs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

try:
    from torch_geometric.data import Data as GraphData
except ModuleNotFoundError:  # pragma: no cover
    GraphData = None  # type: ignore[assignment]


@dataclass
class SceneGraphBuilder:
    """
    Encapsulates the logic that turns raw scene annotations into graphs.

    The default implementation only sketches the method signatures required
    to support message-passing models; fill in with dataset-specific logic.
    """

    add_static_map: bool = True

    def __call__(
        self,
        agent_trajectories: torch.Tensor,
        agent_attributes: Dict[str, torch.Tensor],
        map_features: Dict[str, torch.Tensor] | None = None,
    ) -> "GraphData":
        """
        Build a graph capturing temporal and spatial relationships.

        Args:
            agent_trajectories: Tensor of shape (num_agents, T_h, 2).
            agent_attributes: Metadata per agent (type, velocity, etc.).
            map_features: Optional tensors that describe lanes or boundaries.
        """

        if GraphData is None:
            raise ModuleNotFoundError(
                "torch-geometric is required to instantiate SceneGraphBuilder."
            )

        node_features, node_positions = self._encode_nodes(
            agent_trajectories, agent_attributes
        )
        edge_index, edge_attributes = self._encode_edges(agent_trajectories)

        static_map = self._encode_map(map_features) if self.add_static_map else None

        return GraphData(
            x=node_features,
            pos=node_positions,
            edge_index=edge_index,
            edge_attr=edge_attributes,
            static_map=static_map,
        )

    def _encode_nodes(
        self,
        trajectories: torch.Tensor,
        attributes: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement node feature engineering.")

    def _encode_edges(self, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement edge construction strategy.")

    def _encode_map(self, map_features: Dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        raise NotImplementedError("Implement map feature encoding.")
