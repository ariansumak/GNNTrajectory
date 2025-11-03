"""
Dataset abstractions for trajectory forecasting experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data as GraphData
except ModuleNotFoundError:  # pragma: no cover - soft dependency
    GraphData = None  # type: ignore[assignment]


@dataclass
class TrajectorySample:
    """
    Container describing a single training example.

    Attributes:
        scene_id: Unique identifier of the scene.
        agent_id: Identifier of the focal agent whose trajectory we predict.
        history: Tensor of shape (T_h, 2) containing past xy coordinates.
        future: Optional tensor of shape (T_f, 2) with future coordinates.
        graph: Optional PyG graph describing the scene context.
    """

    scene_id: str
    agent_id: str
    history: torch.Tensor
    future: Optional[torch.Tensor]
    graph: Optional["GraphData"]


class TrajectoryGraphDataset(Dataset[TrajectorySample]):
    """
    Thin `torch.utils.data.Dataset` wrapper around preprocessed trajectory data.

    Parameters:
        root: Directory that stores trajectory files.
        split: One of {"train", "val", "test"}.
        graph_builder: Callable that converts raw scene data into a PyG graph.
        cache_in_memory: If True, keep processed samples in memory.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        graph_builder,
        cache_in_memory: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.graph_builder = graph_builder
        self.cache_in_memory = cache_in_memory

        self._samples: Optional[list[TrajectorySample]] = None
        self._index: list[Dict[str, str]] = self._load_index()

    def _load_index(self) -> list[Dict[str, str]]:
        """
        Discover files belonging to the chosen split.

        In practice this could parse JSON/CSV manifests supplied with datasets
        such as Argoverse or Waymo. For now we only return an empty placeholder
        list to be filled once data is available.
        """

        manifest_path = self.root / f"{self.split}_manifest.json"
        if manifest_path.exists():
            raise NotImplementedError(
                "Manifest parsing not yet implemented. Provide custom loader."
            )
        # Returning an empty list makes the dataset iterable even without data.
        return []

    def _maybe_prepare_cache(self) -> None:
        if self._samples is not None or not self.cache_in_memory:
            return

        self._samples = [self._load_sample(meta) for meta in self._index]

    def _load_sample(self, meta: Dict[str, str]) -> TrajectorySample:
        """
        Translate metadata describing a scene into a `TrajectorySample`.

        Replace the body of this method once the dataset format is known.
        """

        raise NotImplementedError(
            "Implement dataset-specific loading logic once data is available."
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> TrajectorySample:
        self._maybe_prepare_cache()

        if self.cache_in_memory and self._samples is not None:
            return self._samples[idx]

        return self._load_sample(self._index[idx])

    @staticmethod
    def collate(samples: Iterable[TrajectorySample]) -> Dict[str, torch.Tensor]:
        """
        Default collate function assembling batch tensors for a DataLoader.
        """

        samples = list(samples)
        if not samples:
            return {
                "history": torch.empty(0, 0, 2),
                "future": torch.empty(0, 0, 2),
                "edge_index": torch.empty(2, 0, dtype=torch.long),
                "batch": torch.empty(0, dtype=torch.long),
            }

        histories = torch.stack([sample.history for sample in samples])
        future_list = [sample.future for sample in samples if sample.future is not None]
        if len(future_list) != len(samples):
            raise ValueError(
                "All samples must include future trajectories for supervised training."
            )
        futures = torch.stack(future_list)

        # Placeholder edge_index/batch until graphs are built.
        edge_index = torch.empty(2, 0, dtype=torch.long)
        batch_index = torch.arange(histories.size(0), dtype=torch.long)

        return {
            "history": histories,
            "future": futures,
            "edge_index": edge_index,
            "batch": batch_index,
        }
