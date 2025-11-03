"""
Command-line entry point for training runs.
"""
from __future__ import annotations

from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import SceneGraphBuilder, TrajectoryGraphDataset
from .metrics import average_displacement_error, final_displacement_error
from .models import TrajectoryPredictor
from .training import Trainer
from .utils import seed_everything


def run_experiment(config: ExperimentConfig | None = None) -> None:
    """
    Assemble data loaders, instantiate the model, and launch training.
    """

    cfg = config or ExperimentConfig()
    model_cfg = cfg.materialize_model_config()

    seed_everything(42)

    graph_builder = SceneGraphBuilder()
    dataset = TrajectoryGraphDataset(
        root=cfg.data.root,
        split="train",
        graph_builder=graph_builder,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=TrajectoryGraphDataset.collate,
        shuffle=True,
    )

    val_loader = None  # Replace with validation loader when data is ready.

    model = TrajectoryPredictor(
        history_steps=model_cfg.history_steps,
        future_steps=model_cfg.future_steps,
        embedding_dim=model_cfg.embedding_dim,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=average_displacement_error,
        metrics={
            "ADE": average_displacement_error,
            "FDE": final_displacement_error,
        },
        config=cfg.training,
    )

    trainer.fit()


if __name__ == "__main__":
    run_experiment()
