# Graph Neural Networks for Multi-Agent Trajectory Prediction

## Overview
This project explores Graph Neural Networks (GNNs) for forecasting the future motion of agents (vehicles, pedestrians) in complex traffic scenes. Each scene is modelled as a dynamic graph: agents become nodes, their interactions form edges, and contextual map features enrich message passing. The long-term goal is an end-to-end pipeline that ingests historical tracks and high-definition map data to predict accurate, socially consistent trajectories for an agent of interest.

## Objectives
- Build a scalable data pipeline that ingests trajectory datasets such as Argoverse 1 or Waymo Open.
- Encode temporal motion histories with sequence models (e.g. LSTM) to initialise agent embeddings.
- Leverage graph message passing (PyTorch Geometric) to capture multi-agent interactions and map context.
- Train and evaluate models using standard motion forecasting metrics: Average Displacement Error (ADE) and Final Displacement Error (FDE).

## Repository Layout
```
├── requirements.txt             # Core Python dependencies
├── src/
│   └── gnn_trajectory/
│       ├── __init__.py
│       ├── config.py            # Dataclasses describing experiment configuration
│       ├── data/                # Dataset wrappers and graph construction utilities
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── graph_builder.py
│       ├── metrics.py           # ADE, FDE implementations
│       ├── models/              # Encoders, interaction network, decoder
│       │   ├── __init__.py
│       │   └── gnn.py
│       ├── training/            # Training loop scaffold
│       │   ├── __init__.py
│       │   └── trainer.py
│       ├── train.py             # Experiment entry point
│       └── utils.py             # Common helpers (seeding, filesystem)
```

The code is intentionally minimalist: methods that depend on dataset specifics raise `NotImplementedError` with guidance on what to extend next.

## Getting Started
1. **Python environment**  
   - Install Anaconda or Miniconda.  
   - Create and activate a dedicated environment:
     ```bash
     conda create -n gnn-trajectory python=3.10
     conda activate gnn-trajectory
     ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: PyTorch Geometric has platform-specific wheels. Follow the official [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the simple install fails.
3. **Organise data**
   - Download and extract a motion forecasting dataset (e.g. Argoverse 1, Waymo Open).  
   - Update `ExperimentConfig.data.root` in `src/gnn_trajectory/config.py` or pass a custom config when calling `run_experiment`.
4. **Run the scaffold (dry run)**
   ```bash
   python -m gnn_trajectory.train
   ```
   With placeholder datasets the loop will exit immediately; once loaders are implemented, this launches training.

## Next Steps
- Implement dataset-specific parsing inside `TrajectoryGraphDataset._load_sample` and the corresponding collate function.
- Define the message-passing scheme within `SceneGraphBuilder` to incorporate real agent interactions and map features.
- Experiment with alternative GNN layers (GCN, Graph Transformer) and richer decoders for multi-modal predictions.
- Add logging (TensorBoard/W&B) and checkpoint evaluation scripts once the base pipeline is functional.
