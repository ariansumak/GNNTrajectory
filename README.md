# Graph Neural Networks for Multi-Agent Trajectory Prediction

Lightweight research scaffold for experimenting with GNN/LSTM-based trajectory forecasting. The repo contains minimal PyTorch code under `src/gnn_trajectory/`, a ready-to-use Argoverse 2 dataset wrapper, and a dummy LSTM encoder/decoder that can be swapped out for richer GNN models. Install the official [Argoverse 2 API](https://github.com/argoverse/av2-api) via `pip` to access the scenario readers used throughout the codebase.

## Repository layout
```
├── README.md
├── requirements.txt              # Core Python deps for the scaffold
├── src/gnn_trajectory            # Config, AV2 dataset, models (LSTM baseline), training script
├── av2-api/                      # Optional local checkout of the Argoverse 2 API
├── argoverse1_dataset_processing.py
└── test_argoverse2.py            # CLI for visualizing AV2 samples with matplotlib
```

Everything in `src/gnn_trajectory` is intentionally lightweight. The default training run simply feeds `AV2GNNForecastingDataset` into a dummy LSTM encoder/linear decoder so you can verify the plumbing and then iterate toward richer GNN architectures.

## Environment setup
1. **Clone**
   ```bash
   git clone <your-fork-url> GNNTrajectory
   cd GNNTrajectory
   ```
2. **Python env (example with conda)**
   ```bash
   conda create -n gnn-trajectory python=3.10
   conda activate gnn-trajectory
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt        # core scaffold deps
   pip install av2                        # pulls latest Argoverse 2 API from PyPI
   # or install straight from GitHub if you need bleeding edge:
   # pip install "av2 @ git+https://github.com/argoverse/av2-api.git"
   ```
   PyTorch Geometric ships platform-specific wheels; if `pip install torch-geometric` fails, follow their [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

   *Already installed `av2` earlier?* As long as the same Python environment is still active, those modules remain available—you only need to reinstall if you created a fresh environment or upgraded Python.

## Data preparation
- Set `ExperimentConfig.data.root` (see `src/gnn_trajectory/config.py`) to either the AV2 motion-forecasting root (containing `train/`, `val/`, `test/`) or directly to a single split directory if you only downloaded one of them.
- The code expects each scenario folder to contain both the parquet file and the matching `log_map_archive_*.json` static map, mirroring the official dataset release.
- Adjust `obs_seconds`, `fut_seconds`, and `frequency_hz` in `DataConfig` to change the observation/prediction horizons used by the dataset and model.

## Training scaffold
With the dataset path configured:
```bash
PYTHONPATH=src python -m gnn_trajectory.train
```
What this does:
- Loads `AV2GNNForecastingDataset` (batch size 1 for now) via the parameters in `DataConfig`.
- Feeds each scenario to `AgentForecastingModel`, which bundles the `DummyLSTMEncoder` with a small MLP that regresses future xy coordinates for every agent.
- Optimizes a masked L2 loss (only valid future steps contribute) and logs ADE/FDE metrics computed with the same mask.

To swap in a different encoder/decoder, modify `AgentForecastingModel` (or create a new module) and rewire `train.py` accordingly—the built-in training loop already handles masked losses/metrics and automatic device placement.

## Visualization with `AV2GNNForecastingDataset`
Use the custom dataset in `src/gnn_trajectory/data/argoverse2_dataset.py` (authored for local visualization) whenever you want to work with Argoverse scenes directly inside notebooks or scripts. It expects the standard folder structure:
```
/path/to/av2/motion-forecasting/
    train/
        <scenario-id>/
            scenario_<scenario-id>.parquet
            log_map_archive_<scenario-id>.json
    val/
    test/
```
If you only downloaded a single split (e.g., just `train/`), pass the split directory itself as `root` and the loader will pick it up automatically.

Quick Python usage:

```python
from pathlib import Path
from src.gnn_trajectory.data.argoverse2_dataset import AV2GNNForecastingDataset

dataset = AV2GNNForecastingDataset(
    root=Path("/abs/path/to/av2/motion-forecasting"),  # or .../train if you only downloaded that split
    split="val",                                       # ignored when `root` already points to the split dir
    obs_sec=7,
    fut_sec=4,
    hz=10.0,
)
sample = dataset[0]
print(sample["scenario_id"])
print(sample["agent_hist"].shape)  # (max_agents, obs_len, 5)
# feed `sample` into DummyLSTMEncoder or the provided matplotlib helper.
```

Command-line helper:
```bash
python -m src.gnn_trajectory.data.argoverse2_dataset \
  --root /abs/path/to/av2/motion-forecasting \
  --split val \
  --index 0

python test_argoverse2.py \
  --root /abs/path/to/av2/motion-forecasting \
  --split val \
  --num 1 \
  --vis         # add this flag to render the local-frame trajectories + lanes
```
`test_argoverse2.py` instantiates `AV2GNNForecastingDataset`, prints tensor shapes, and (optionally) uses matplotlib to draw agent histories/futures over the local lane graph—handy for verifying your installation and the friend-provided visualization workflow.

The dataset output dictionary includes:
- `agent_hist`: local-frame xy/vx/vy/heading histories shaped `(max_agents, obs_len, 5)`.
- `fut_traj` / `fut_mask`: future targets and validity mask.
- `edge_index_aa`: agent-agent proximity graph (radius graph).
- `edge_index_al` + `lane_nodes` + `lane_topology`: lane geometry and connectivity for map-conditioned models.

Wire these tensors into `DummyLSTMEncoder`, your GNN layers, or custom plotting utilities as needed.

## Visualization tools (official `av2` package)
Installing `av2` via `pip` also gives you the official CLI utilities. For example:
```bash
python -m av2.tutorials.generate_forecasting_scenario_visualizations \
  --argoverse-scenario-dir /path/to/av2/motion-forecasting/scenarios/val \
  --viz-output-dir ./viz_outputs \
  --num-scenarios 25 \
  --selection-criteria random
```
The module path mirrors the GitHub tutorials folder and relies on the same dependencies (click, joblib, rich, ffmpeg). Use `python -m av2.tutorials.<script_name> --help` to discover other renderers such as sensor dataset or ego-view overlays.

## Next steps
- Extend `AgentForecastingModel` with graph-based interactions (e.g., swap in `InteractionNetwork` from `models/gnn.py`) or a better decoder.
- Implement smarter batching so multiple scenarios can be processed together (the current loader enforces `batch_size=1` because edge counts vary).
- Enrich the training loop with validation splits, checkpoints, and experiment tracking once you have a stable dataset path.
- Automate visualization or evaluation jobs (e.g., logging rendered trajectories per epoch) using the dataset utilities above.
