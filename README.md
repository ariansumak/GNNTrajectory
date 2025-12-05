# GNNTrajectory – GNN-based Motion Forecasting for Argoverse 2

GNNTrajectory is a compact PyTorch codebase for graph/lane-aware motion forecasting on the Argoverse 2 motion-forecasting benchmark. It provides configurable encoders/decoders (GAT/GCN + MLP/LSTM), solid data handling for AV2 scenarios, and ready-to-use tooling for training and qualitative visualization.

## What the project does
- Builds local-frame scene graphs from AV2 scenarios (agents, lanes, edges).
- Trains graph-based encoders with flexible decoders to predict the focal agent’s future trajectory.
- Evaluates with ADE/FDE/hit-rate metrics and saves checkpoints.
- Visualizes predictions vs. ground truth for multiple checkpoints side-by-side (see notebook below).

## Setup (works on any laptop)
```bash
git clone <your-repo-url> GNNTrajectory
cd GNNTrajectory

# Grab the official AV2 API locally (needed for scenario + map loading)
git clone https://github.com/argoverse/av2-api.git

# Create/activate a conda env from inside the av2-api checkout (recommended)
cd av2-api
conda create -n av2 python=3.10 -y
conda activate av2

# Install the package and dependencies
pip install -e .                 # editable install of av2
pip install -r requirements.txt  # project/runtime dependencies

# Register the Jupyter kernel for this environment
pip install ipykernel
python -m ipykernel install --user --name av2 --display-name "av2"
```

## Data
Download the Argoverse 2 motion-forecasting dataset and point `data.root` to it. You can either:
- Place splits under `data/` (e.g., `data/train/<scenario_id>/`, `data/val/<scenario_id>/`, `data/test/<scenario_id>/`), or
- Set an absolute path in your config / notebook (supports pointing directly at a split).
- For a quick demo, you can drop a few scenario folders under `notebooks/data` and use that as `AV2_ROOT` in the notebook.

## Train
```bash
# With defaults
python -m gnn_trajectory.train

# With a JSON config
python -m gnn_trajectory.train --config configs/example_train.json
```

## Visualize (CLI)
Plots focal history + full GT and overlays multiple checkpoints. Pass as many checkpoints as you like; the script auto-extends the dataset future horizon to the longest checkpoint so GT is never truncated.
```bash
PYTHONPATH=src python -m gnn_trajectory.visualize_predictions \
  --root /path/to/av2 \
  --split val --index 0 --num 1 \
  --focal-only --separate \
  --checkpoint \
    src/gnn_trajectory/checkpoints/ckpt_2sec_gat-lstm.pt \
    src/gnn_trajectory/checkpoints/ckpt_2sec_gat-mlp.pt
```
- Flags: `--separate` plots one row per checkpoint; `--focal-only` hides non-focal agents; `--index/--num` select scenarios; `--device` overrides GPU/CPU choice.
- Make sure `PYTHONPATH=src` (or install with `pip install -e .`) so imports resolve.

## Visualize (Jupyter, shareable)
Open `notebooks/visualize_and_compare.ipynb` for a quick, shareable view:
- Set `AV2_ROOT` (defaults to `./notebooks/data`), choose scenario indices, and pick which checkpoint groups to plot.
- Uses the same `visualize_with_prediction` helper as the CLI, so the full GT horizon is always shown (independent of prediction length) and multiple checkpoints can be overlaid or separated into rows.
- Toggle `FOCAL_ONLY`/`SEPARATE` in the config cell; the notebook already lists the repo-relative checkpoints used in the paper figures.

## Repository layout
```
├── README.md
├── pyproject.toml                # enables pip install -e .
├── requirements.txt              # runtime deps (PyTorch, torch-geometric, matplotlib, etc.)
├── notebooks/
│   └── visualize_and_compare.ipynb
├── src/gnn_trajectory/
│   ├── config.py                 # Experiment configuration dataclasses
│   ├── data/                     # AV2 dataset wrappers
│   ├── models/                   # Encoders/decoders (GNN + LSTM/MLP)
│   ├── train.py                  # Training entrypoint
│   └── visualize_predictions.py  # Multi-ckpt visualization utilities/CLI
├── av2-api/                      # Local checkout of the AV2 API (git clone)
├── checkpoints/                  # Place your trained .pt files here
└── configs/                      # JSON config examples
```

## Extra notes
- Metrics: ADE/FDE/hit-rate live in `src/gnn_trajectory/metrics.py`.
- Checkpoint loading: the visualizer reads per-ckpt configs to build the right model, and auto-extends the dataset future horizon to the longest ckpt you pass.
- TensorBoard: enabled when `tensorboard` is installed; logs go to `runs/`.

Questions or issues? Send an email to one of the creators. Happy forecasting!
