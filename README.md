# GNNTrajectory – GNN-based Motion Forecasting Scaffold

Minimal PyTorch project for experimenting with multi-agent trajectory forecasting on Argoverse 2. The repository focuses on clear data handling, configurable model pipelines, and reproducible training so you can prototype graph/LSTM architectures quickly.

## Repository Layout

```
├── README.md
├── requirements.txt
├── src/gnn_trajectory
│   ├── config.py              # Experiment configuration dataclasses
│   ├── data/                  # AV2 dataset wrappers
│   ├── losses.py
│   ├── metrics.py
│   ├── models/                # Encoders/decoders (LSTM + GNN variants)
│   ├── train.py               # Training entrypoint
│   └── utils.py
├── av2-api/                   # Optional local checkout of Argoverse 2 API
├── data/                      # Expected location of AV2 splits by default
└── test_argoverse2.py         # Quick visualization CLI
```

## End-to-End Pipeline Overview

1. **Dataset construction** – `AV2GNNForecastingDataset` (`src/gnn_trajectory/data/argoverse2_dataset.py`) reads an AV2 scenario directory (parquet + map JSON), converts agent trajectories and lane graphs into the focal agent’s local frame, and emits a tensor dictionary containing:
   - Agent histories (`agent_hist`), future targets/masks, agent types.
   - Agent–agent proximity edges (radius graph) and agent–lane k-NN edges.
   - Lane nodes/topology sampled from the static map.
2. **Batching** – `collate_fn` (currently defined next to the model you run) trims padded entries, concatenates agents/lanes across scenarios, updates edge indices with offsets, and records per-scenario `focal_indices` so encoders know the ego target.
3. **Model** – `MotionForecastModel` consumes the batched graph, encodes agent/map context using whichever encoder you select (GCN/GAT variants), and decodes the focal agent’s trajectory via the configured decoder (MLP displacement head or autoregressive LSTM). Pick combinations directly through `ModelConfig`/JSON without code changes.
4. **Loss & metrics** – `masked_l2_loss` (`src/gnn_trajectory/losses.py`) computes the training objective with masks for variable-length futures. `metrics.py` provides ADE, FDE, and hit-rate style accuracy.
5. **Training loop** – `src/gnn_trajectory/train.py`:
   - Builds PyTorch `DataLoader`s for train (and optional val) splits using `ExperimentConfig.data`.
   - Seeds reproducibility, creates the model with `ExperimentConfig.model`, and optimizes via Adam.
   - Logs losses/metrics to stdout and optionally TensorBoard (`TrainingConfig.log_dir`).
   - Runs validation each epoch when `data.val_split` is set, using the same loss/metric helpers.
6. **Visualization / inspection** – `test_argoverse2.py` and the dataset module can be invoked to inspect samples or plot trajectories for sanity checks.

## Quick Start

```bash
git clone --recurse-submodules <your-fork> GNNTrajectory
cd GNNTrajectory
python -m venv .venv && source .venv/bin/activate  # or use conda/mamba
pip install -r requirements.txt
pip install av2  # official Argoverse 2 API (needed for scenario/map readers)
```

Download the Argoverse 2 motion-forecasting dataset and place the split directories under `data/` (default), e.g. `data/train/<scenario_id>/…`. Alternatively set `ExperimentConfig.data.root` to wherever the dataset lives (the code accepts either the split directory itself or the root containing `train/`, `val/`, `test/`).

Launch a baseline training run (two equivalent entry points):

```bash
# Run as a script, optionally pointing to a JSON config
python src/gnn_trajectory/train.py --config configs/example_train.json

# Or run as a module (also accepts --config)
PYTHONPATH=src python -m gnn_trajectory.train
```

Both commands pick up the defaults in `ExperimentConfig` unless you pass `--config path/to/config.json`. Logs stream to stdout; TensorBoard scalars are stored in `runs/` if `tensorboard` is installed (`tensorboard --logdir runs` to visualize curves).

## Configuration & Customization

### JSON config files

`src/gnn_trajectory/train.py` accepts `--config path/to/config.json`. See `configs/example_train.json` for the schema (nested `data`, `model`, `training` blocks mirroring the dataclasses). Override only the keys you need; unspecified fields fall back to their defaults.

Example snippet:

```json
{
  "data": {
    "root": "/datasets/av2",
    "split": "train",
    "val_split": "val"
  },
  "model": {
    "encoder": "gat_v2",
    "decoder": "lstm",
    "encoder_kwargs": {
      "lstm_hidden": 96,
      "gcn_hidden": 96,
      "gcn_layers": 3
    },
    "decoder_kwargs": {
      "hidden_dim": 96,
      "num_layers": 2
    }
  },
  "training": {
    "epochs": 10,
    "learning_rate": 0.0005,
    "log_dir": "runs/exp1"
  }
}
```

Then run:

```bash
python src/gnn_trajectory/train.py --config configs/exp1.json
```

### Architecture combinations

`motion_estimation_model.MotionForecastModel` exposes multiple encoder/decoder families. In JSON configs (and the `ModelConfig` dataclass) you can set:

- `model.encoder`: choose from `gcn_v2`, `gat_v2`, `gcn`, `gat`.
- `model.decoder`: choose from `mlp` or `lstm`.
- `model.encoder_kwargs`: extra keyword args passed to the encoder constructor (e.g., `lstm_hidden`, `gcn_hidden`, `gcn_layers`).
- `model.decoder_kwargs`: keyword args for the decoder (e.g., `hidden_dim`, `num_layers`). When `pred_len` is omitted it automatically matches the configured future horizon.

Mix and match values in your config file to explore different GNN/backbone pairings without touching code.

### Programmatic configuration

Prefer Python-level control? Import the dataclasses directly:

```python
from pathlib import Path
from gnn_trajectory.config import ExperimentConfig
from gnn_trajectory.train import run_experiment

cfg = ExperimentConfig()
cfg.data.root = Path("/data/av2/motion-forecasting")
cfg.data.split = "train"
cfg.data.val_split = "val"      # leave as None if you only have train
cfg.data.obs_seconds = 7.0
cfg.data.fut_seconds = 4.0
cfg.training.epochs = 10
cfg.training.learning_rate = 5e-4
cfg.training.log_every = 20
cfg.training.log_dir = Path("runs/custom")
run_experiment(cfg)
```

Key fields:

- `DataConfig`
  - `root`: dataset directory (split or root containing multiple splits).
  - `split` / `val_split`: names of subdirectories under `root`; set `val_split=None` for train-only.
  - Temporal/spatial limits: `obs_seconds`, `fut_seconds`, `frequency_hz`, `max_agents`, `max_lanes`, `max_lane_points`, `agent_radius`, `lane_knn`.
  - `batch_size` (collated with graph offsets, so >1 is now supported) and `num_workers`.
- `ModelConfig`
  - `history_steps` / `future_steps` get derived from the data config.
  - `encoder` / `decoder`: picks from the registries described above.
  - `encoder_kwargs` / `decoder_kwargs`: dicts forwarded to those constructors.
- `TrainingConfig`
  - `epochs`, `learning_rate`, `grad_clip`, `device`.
  - `log_every`: frequency (in optimizer steps) for printing/logging train metrics.
  - `log_dir`: set to `None` to disable TensorBoard; otherwise events go under the specified directory.
  - `checkpoint_dir`: target folder for saving checkpoints (hook this up when you add checkpointing).

### Switching Architectures

- **Simple swap** – adjust `model.encoder`, `model.decoder`, and their `*_kwargs` entries in your config JSON. The registries in `motion_estimation_model.py` map strings to classes (`gat_v2`, `gcn_v2`, etc.), so you can flip between combinations without code edits.
- **New model class** – create `src/gnn_trajectory/models/my_model.py` that consumes the collated batch dict, then update `train.py` to instantiate it instead of `MotionForecastModel`. Keep the contract (`pred_traj, focal_idx = model(batch)`) so losses and metrics keep functioning.
- **Different batching** – if you want a radically different data layout, implement another `collate_fn` and pass it to `_build_dataloader`. The provided collate already handles multi-scenario batches with graph index offsets, so you only need to change it for non-standard batching strategies.

## Evaluation & Metrics

Metrics live in `src/gnn_trajectory/metrics.py`:

- `average_displacement_error` (ADE)
- `final_displacement_error` (FDE)
- `make_hit_rate_metric(threshold)` – yields final displacement hit-rate at a given threshold (default 2 m in the training loop).

Add your own functions in the same file (accepting `pred, target, mask`) and register them in the `metrics` dict inside `train.py`. They will automatically be computed/logged during training and validation.

Validation is optional: when `cfg.data.val_split` is set, `train.py` builds a validation loader and runs `_evaluate` after each epoch, logging averaged loss/metrics with a `val/` prefix (also written to TensorBoard when enabled). Leave `val_split=None` if you only have the train dataset.

## Visualization / Sanity Checks

- `test_argoverse2.py` – CLI that samples from `AV2GNNForecastingDataset`, prints tensor shapes, and can plot local-frame trajectories plus lane graphs (`--vis` flag).
- `scripts/demo_lstm_encoder.py` – runs the LSTM encoder on synthetic data to confirm shapes.
- Official AV2 visualizers – once `av2` is installed, explore `python -m av2.tutorials.*` scripts to render motion-forecasting scenarios.

## Extending the Project

- **GNN encoders** – leverage `MotionEncoderGCN`, `MotionEncoderGAT`, or create new graph layers for agent-map interactions.
- **Better decoders** – integrate multi-modal prediction heads, mixture density networks, or trajectory sampling decoders.
- **Evaluation** – add forecasting benchmarks (minADE/minFDE across K samples, negative log-likelihood, etc.) as new metrics.
- **Experiment tracking** – integrate Weights & Biases or other loggers by swapping `_log_metrics`/TensorBoard calls.
- **Checkpoints** – save `model.state_dict()` and optimizer state in `checkpoint_dir` at desired intervals or when validation improves.

## Troubleshooting

- `ModuleNotFoundError: gnn_trajectory…` – ensure you run scripts with `PYTHONPATH=src` or install the package (`pip install -e .` if you add a `pyproject.toml`).
- Missing AV2 dataset files – the dataset loader expects each scenario folder to contain both the parquet and `log_map_archive_*.json`. Double-check your download.
- TensorBoard not installed – the training script gracefully disables TB logging when `tensorboard` isn’t available. `pip install tensorboard` to enable curves.

Happy experimenting! Open issues/PRs to contribute new encoders, datasets, or training utilities.
