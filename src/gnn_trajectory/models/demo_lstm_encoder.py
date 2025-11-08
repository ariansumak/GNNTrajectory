"""
Quick smoke test for `DummyLSTMEncoder` without touching any dataset.

Run:
    PYTHONPATH=src python scripts/demo_lstm_encoder.py --agents 4 --timesteps 70 --features 5
"""
from __future__ import annotations

import argparse

import torch

from lstm_encoder import DummyLSTMEncoder, LSTMEncoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy run of the LSTM encoder.")
    parser.add_argument("--agents", type=int, default=4, help="Number of dummy agents.")
    parser.add_argument("--timesteps", type=int, default=70, help="Length of the dummy history.")
    parser.add_argument("--features", type=int, default=5, help="Feature dimension per timestep.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size inside the encoder.")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--no-velocity", action="store_true", help="Only feed xy positions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    dummy_history = torch.randn(args.agents, args.timesteps, args.features)
    config = LSTMEncoderConfig(
        feature_dim=args.features,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        use_velocity=not args.no_velocity,
    )
    encoder = DummyLSTMEncoder(config)

    embeddings = encoder(dummy_inputs=dummy_history)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(embeddings)


if __name__ == "__main__":
    main()
