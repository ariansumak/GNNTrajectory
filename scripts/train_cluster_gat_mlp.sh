#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_gnn_gat_mlp
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 04:00
#BSUB -u s253819@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o logs/train_gnn_gat_mlp.out
#BSUB -e logs/train_gnn_gat_mlp.err

CONFIG_PATH="configs/cluster_train_gat_mlp.json"

if [ ! -d "/dtu/blackhole/07/224071/GNNenv" ]; then
    python3 -m venv "/dtu/blackhole/07/224071/GNNenv"
fi

source "/dtu/blackhole/07/224071/GNNenv/bin/activate"
module load cuda/12.8.1

cd /dtu/blackhole/07/224071/GNNTrajectory

export PYTHONPATH=src
python -m gnn_trajectory.train --config "$CONFIG_PATH"
