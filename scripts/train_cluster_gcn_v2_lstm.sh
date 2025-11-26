#!/bin/sh
### General options
### -- specify queue --
#BSUB -q c02516
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### –- specify queue --
### -- set the job Name --
#BSUB -J train_gnn_traj
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need xGB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
###BSUB -R "select[gpu80gb]"
###BSUB -R "select[gpu32gb]"
#### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
###BSUB -M 128GB
### -- set walltime limit: hh:mm --
#BSUB -W 10:00
### -- set the email address --
#BSUB -u s253819@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- set the job output file --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_gnn_traj_%J.out
#BSUB -e train_gnn_traj_%J.err
# all  BSUB option comments should be above this line!

if [ ! -d "/dtu/blackhole/07/224071/GNNenv" ]; then
    python3 -m venv "/dtu/blackhole/07/224071/GNNenv"
fi

# Activate your venv in blackhole
source "/dtu/blackhole/07/224071/GNNenv/bin/activate"
module load cuda/12.8.1

# go to the repo root on blackhole
cd /dtu/blackhole/07/224071/GNNTrajectory

# Run training from repo root
export PYTHONPATH=src
python src/gnn_trajectory/train.py --config configs/cluster_train_gcn_v2_lstm.json
