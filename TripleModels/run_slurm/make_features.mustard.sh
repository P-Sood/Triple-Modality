#!/bin/bash

# Give job a name

#SBATCH --time 00-01:00 # time (DD-HH:MM)
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

# #SBATCH --output=../outputs/mustard_test_prev_SOTA.txt
#SBATCH --output=../outputs/mustard_features.txt

python ../tav_nn.py -e 1 -d ../../data/must -lt sarcasm

#wandb agent ddi/TiktokTest/nreuhli2 # OURS
# wandb agent ddi/MustardTriple/nznlgus3  # Prev. SOTA paper
