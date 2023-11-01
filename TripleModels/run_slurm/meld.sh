#!/bin/bash

# Give job a name

#SBATCH --time 00-12:00 # time (DD-HH:MM)
#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/meld_triple.txt
# wandb agent ddi/MustardTest/y97hyzfr # OURS
# wandb agent ddi/MustardTriple/nznlgus3  # Prev. SOTA paper
wandb agent ddi/MeldTriple/8krlsw9c
