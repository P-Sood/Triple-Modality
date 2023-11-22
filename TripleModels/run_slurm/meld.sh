#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/meld_fusion.txt

wandb agent ddi/Fusion/0y8z5jjv
# nohup bash -c "sleep 10; git stash; git checkout NewFeatures; sleep 10; git pull; sleep 10; sbatch meld.sh" & disown