#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

# wandb agent ddi/IemoTriple/uufy7lov
# wandb agent ddi/IemoFusion/uuj7hfce
# wandb agent ddi/IemoFusion/wynl2e0t



wandb agent ddi/NewPepe_N/b5broifi

