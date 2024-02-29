#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

# "t_c" 5.5 hours


wandb agent ddi/UrFunny_Tri_Acc_Final/nquu38p7