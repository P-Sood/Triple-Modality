#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/mustfusion.txt

# wandb agent ddi/Fusion/ph1w0vvg
# wandb agent ddi/MustFusion/im41427r
wandb agent ddi/MustFusion/haq66ypn