#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/must_video.txt

# wandb agent ddi/meld_video/ajzxhk4t --count 4; 
wandb agent ddi/must_video/tzwjwj0x --count 4;
# wandb agent ddi/must_video/dqimkuop --count 4;
