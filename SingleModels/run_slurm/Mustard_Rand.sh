#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/random.txt

wandb agent ddi/must_bert/83itrnt2 --count 5
wandb agent ddi/must_whisper/vav47e7q --count 5
wandb agent ddi/must_video/3c2n2axs --count 5
