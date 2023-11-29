#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/random.txt

wandb agent ddi/iemo_bert/0ctj9sa8 --count 5
wandb agent ddi/iemo_whisper/clvduwma --count 5
wandb agent ddi/iemo_video/lurx35me --count 5
