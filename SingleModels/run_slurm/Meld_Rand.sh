#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/random.txt

wandb agent ddi/meld_bert/hylxc6zm --count 5
wandb agent ddi/meld_whisper/0tjkf2yu --count 5
wandb agent ddi/meld_video/87j946er --count 5
