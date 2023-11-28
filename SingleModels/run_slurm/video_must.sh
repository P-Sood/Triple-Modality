#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/videoMAE_fullseq.txt

wandb agent ddi/must_video/apl9ufpx
# wandb agent ddi/VideoDA/4g2igbuv
# wandb sweep -p must_video -e ddi ../hyper_parameter_config/video_must.yaml