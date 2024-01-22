#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

wandb agent ddi/MeldFusion/c33js83w --count 4
wandb agent ddi/IemoFusion/j01mo9sy --count 4
wandb agent ddi/MustFusion/nm8f1nqv --count 4
# wandb sweep -p IemoFusion -e ddi ../hyper_parameter_config/iemo.yaml
# wandb sweep -p MustFusion -e ddi ../hyper_parameter_config/must.yaml
