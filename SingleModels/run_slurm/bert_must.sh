#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/big_batch.txt


wandb agent ddi/must_bert/aoago6hk
# wandb agent ddi/meld_iemo_text/de3djeoo
# wandb sweep -p must_bert -e ddi ../hyper_parameter_config/bert_must.yaml