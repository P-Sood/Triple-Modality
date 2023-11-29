#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/random.txt

# wandb agent ddi/must_bert/q9jen8fn --count 4 ; 
wandb agent ddi/must_whisper/v8q1o8pe --count 4 ;wandb agent ddi/must_video/tzwjwj0x --count 4

# wandb sweep -p must_bert -e ddi ../hyper_parameter_config/bert_must.yaml
# wandb sweep -p must_whisper -e ddi ../hyper_parameter_config/whisper_must.yaml
# wandb sweep -p must_video -e ddi ../hyper_parameter_config/video_must.yaml

# wandb sweep -p iemo_bert -e ddi ../hyper_parameter_config/bert_iemo.yaml
# wandb sweep -p iemo_whisper -e ddi ../hyper_parameter_config/whisper_iemo.yaml
# wandb sweep -p iemo_video -e ddi ../hyper_parameter_config/video_iemo.yaml

# wandb sweep -p meld_bert -e ddi ../hyper_parameter_config/bert.yaml
# wandb sweep -p meld_whisper -e ddi ../hyper_parameter_config/whisper.yaml
# wandb sweep -p meld_video -e ddi ../hyper_parameter_config/video.yaml