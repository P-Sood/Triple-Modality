#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/iemo_whisper_rand.txt

# wandb agent ddi/must_bert/q9jen8fn --count 2 ; 
# wandb agent ddi/iemo_bert/643u0zsg --count 2
# wandb agent ddi/must_whisper/1lcj708r --count 2

# wandb sweep -p must_bert -e ddi ../hyper_parameter_config/bert_must.yaml
# wandb sweep -p must_whisper -e ddi ../hyper_parameter_config/whisper_must.yaml
# wandb sweep -p must_video -e ddi ../hyper_parameter_config/video_must.yaml

# wandb sweep -p iemo_bert -e ddi ../hyper_parameter_config/bert_iemo.yaml
# wandb sweep -p iemo_whisper -e ddi ../hyper_parameter_config/whisper_iemo.yaml
# wandb sweep -p iemo_video -e ddi ../hyper_parameter_config/video_iemo.yaml

# wandb sweep -p meld_bert -e ddi ../hyper_parameter_config/bert.yaml
# wandb sweep -p meld_whisper -e ddi ../hyper_parameter_config/whisper.yaml
# wandb sweep -p meld_video -e ddi ../hyper_parameter_config/video.yaml

python ../whisper_nn.py -d ../../data/iemo   --BertModel openai/whisper-large --T_max 3 --batch_size 1 --beta 1 --clip 0.1 --dropout 0.3 --early_div false --epoch 8 --epoch_switch 2 --hidden_size 64 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss NewCrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Both --seed   102  --sota true --text_column audio_path --weight_decay 0.01 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../whisper_nn.py -d ../../data/iemo   --BertModel openai/whisper-large --T_max 3 --batch_size 1 --beta 1 --clip 0.1 --dropout 0.3 --early_div false --epoch 8 --epoch_switch 2 --hidden_size 64 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss NewCrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Both --seed   103 --sota true --text_column audio_path --weight_decay 0.01 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
