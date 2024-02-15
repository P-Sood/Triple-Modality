#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/iemo_whisper_rand.txt

# wandb agent ddi/meld_whisper/eazriup3 --count 1; wandb agent ddi/iemo_whisper/fqkokoud --count 1

python ../whisper_nn.py -d ../../data/iemo   --BertModel openai/whisper-large --T_max 3 --batch_size 1 --beta 1 --clip 0.1 --dropout 0.3 --early_div false --epoch 8 --epoch_switch 2 --hidden_size 64 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss NewCrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Both --seed 100  --sota true --text_column audio_path --weight_decay 0.01 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../whisper_nn.py -d ../../data/iemo   --BertModel openai/whisper-large --T_max 3 --batch_size 1 --beta 1 --clip 0.1 --dropout 0.3 --early_div false --epoch 8 --epoch_switch 2 --hidden_size 64 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss NewCrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Both --seed  101  --sota true --text_column audio_path --weight_decay 0.01 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
