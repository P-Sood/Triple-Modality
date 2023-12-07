#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/iemo.txt

#MELD
python ../whisper_nn.py --BertModel openai/whisper-large --T_max 3 --batch_size 2 --beta 1 --clip 1 --dropout 0.5 --early_div false --epoch 6 --epoch_switch 2 --hidden_size 128 --label_task sentiment --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss NewCrossEntropy --mask false --model MAE_encoder --num_layers 7 --patience 14 --sampler Both --seed 100 --sota true --text_column audio_path --weight_decay 0.001 --dataset ../../data/meld --input_dim 2 --output_dim 7 --lstm_layers 1
#IEMO
python ../whisper_nn.py  --BertModel openai/whisper-large --T_max 3 --batch_size 2 --beta 1 --clip 5 --dropout 0.2 --early_div false --epoch 7 --epoch_switch 2 --hidden_size 256 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-05 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Iterative --seed 100 --sota true --text_column audio_path --weight_decay 0.001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
#MUST 
python ../whisper_nn.py --BertModel openai/whisper-large --T_max 3 --batch_size 2 --beta 1 --clip 1 --dropout 0.2 --early_div false --epoch 6 --epoch_switch 3 --hidden_size 1536 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Weighted --seed 100 --sota true --text_column audio_path --weight_decay 0.0001 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1



