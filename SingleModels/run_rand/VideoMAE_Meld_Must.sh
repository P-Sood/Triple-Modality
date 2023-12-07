#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/must_video.txt


python3 ../video_nn.py -d ../../data/must --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 4 --beta 1 --clip 1 --dropout 0.5 --early_div false --epoch 7 --epoch_switch 3 --hidden_size 128 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Weighted --seed 100 --sota true --text_column video_path --weight_decay 0.01 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1
python3 ../video_nn.py -d ../../data/must --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 4 --beta 1 --clip 1 --dropout 0.5 --early_div false --epoch 7 --epoch_switch 3 --hidden_size 128 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Weighted --seed 101  --sota true --text_column video_path --weight_decay 0.01 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1
python3 ../video_nn.py -d ../../data/must --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 4 --beta 1 --clip 1 --dropout 0.5 --early_div false --epoch 7 --epoch_switch 3 --hidden_size 128 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Weighted --seed  102 --sota true --text_column video_path --weight_decay 0.01 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1
python3 ../video_nn.py -d ../../data/must --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 4 --beta 1 --clip 1 --dropout 0.5 --early_div false --epoch 7 --epoch_switch 3 --hidden_size 128 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 4 --patience 14 --sampler Weighted --seed  103  --sota true --text_column video_path --weight_decay 0.01 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1
