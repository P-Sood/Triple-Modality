#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

python ../tav_nn.py --T_max 2 --batch_size 32 --clip 5 --dropout 0.3 --early_div false --epoch 7 --epoch_switch 3 --fusion concat --hidden_size 64 --label_task sentiment --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_encoders 5 --num_layers 7 --patience 14 --sampler Weighted --seed 104 --weight_decay 0.001 --sota false --BertModel roberta-large --text_column text --beta 1 --dataset ../../data/meld --input_dim 2 --output_dim 7 --lstm_layers 1 --hidden_layers 512

