#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

python ../tav_nn.py --T_max 2 --batch_size 32 --clip 0.1 --dropout 0.4 --early_div false --epoch 11 --epoch_switch 2 --fusion sota --hidden_size 256 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 8 --patience 14 --sampler Weighted --seed 100 --weight_decay 0.001 --sota false --BertModel roberta-large --text_column text --beta 1 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1 --num_encoders 3 --hidden_layers 512 
# python ../tav_nn.py --T_max 2 --batch_size 16 --clip 0.1 --dropout 0.4 --early_div false --epoch 7 --epoch_switch 3 --fusion sota --hidden_size 128 --label_task sarcasm --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_encoders 2 --num_layers 8 --patience 14 --sampler Iterative --seed 100 --weight_decay 0.0001 --sota false --BertModel roberta-large --text_column text --beta 1 --dataset ../../data/must --input_dim 2 --output_dim 7 --lstm_layers 1 --hidden_layers 512
