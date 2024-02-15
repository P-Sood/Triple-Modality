#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/meld_bert_random.txt

python ../text_nn.py -d ../../data/meld  --BertModel roberta-large --T_max 3 --batch_size 24 --beta 1 --clip 0.1 --dropout 0.5 --early_div false --epoch 7 --epoch_switch 3 --hidden_size 768 --label_task sentiment --learn_PosEmbeddings false --learning_rate 1.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 6 --patience 14 --sampler Both --seed 100 --sota false --text_column text --weight_decay 0.001 --dataset ../../data/meld --input_dim 2 --output_dim 7 --lstm_layers 1
