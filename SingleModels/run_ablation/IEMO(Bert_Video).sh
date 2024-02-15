#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/bert_video.txt

#BERT
python ../text_nn.py --BertModel roberta-large --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 3 --hidden_size 768 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Iterative --seed 100 --sota false --text_column text --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../text_nn.py --BertModel roberta-large --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 3 --hidden_size 768 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Iterative --seed 101 --sota false --text_column text --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../text_nn.py --BertModel roberta-large --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 3 --hidden_size 768 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Iterative --seed 102 --sota false --text_column text --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../text_nn.py --BertModel roberta-large --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 3 --hidden_size 768 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss WeightedCrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Iterative --seed 103 --sota false --text_column text --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1

#VideoMAE
python ../video_nn.py  --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 2 --hidden_size 128 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Weighted --seed 100 --sota false --text_column video_path --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../video_nn.py  --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 2 --hidden_size 128 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Weighted --seed 101 --sota false --text_column video_path --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../video_nn.py  --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 2 --hidden_size 128 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Weighted --seed 102 --sota false --text_column video_path --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1
python ../video_nn.py  --BertModel MCG-NJU/videomae-base --T_max 3 --batch_size 16 --beta 1 --clip 1 --dropout 0.4 --early_div false --epoch 6 --epoch_switch 2 --hidden_size 128 --label_task emotion --learn_PosEmbeddings false --learning_rate 5.0e-06 --loss CrossEntropy --mask false --model MAE_encoder --num_layers 5 --patience 14 --sampler Weighted --seed 103 --sota false --text_column video_path --weight_decay 0.0001 --dataset ../../data/iemo --input_dim 2 --output_dim 7 --lstm_layers 1