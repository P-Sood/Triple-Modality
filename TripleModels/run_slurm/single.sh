#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


for i in 101 102 103 104 105
do
    python ../tav_nn.py --beta 1 --clip 5 --loss "CrossEntropy" --mask false --seed $i --sota false --T_max 2 --epoch 12 --model "MAE_encoder" --fusion "t_p" --dataset "../../data/iemo" --dropout 0.2 --sampler "Both" --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 8 --output_dim 7 --hidden_size 1536 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
done

for i in 201 202 203 204 205
do
    python ../tav_nn.py --beta 1 --clip 1 --loss "NewCrossEntropy" --mask false --seed $i --sota false --T_max 3 --epoch 12 --model "MAE_encoder" --fusion "t_p" --dataset "../../data/iemo" --dropout 0.2 --sampler "Both" --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 7 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.01 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false
done

for i in 301 302 303 304 305
do
    python ../tav_nn.py --beta 1 --clip 5 --loss "NewCrossEntropy" --mask false --seed $i --sota false --T_max 2 --epoch 9 --model "MAE_encoder" --fusion "t_p" --dataset "../../data/iemo" --dropout 0.3 --sampler "Both" --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 6 --output_dim 7 --hidden_size 768 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false
done

