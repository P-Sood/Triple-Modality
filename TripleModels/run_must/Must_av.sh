#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

# "dp_av" 8 hours

for i in 1 2 3 4 5
do
    python ../tav_nn.py --fusion "dp_av" --sampler "Both"  --loss "NewCrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 11 --model "MAE_encoder" --dataset "../../data/must" --dropout 0.2 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "sarcasm" --num_layers 6 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Both_NoAccum"  --loss "NewCrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 11 --model "MAE_encoder" --dataset "../../data/must" --dropout 0.2 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "sarcasm" --num_layers 6 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 11 --model "MAE_encoder" --dataset "../../data/must" --dropout 0.2 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "sarcasm" --num_layers 6 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 11 --model "MAE_encoder" --dataset "../../data/must" --dropout 0.2 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "sarcasm" --num_layers 6 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 11 --model "MAE_encoder" --dataset "../../data/must" --dropout 0.2 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "sarcasm" --num_layers 6 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false
done

