#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


for i in 1 2 3 4 443
do
    python ../tav_nn.py --fusion "d_c" --sampler "Both"  --seed $i --clip 0.1  --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 384 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c" --sampler "Both_NoAccum"  --seed $i --clip 0.1  --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 384 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c" --sampler "Iter_Accum"  --seed $i --clip 0.1  --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 384 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c" --sampler "Iterative"  --seed $i --clip 0.1  --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 384 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c" --sampler "Weighted"  --seed $i --clip 0.1  --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 384 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
done


for i in 1 2 3 4 594
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both"  --seed $i --clip 0.1 --T_max 2 --epoch 10 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 256 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum"  --seed $i --clip 0.1 --T_max 2 --epoch 10 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 256 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum"  --seed $i --clip 0.1 --T_max 2 --epoch 10 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 256 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative"  --seed $i --clip 0.1 --T_max 2 --epoch 10 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 256 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted"  --seed $i --clip 0.1 --T_max 2 --epoch 10 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 256 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000001
done

for i in 1 2 3 4 682
do
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both"  --seed $i --clip 5 --T_max 3 --epoch 6 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 64 --epoch_switch 2 --num_encoders 5 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both_NoAccum"  --seed $i --clip 5 --T_max 3 --epoch 6 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 64 --epoch_switch 2 --num_encoders 5 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iter_Accum"  --seed $i --clip 5 --T_max 3 --epoch 6 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 64 --epoch_switch 2 --num_encoders 5 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iterative"  --seed $i --clip 5 --T_max 3 --epoch 6 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 64 --epoch_switch 2 --num_encoders 5 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Weighted"  --seed $i --clip 5 --T_max 3 --epoch 6 --dataset "../../data/must" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 5 --hidden_size 64 --epoch_switch 2 --num_encoders 5 --weight_decay 0.0001 --learning_rate 0.000001
done

for i in 1 2 3 4 304
do
    python ../tav_nn.py --fusion "t_c" --sampler "Both"  --seed $i --clip 0.1 --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_c" --sampler "Both_NoAccum"  --seed $i --clip 0.1 --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_c" --sampler "Iter_Accum"  --seed $i --clip 0.1 --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_c" --sampler "Iterative"  --seed $i --clip 0.1 --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "t_c" --sampler "Weighted"  --seed $i --clip 0.1 --T_max 3 --epoch 10 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 8 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.001 --learning_rate 0.000001
done
