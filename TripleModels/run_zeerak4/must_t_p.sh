#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/cpu_must.txt

# "t_c" 5.5 hours


for i in 1 2 3 4 1422
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both"  --seed $i --clip 1.0102019788831953 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.8423851638826877 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 16 --hidden_size 774 --epoch_switch 3 --num_encoders 3 --weight_decay 0.000595770184027987 --learning_rate 0.00004262760876741907
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum"  --seed $i --clip 1.0102019788831953 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.8423851638826877 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 16 --hidden_size 774 --epoch_switch 3 --num_encoders 3 --weight_decay 0.000595770184027987 --learning_rate 0.00004262760876741907
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum" --seed $i --clip 1.0102019788831953 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.8423851638826877 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 16 --hidden_size 774 --epoch_switch 3 --num_encoders 3 --weight_decay 0.000595770184027987 --learning_rate 0.00004262760876741907
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --seed $i --clip 1.0102019788831953 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.8423851638826877 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 16 --hidden_size 774 --epoch_switch 3 --num_encoders 3 --weight_decay 0.000595770184027987 --learning_rate 0.00004262760876741907
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --seed $i --clip 1.0102019788831953 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.8423851638826877 --patience 14 --batch_size 24 --early_stop "f1" --label_task "sarcasm" --num_layers 16 --hidden_size 774 --epoch_switch 3 --num_encoders 3 --weight_decay 0.000595770184027987 --learning_rate 0.00004262760876741907
done

