#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


for i in 1 2 3 4 664
do
    python ../tav_nn.py --fusion "d_c_av" --sampler "Both"  --seed $i --clip 0.1 --T_max 2 --epoch 9 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "f1" --label_task "sarcasm" --num_layers 6 --hidden_size 256 --epoch_switch 3 --num_encoders 6 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_av" --sampler "Both_NoAccum"  --seed $i --clip 0.1 --T_max 2 --epoch 9 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "f1" --label_task "sarcasm" --num_layers 6 --hidden_size 256 --epoch_switch 3 --num_encoders 6 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_av" --sampler "Iter_Accum"  --seed $i --clip 0.1 --T_max 2 --epoch 9 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "f1" --label_task "sarcasm" --num_layers 6 --hidden_size 256 --epoch_switch 3 --num_encoders 6 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_av" --sampler "Iterative"  --seed $i --clip 0.1 --T_max 2 --epoch 9 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "f1" --label_task "sarcasm" --num_layers 6 --hidden_size 256 --epoch_switch 3 --num_encoders 6 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_av" --sampler "Weighted"  --seed $i --clip 0.1 --T_max 2 --epoch 9 --dataset "../../data/must" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "f1" --label_task "sarcasm" --num_layers 6 --hidden_size 256 --epoch_switch 3 --num_encoders 6 --weight_decay 0.01 --learning_rate 0.000005
done

for i in 1 2 3 4 189
do
    python ../tav_nn.py --fusion "dp_av" --sampler "Both"  --seed $i --clip 0.1 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 16 --early_stop "f1" --label_task "sarcasm" --num_layers 4 --hidden_size 1536 --epoch_switch 3 --num_encoders 6 --weight_decay 0.001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_av" --sampler "Both_NoAccum"  --seed $i --clip 0.1 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 16 --early_stop "f1" --label_task "sarcasm" --num_layers 4 --hidden_size 1536 --epoch_switch 3 --num_encoders 6 --weight_decay 0.001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_av" --sampler "Iter_Accum"  --seed $i --clip 0.1 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 16 --early_stop "f1" --label_task "sarcasm" --num_layers 4 --hidden_size 1536 --epoch_switch 3 --num_encoders 6 --weight_decay 0.001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_av" --sampler "Iterative"  --seed $i --clip 0.1 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 16 --early_stop "f1" --label_task "sarcasm" --num_layers 4 --hidden_size 1536 --epoch_switch 3 --num_encoders 6 --weight_decay 0.001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_av" --sampler "Weighted"  --seed $i --clip 0.1 --T_max 3 --epoch 8 --dataset "../../data/must" --dropout 0.5 --patience 14 --batch_size 16 --early_stop "f1" --label_task "sarcasm" --num_layers 4 --hidden_size 1536 --epoch_switch 3 --num_encoders 6 --weight_decay 0.001 --learning_rate 0.000005
done

wandb agent ddi/Must_Tri_Acc_Final/vazpjoco