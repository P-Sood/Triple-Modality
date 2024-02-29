#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

# "t_c" 5.5 hours

for i in 1 2 3 4 953
do
    python ../tav_nn.py --fusion "t_c" --sampler "Both"    --seed $i --clip 5 --T_max 3 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.5 --patience 14 --batch_size 32 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --learning_rate 0.00005 
    python ../tav_nn.py --fusion "t_c" --sampler "Both_NoAccum"    --seed $i --clip 5 --T_max 3 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.5 --patience 14 --batch_size 32 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --learning_rate 0.00005 
    python ../tav_nn.py --fusion "t_c" --sampler "Iter_Accum"   --seed $i --clip 5 --T_max 3 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.5 --patience 14 --batch_size 32 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --learning_rate 0.00005 
    python ../tav_nn.py --fusion "t_c" --sampler "Iterative"   --seed $i --clip 5 --T_max 3 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.5 --patience 14 --batch_size 32 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --learning_rate 0.00005 
    python ../tav_nn.py --fusion "t_c" --sampler "Weighted"   --seed $i --clip 5 --T_max 3 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.5 --patience 14 --batch_size 32 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --learning_rate 0.00005 
done


wandb agent ddi/Must_Tri_Acc_Final/vazpjoco
