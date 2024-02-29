#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


for i in 1 2 3 4 305
do
    python ../tav_nn.py --fusion "d_c" --sampler "Both"  --seed $i --clip 5  --T_max 3 --epoch 7 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 1 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.0001
    python ../tav_nn.py --fusion "d_c" --sampler "Both_NoAccum"  --seed $i --clip 5  --T_max 3 --epoch 7 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 1 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.0001
    python ../tav_nn.py --fusion "d_c" --sampler "Iter_Accum"  --seed $i --clip 5  --T_max 3 --epoch 7 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 1 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.0001
    python ../tav_nn.py --fusion "d_c" --sampler "Iterative"  --seed $i --clip 5  --T_max 3 --epoch 7 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 1 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.0001
    python ../tav_nn.py --fusion "d_c" --sampler "Weighted"  --seed $i --clip 5  --T_max 3 --epoch 7 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 1 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.0001
done

wandb agent ddi/Must_Tri_Acc_Final/vazpjoco
