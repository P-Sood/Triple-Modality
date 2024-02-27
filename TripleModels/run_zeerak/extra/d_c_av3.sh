#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/mosei_fusion.txt


# 51 minutes * 25

for i in 160
do
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Both"  --clip 1 --seed $i --T_max 2 --epoch 12 --dataset "../../data/urfunny" --dropout 0.3 --sampler "Weighted" --patience 14 --batch_size 4 --early_stop "acc" --label_task "humour" --num_layers 8 --hidden_size 768 --epoch_switch 3 --num_encoders 2 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Both_NoAccum"  --clip 1 --seed $i --T_max 2 --epoch 12 --dataset "../../data/urfunny" --dropout 0.3 --sampler "Weighted" --patience 14 --batch_size 4 --early_stop "acc" --label_task "humour" --num_layers 8 --hidden_size 768 --epoch_switch 3 --num_encoders 2 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Iter_Accum"  --clip 1 --seed $i --T_max 2 --epoch 12 --dataset "../../data/urfunny" --dropout 0.3 --sampler "Weighted" --patience 14 --batch_size 4 --early_stop "acc" --label_task "humour" --num_layers 8 --hidden_size 768 --epoch_switch 3 --num_encoders 2 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Iterative"  --clip 1 --seed $i --T_max 2 --epoch 12 --dataset "../../data/urfunny" --dropout 0.3 --sampler "Weighted" --patience 14 --batch_size 4 --early_stop "acc" --label_task "humour" --num_layers 8 --hidden_size 768 --epoch_switch 3 --num_encoders 2 --weight_decay 0.0001 --learning_rate 0.000001
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Weighted"  --clip 1 --seed $i --T_max 2 --epoch 12 --dataset "../../data/urfunny" --dropout 0.3 --sampler "Weighted" --patience 14 --batch_size 4 --early_stop "acc" --label_task "humour" --num_layers 8 --hidden_size 768 --epoch_switch 3 --num_encoders 2 --weight_decay 0.0001 --learning_rate 0.000001
done

