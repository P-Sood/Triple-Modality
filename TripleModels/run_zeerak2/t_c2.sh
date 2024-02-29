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

for i in 3 4 
do
    python ../tav_nn.py --fusion "t_c" --sampler "Both"  --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Both_NoAccum"  --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Iter_Accum" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Iterative" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Weighted" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
done



