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


for i in 72
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both"  --seed $i --clip 0.1 --T_max 3 --epoch 11  --dataset "../../data/urfunny" --dropout 0.4  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 7 --hidden_size 1536 --epoch_switch 3 --num_encoders 2 --weight_decay 0.001 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum"  --seed $i --clip 0.1 --T_max 3 --epoch 11  --dataset "../../data/urfunny" --dropout 0.4  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 7 --hidden_size 1536 --epoch_switch 3 --num_encoders 2 --weight_decay 0.001 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum" --seed $i --clip 0.1 --T_max 3 --epoch 11  --dataset "../../data/urfunny" --dropout 0.4  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 7 --hidden_size 1536 --epoch_switch 3 --num_encoders 2 --weight_decay 0.001 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --seed $i --clip 0.1 --T_max 3 --epoch 11  --dataset "../../data/urfunny" --dropout 0.4  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 7 --hidden_size 1536 --epoch_switch 3 --num_encoders 2 --weight_decay 0.001 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --seed $i --clip 0.1 --T_max 3 --epoch 11  --dataset "../../data/urfunny" --dropout 0.4  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 7 --hidden_size 1536 --epoch_switch 3 --num_encoders 2 --weight_decay 0.001 --learning_rate 0.00001
done

