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

for i in 452
do
    python ../tav_nn.py --fusion "t_c" --sampler "Both"  --loss "CrossEntropy" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Both_NoAccum"  --loss "CrossEntropy" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "t_c" --sampler "Weighted" --loss "CrossEntropy" --seed $i --clip 1  --T_max 3 --epoch 12 --dataset "../../data/urfunny" --dropout 0.2 --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 5 --hidden_size 384 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.00001
done



