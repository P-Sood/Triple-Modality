#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


# "d_c" 4.5 hours

for i in 1 2 
do
    python ../tav_nn.py --fusion "d_c" --sampler "Both" --loss "WeightedCrossEntropy" --seed $i --clip 0.1  --T_max 2 --epoch 9  --dataset "../../data/urfunny" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "acc" --label_task "humour" --num_layers 2 --hidden_size 384 --epoch_switch 2 --num_encoders 3 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "d_c" --sampler "Both_NoAccum" --loss "WeightedCrossEntropy" --seed $i --clip 0.1  --T_max 2 --epoch 9  --dataset "../../data/urfunny" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "acc" --label_task "humour" --num_layers 2 --hidden_size 384 --epoch_switch 2 --num_encoders 3 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "d_c" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --clip 0.1  --T_max 2 --epoch 9  --dataset "../../data/urfunny" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "acc" --label_task "humour" --num_layers 2 --hidden_size 384 --epoch_switch 2 --num_encoders 3 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "d_c" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --clip 0.1  --T_max 2 --epoch 9  --dataset "../../data/urfunny" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "acc" --label_task "humour" --num_layers 2 --hidden_size 384 --epoch_switch 2 --num_encoders 3 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "d_c" --sampler "Weighted" --loss "CrossEntropy" --seed $i --clip 0.1  --T_max 2 --epoch 9  --dataset "../../data/urfunny" --dropout 0.5 --patience 14 --batch_size 24 --early_stop "acc" --label_task "humour" --num_layers 2 --hidden_size 384 --epoch_switch 2 --num_encoders 3 --weight_decay 0.01 --learning_rate 0.00001
done




