#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/mosei_fusion.txt


for i in 1 2 3 
do
    python ../tav_nn.py --fusion "d_c_tv"  --sampler "Both"  --loss "NewCrossEntropy" --seed $i --clip 0.1 --T_max 3 --epoch 7 --dataset "../../data/urfunny" --dropout 0.3 --patience 14 --batch_size 16 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_tv"  --sampler "Both_NoAccum"  --loss "NewCrossEntropy" --seed $i --clip 0.1 --T_max 3 --epoch 7 --dataset "../../data/urfunny" --dropout 0.3 --patience 14 --batch_size 16 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_tv"  --sampler "Iter_Accum"  --loss "WeightedCrossEntropy" --seed $i --clip 0.1 --T_max 3 --epoch 7 --dataset "../../data/urfunny" --dropout 0.3 --patience 14 --batch_size 16 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_tv"  --sampler "Iterative"  --loss "WeightedCrossEntropy" --seed $i --clip 0.1 --T_max 3 --epoch 7 --dataset "../../data/urfunny" --dropout 0.3 --patience 14 --batch_size 16 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "d_c_tv"  --sampler "Weighted"  --loss "CrossEntropy" --seed $i --clip 0.1 --T_max 3 --epoch 7 --dataset "../../data/urfunny" --dropout 0.3 --patience 14 --batch_size 16 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005
done






