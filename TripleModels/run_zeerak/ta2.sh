#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt


# "dp_ta" 10 hours

for i in 2 746
do
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both"  --clip 0.1  --seed $i --T_max 3 --epoch 10 --fusion "dp_ta" --dataset "../../data/urfunny" --dropout 0.5  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both_NoAccum"  --clip 0.1  --seed $i --T_max 3 --epoch 10 --fusion "dp_ta" --dataset "../../data/urfunny" --dropout 0.5  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iter_Accum" --clip 0.1  --seed $i --T_max 3 --epoch 10 --fusion "dp_ta" --dataset "../../data/urfunny" --dropout 0.5  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iterative" --clip 0.1  --seed $i --T_max 3 --epoch 10 --fusion "dp_ta" --dataset "../../data/urfunny" --dropout 0.5  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.000005
    python ../tav_nn.py --fusion "dp_ta" --sampler "Weighted" --clip 0.1  --seed $i --T_max 3 --epoch 10 --fusion "dp_ta" --dataset "../../data/urfunny" --dropout 0.5  --patience 14 --batch_size 32 --early_stop "acc" --label_task "humour" --num_layers 4 --hidden_size 256 --epoch_switch 2 --num_encoders 4 --weight_decay 0.01 --learning_rate 0.000005
done



