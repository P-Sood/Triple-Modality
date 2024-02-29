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

for i in 1 2 3 4 708
do
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both"  --seed $i --clip 1  --T_max 3 --epoch 8 --fusion "dp_ta" --dataset "../../data/more_text_IEMO" --dropout 0.6  --patience 14 --batch_size 32 --label_task "emotion" --early_stop "f1" --num_layers 8 --hidden_size 768 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both_NoAccum"  --seed $i --clip 1  --T_max 3 --epoch 8 --fusion "dp_ta" --dataset "../../data/more_text_IEMO" --dropout 0.6  --patience 14 --batch_size 32 --label_task "emotion" --early_stop "f1" --num_layers 8 --hidden_size 768 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iter_Accum" --seed $i --clip 1  --T_max 3 --epoch 8 --fusion "dp_ta" --dataset "../../data/more_text_IEMO" --dropout 0.6  --patience 14 --batch_size 32 --label_task "emotion" --early_stop "f1" --num_layers 8 --hidden_size 768 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iterative" --seed $i --clip 1  --T_max 3 --epoch 8 --fusion "dp_ta" --dataset "../../data/more_text_IEMO" --dropout 0.6  --patience 14 --batch_size 32 --label_task "emotion" --early_stop "f1" --num_layers 8 --hidden_size 768 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.00001
    python ../tav_nn.py --fusion "dp_ta" --sampler "Weighted" --seed $i --clip 1  --T_max 3 --epoch 8 --fusion "dp_ta" --dataset "../../data/more_text_IEMO" --dropout 0.6  --patience 14 --batch_size 32 --label_task "emotion" --early_stop "f1" --num_layers 8 --hidden_size 768 --epoch_switch 2 --num_encoders 2 --weight_decay 0.01 --learning_rate 0.00001
done