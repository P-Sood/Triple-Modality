#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/cpu_iemo_tp.txt

# "t_c" 5.5 hours

python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --seed 4 --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005

for i in 131
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both"  --seed $i --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum"  --seed $i --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum" --seed $i --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --seed $i --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --seed $i --clip 1 --T_max 2 --epoch 11 --dataset "../../data/more_text_IEMO" --dropout 0.6 --patience 14 --batch_size 24 --early_stop "f1" --label_task "emotion" --num_layers 6 --hidden_size 1536 --epoch_switch 2 --num_encoders 1 --weight_decay 0.0001 --learning_rate 0.000005
done

