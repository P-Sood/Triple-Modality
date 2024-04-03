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


for i in 149
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both"  --seed $i --clip 3.7434379781749527 --T_max 2 --epoch 16 --dataset "../../data/urfunny" --dropout 0.3437044122742988 --patience 14 --batch_size 8 --early_stop "acc" --label_task "humour" --num_layers 11 --hidden_size 1219 --epoch_switch 3 --num_encoders 5 --weight_decay 0.00718668007102603 --learning_rate 0.00003216023106367913
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum"  --seed $i --clip 3.7434379781749527 --T_max 2 --epoch 16 --dataset "../../data/urfunny" --dropout 0.3437044122742988 --patience 14 --batch_size 8 --early_stop "acc" --label_task "humour" --num_layers 11 --hidden_size 1219 --epoch_switch 3 --num_encoders 5 --weight_decay 0.00718668007102603 --learning_rate 0.00003216023106367913
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum" --seed $i --clip 3.7434379781749527 --T_max 2 --epoch 16 --dataset "../../data/urfunny" --dropout 0.3437044122742988 --patience 14 --batch_size 8 --early_stop "acc" --label_task "humour" --num_layers 11 --hidden_size 1219 --epoch_switch 3 --num_encoders 5 --weight_decay 0.00718668007102603 --learning_rate 0.00003216023106367913
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --seed $i --clip 3.7434379781749527 --T_max 2 --epoch 16 --dataset "../../data/urfunny" --dropout 0.3437044122742988 --patience 14 --batch_size 8 --early_stop "acc" --label_task "humour" --num_layers 11 --hidden_size 1219 --epoch_switch 3 --num_encoders 5 --weight_decay 0.00718668007102603 --learning_rate 0.00003216023106367913
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --seed $i --clip 3.7434379781749527 --T_max 2 --epoch 16 --dataset "../../data/urfunny" --dropout 0.3437044122742988 --patience 14 --batch_size 8 --early_stop "acc" --label_task "humour" --num_layers 11 --hidden_size 1219 --epoch_switch 3 --num_encoders 5 --weight_decay 0.00718668007102603 --learning_rate 0.00003216023106367913
done

