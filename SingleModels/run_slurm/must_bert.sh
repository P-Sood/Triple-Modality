#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/videoMAE_fullseq.txt

# wandb agent ddi/Must_Text_Final_4_Steps/ownygcfg
python ../text_nn.py  --clip 1 --seed 154 --T_max 3 --epoch 6 --fusion "sota" --dataset "../../data/must" --dropout 0.5 --sampler "Iterative" --patience 14 --batch_size 8 --early_stop "f1" --label_task "sarcasm" --num_layers 12 --hidden_size 768 --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --learning_rate 0.000005