#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/iemo_fusion.txt

# "t_p" 8.5 hours
python ../tav_nn.py --fusion "t_p" --sampler "Both" --loss "NewCrossEntropy" --seed 2 --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum" --loss "NewCrossEntropy" --seed 2 --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed 4 --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --loss "CrossEntropy" --seed 4 --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 

for i in 1 5
do
    python ../tav_nn.py --fusion "t_p" --sampler "Both" --loss "NewCrossEntropy" --seed $i --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
    python ../tav_nn.py --fusion "t_p" --sampler "Both_NoAccum" --loss "NewCrossEntropy" --seed $i --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
    python ../tav_nn.py --fusion "t_p" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
    python ../tav_nn.py --fusion "t_p" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
    python ../tav_nn.py --fusion "t_p" --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1 --clip 1  --mask false --sota false --T_max 3 --epoch 4 --model "MAE_encoder"  --dataset "../../data/urfunny" --dropout 0.2  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 8 --output_dim 7 --hidden_size 128 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.0001 --hidden_layers 512 --learning_rate 0.00005 --learn_PosEmbeddings false 
done

