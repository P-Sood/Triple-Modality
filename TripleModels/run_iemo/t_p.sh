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

# for i in 1 2 3 4 
# do
    # python ../tav_nn.py --sampler "Both" --loss "NewCrossEntropy" --seed $i --beta 1 --clip 5 --mask false  --sota false --T_max 2 --epoch 9 --model "MAE_encoder"  --dataset "../../data/iemo" --dropout 0.5  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 5 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false --fusion "t_p"
    # python ../tav_nn.py --sampler "Both" --loss "NewCrossEntropy" --seed $i --beta 1 --clip 5 --mask false --sota false --T_max 3 --epoch 6 --model "MAE_encoder" --dataset "../../data/iemo" --dropout 0.33091313389138777 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "emotion" --num_layers 7 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 4 --weight_decay 0.0013071205870598566 --hidden_layers 512 --learning_rate 0.0000013200424426947532 --learn_PosEmbeddings true 
    # python ../tav_nn.py --sampler "Both_NoAccum" --loss "CrossEntropy" --seed $i --beta 1 --clip 5 --mask false  --sota false --T_max 2 --epoch 9 --model "MAE_encoder"  --dataset "../../data/iemo" --dropout 0.5  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 5 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false --fusion "t_p"
    # python ../tav_nn.py --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 5 --mask false  --sota false --T_max 2 --epoch 9 --model "MAE_encoder"  --dataset "../../data/iemo" --dropout 0.5  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 5 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false --fusion "t_p"
    # python ../tav_nn.py --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1 --clip 5 --mask false  --sota false --T_max 2 --epoch 9 --model "MAE_encoder"  --dataset "../../data/iemo" --dropout 0.5  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 5 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false --fusion "t_p"
    # python ../tav_nn.py --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1 --clip 5 --mask false  --sota false --T_max 2 --epoch 9 --model "MAE_encoder"  --dataset "../../data/iemo" --dropout 0.5  --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 32 --label_task "emotion" --num_layers 5 --output_dim 7 --hidden_size 384 --lstm_layers 1 --text_column "text" --epoch_switch 3 --num_encoders 3 --weight_decay 0.00001 --hidden_layers 512 --learning_rate 0.000005 --learn_PosEmbeddings false --fusion "t_p"
# done

python ../tav_nn.py --sampler "Both" --loss "NewCrossEntropy" --seed 101 --clip 1 --T_max 2 --epoch 11 --fusion "t_p" --dataset "../../data/iemo" --dropout 0.6 --patience 14 --batch_size 32 --label_task "emotion" --num_layers 3 --hidden_size 1536 --epoch_switch 2 --num_encoders 2 --weight_decay 0.00001 --learning_rate 0.000005

