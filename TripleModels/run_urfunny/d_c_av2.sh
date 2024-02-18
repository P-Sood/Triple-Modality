#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/mosei_fusion.txt

for i in 3
do
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Both"  --loss "NewCrossEntropy" --seed $i  --beta 1  --clip 0.1  --mask false   --sota false  --T_max 3  --epoch 8  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 8  --output_dim 7  --hidden_size 128  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.00005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Both_NoAccum"  --loss "CrossEntropy" --seed $i  --beta 1  --clip 0.1  --mask false   --sota false  --T_max 3  --epoch 8  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 8  --output_dim 7  --hidden_size 128  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.00005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Iter_Accum"  --loss "WeightedCrossEntropy" --seed $i  --beta 1  --clip 0.1  --mask false   --sota false  --T_max 3  --epoch 8  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 8  --output_dim 7  --hidden_size 128  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.00005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Iterative"  --loss "WeightedCrossEntropy" --seed $i  --beta 1  --clip 0.1  --mask false   --sota false  --T_max 3  --epoch 8  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 8  --output_dim 7  --hidden_size 128  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.00005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "d_c_av"  --sampler "Weighted"  --loss "CrossEntropy" --seed $i  --beta 1  --clip 0.1  --mask false   --sota false  --T_max 3  --epoch 8  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 8  --output_dim 7  --hidden_size 128  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.00005  --learn_PosEmbeddings false
done