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


python ../tav_nn.py --fusion "d_c" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed 5 --beta 1 --clip 1  --mask false --sota false --T_max 2 --epoch 9 --model "MAE_encoder" --dataset "../../data/urfunny" --dropout 0.4 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 4 --output_dim 7 --hidden_size 1536 --lstm_layers 1 --text_column "text" --epoch_switch 2 --num_encoders 3 --weight_decay 0.001 --hidden_layers 512 --learning_rate 0.00001 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "d_c" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed 5 --beta 1 --clip 1  --mask false --sota false --T_max 2 --epoch 9 --model "MAE_encoder" --dataset "../../data/urfunny" --dropout 0.4 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 4 --output_dim 7 --hidden_size 1536 --lstm_layers 1 --text_column "text" --epoch_switch 2 --num_encoders 3 --weight_decay 0.001 --hidden_layers 512 --learning_rate 0.00001 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "d_c" --sampler "Weighted" --loss "CrossEntropy" --seed 5 --beta 1 --clip 1  --mask false --sota false --T_max 2 --epoch 9 --model "MAE_encoder" --dataset "../../data/urfunny" --dropout 0.4 --patience 14 --BertModel "roberta-large" --early_div false --input_dim 2 --batch_size 24 --label_task "humour" --num_layers 4 --output_dim 7 --hidden_size 1536 --lstm_layers 1 --text_column "text" --epoch_switch 2 --num_encoders 3 --weight_decay 0.001 --hidden_layers 512 --learning_rate 0.00001 --learn_PosEmbeddings false 
python ../tav_nn.py --fusion "dp_av" --sampler "Weighted" --loss "CrossEntropy" --seed 3 --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false

for i in 4 5
do
    python ../tav_nn.py --fusion "dp_av" --sampler "Both"  --loss "NewCrossEntropy" --seed $i --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Both_NoAccum"  --loss "NewCrossEntropy" --seed $i --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_av" --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1  --clip 0.1   --mask false  --sota false  --T_max 2  --epoch 10  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.4  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 7  --output_dim 7  --hidden_size 64  --lstm_layers 1  --text_column "text"  --epoch_switch 2  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
done



