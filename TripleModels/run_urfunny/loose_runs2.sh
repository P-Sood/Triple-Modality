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

for i in 5
do
    python ../tav_nn.py --fusion "t_c" --sampler "Both"  --loss "CrossEntropy" --seed $i --beta 1  --clip 1  --mask false  --sota false  --T_max 2  --epoch 9  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.3  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 3  --output_dim 7  --hidden_size 384  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "t_c" --sampler "Both_NoAccum"  --loss "CrossEntropy" --seed $i --beta 1  --clip 1  --mask false  --sota false  --T_max 2  --epoch 9  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.3  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 3  --output_dim 7  --hidden_size 384  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "t_c" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 1  --mask false  --sota false  --T_max 2  --epoch 9  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.3  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 3  --output_dim 7  --hidden_size 384  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "t_c" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 1  --mask false  --sota false  --T_max 2  --epoch 9  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.3  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 3  --output_dim 7  --hidden_size 384  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "t_c" --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1  --clip 1  --mask false  --sota false  --T_max 2  --epoch 9  --model "MAE_encoder"  --dataset "../../data/urfunny"  --dropout 0.3  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 32  --label_task "humour"  --num_layers 3  --output_dim 7  --hidden_size 384  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.01  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
done

for i in 5
do
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both"  --loss "NewCrossEntropy" --seed $i --beta 1  --clip 1  --mask false --sota false  --T_max 2  --epoch 11  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 5  --output_dim 7  --hidden_size 256  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.0001  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_ta" --sampler "Both_NoAccum"  --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 1  --mask false --sota false  --T_max 2  --epoch 11  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 5  --output_dim 7  --hidden_size 256  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.0001  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iter_Accum" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 1  --mask false --sota false  --T_max 2  --epoch 11  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 5  --output_dim 7  --hidden_size 256  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.0001  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_ta" --sampler "Iterative" --loss "WeightedCrossEntropy" --seed $i --beta 1  --clip 1  --mask false --sota false  --T_max 2  --epoch 11  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 5  --output_dim 7  --hidden_size 256  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.0001  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
    python ../tav_nn.py --fusion "dp_ta" --sampler "Weighted" --loss "CrossEntropy" --seed $i --beta 1  --clip 1  --mask false --sota false  --T_max 2  --epoch 11  --model "MAE_encoder"   --dataset "../../data/urfunny"  --dropout 0.2  --patience 14  --BertModel "roberta-large"  --early_div false  --input_dim 2  --batch_size 24  --label_task "humour"  --num_layers 5  --output_dim 7  --hidden_size 256  --lstm_layers 1  --text_column "text"  --epoch_switch 3  --num_encoders 3  --weight_decay 0.0001  --hidden_layers 512  --learning_rate 0.000005  --learn_PosEmbeddings false
done


