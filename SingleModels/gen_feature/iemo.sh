#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/videoMAE_fullseq.txt

python ../tav_nn.py -e 1 -b 1 -d ../../data/more_text_IEMO -lt emotion --sampler Iterative --seed 36 # 4 for best, 