#!/bin/bash

# Give job a name

##SBATCH -p gpu
#SBATCH -q cpu-512
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/cpu_iemo.txt

python ../tav_nn.py -e 1 -b 1 -d ../../data/iemo -lt emotion --sampler Iterative --seed 64
