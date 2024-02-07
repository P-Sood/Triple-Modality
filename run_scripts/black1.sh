#!/bin/bash

# Give job a name

##SBATCH -p gpu
#SBATCH -q cpu-512
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=gpu_black.txt


python3 body_face.py -d ../data/urfunny.pkl -e False --beg 4 --end 5