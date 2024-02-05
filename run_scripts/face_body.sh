#!/bin/bash


# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=gpu_video.txt

module load opencv-gpu 
module load opencv-4
python3 body_face.py -d ../data/urfunny.pkl -e True