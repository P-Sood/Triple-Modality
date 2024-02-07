#!/bin/bash

# Give job a name

##SBATCH -p gpu
#SBATCH -q cpu-512
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=urfunny_blackground.txt

# python make_videos_hdf5.py -d ../data/meld.pkl
#python make_videos_hdf5.py -d ../data/iemo.pkl
python make_videos_hdf5.py -d ../data/urfunny_small_combo.pkl
