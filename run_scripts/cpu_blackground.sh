#!/bin/bash

# Give job a name

#SBATCH -p cpu-8
#SBATCH -q cpu-
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=meld_blackground.txt
#SBATCH --output=iemo_blackground.txt

# python make_videos_hdf5.py -d ../data/meld.pkl
#python make_videos_hdf5.py -d ../data/iemo.pkl
python make_videos_hdf5.py -d ../data/must.pkl
