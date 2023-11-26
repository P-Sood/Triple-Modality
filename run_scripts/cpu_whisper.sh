#!/bin/bash

# Give job a name

#SBATCH -p cpu
#SBATCH -q cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=meld_whisper.txt
#SBATCH --output=iemo_whisper.txt

python whisper_make_audios_hdf5 -d ../data/must.pkl
# python whisper_make_audios_hdf5 -d ../data/meld.pkl
# python whisper_make_audios_hdf5 -d ../data/iemo.pkl