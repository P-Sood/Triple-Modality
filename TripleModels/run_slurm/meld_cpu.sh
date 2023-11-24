#!/bin/bash

# Give job a name

#SBATCH --time 00-12:00 # time (DD-HH:MM)
#SBATCH -p cpu
#SBATCH -q cpu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=../outputs/meld_triple_test.txt

#wandb agent ddi/MeldTriple/jtzs474d

