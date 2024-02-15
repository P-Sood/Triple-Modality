#!/bin/bash

# Give job a name

#SBATCH -p gpu
#SBATCH -q gpu-8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --output=../outputs/whisper_start.txt

# python3 ../whisper_nn.py -d ../../data/urfunny --sampler Both --label_task humour --model "/l/users/zeerak.talat/TAV_Train/UrFunny_Audio1/z5c8hmm7/dandy-sweep-16/best.pt"
python3 ../whisper_nn.py -d ../../data/urfunny --sampler Both --label_task humour --model "/l/users/zeerak.talat/TAV_Train/UrFunny_Audio1/z5c8hmm7/polar-sweep-14/best.pt";
python3 ../whisper_nn.py -d ../../data/urfunny --sampler Both --label_task humour --model "/l/users/zeerak.talat/TAV_Train/UrFunny_Audio1/z5c8hmm7/spring-sweep-18/best.pt";
python3 ../whisper_nn.py -d ../../data/urfunny --sampler Both --label_task humour --model "/l/users/zeerak.talat/TAV_Train/UrFunny_Audio1/z5c8hmm7/genial-sweep-20/best.pt";
python3 ../whisper_nn.py -d ../../data/urfunny --sampler Both --label_task humour --model "/l/users/zeerak.talat/TAV_Train/UrFunny_Audio1/z5c8hmm7/grateful-sweep-19/best.pt"