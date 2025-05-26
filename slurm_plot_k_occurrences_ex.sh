#!/bin/bash

#SBATCH --mem=150G
#SBATCH --partition=[INSERT]
#SBATCH --gres=gpu:1
#SBATCH --error=/home/[INSERT]
#SBATCH --output=/home/[INSERT]
#SBATCH --job-name=plot-k-occurrence

source ~/.bashrc


cd /home/[INSERT]
echo $SLURMD_NODENAME
conda activate hubs-freq-tokens

python run.py plot-k-occurrence --output-folder=results --vocab-main-folder=[INSERT] --cuda

