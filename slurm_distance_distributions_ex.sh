#!/bin/bash

#SBATCH --mem=300G
#SBATCH --partition=[INSERT]
#SBATCH --gres=gpu:1
#SBATCH --error=/home/[INSERT]
#SBATCH --output=/home/[INSERT]
#SBATCH --job-name=distance-distributions

source ~/.bashrc


cd /home/[INSERT]
echo $SLURMD_NODENAME
conda activate hubs-freq-tokens

python run.py distance-distributions --output-folder=results --vocab-main-folder=[INSERT] --cuda 

