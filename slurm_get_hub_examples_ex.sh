#!/bin/bash

#SBATCH --mem=100G
#SBATCH --partition=[INSERT]
#SBATCH --gres=gpu:1
#SBATCH --error=/home/[INSERT]
#SBATCH --output=/home/[INSERT]
#SBATCH --job-name=get-hub-examples

source ~/.bashrc


cd /home/[INSERT]
echo $SLURMD_NODENAME
conda activate hubs-freq-tokens

python run.py get-hub-examples --output-folder=results --vocab-main-folder=[INSERT] --copora-folder=[INSERT]

