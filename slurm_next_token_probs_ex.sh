#!/bin/bash

#SBATCH --mem=100G
#SBATCH --partition=[INSERT]
#SBATCH --gres=gpu:1
#SBATCH --error=/home/[INSERT]
#SBATCH --output=/home/[INSERT]
#SBATCH --job-name=save-next-token-probs 

source ~/.bashrc


cd /home/[INSERT]
echo $SLURMD_NODENAME
conda activate hubs-freq-tokens

python run.py save-next-token-probs --output-folder=results --vocab-main-folder=[INSERT] --all-emb-folder=[INSERT] --cuda
