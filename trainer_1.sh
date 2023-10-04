#!/bin/bash
#SBATCH -A souvikg544
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 9
#SBATCH --nodelist gnode023
#SBATCH --time=24:00:00
#SBATCH --output=op_file.txt

source activate va
python trainer.py
