#!/bin/bash
#SBATCH -A souvikg544
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 9
#SBATCH --nodelist gnode023
#SBATCH --time=24:00:00
#SBATCH --output=op_file.txt

cd /ssd_scratch/cvit
mkdir souvikg544
cd souvikg544
mkdir -p checkpoints_landmarknet
rsync -rv --info=progress2 souvikg544@ada:/share3/souvikg544/datasets/gridcorpus .
cd /home2/souvikg544/souvik/exp2_l2s
conda activate base
conda activate va
python trainer.py