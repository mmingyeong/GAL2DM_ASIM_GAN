#!/bin/bash
#SBATCH -J full_eval
#SBATCH -o logs/full_eval.%j.out
#SBATCH -e logs/full_eval.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH -t 0-12:00:00

module load cuda/11.8
source ~/.bashrc
conda activate torch

python full_metric.py
