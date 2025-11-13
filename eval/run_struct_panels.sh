#!/bin/bash
#SBATCH -J struct_panels
#SBATCH -o logs/struct_panels.%j.out
#SBATCH -e logs/struct_panels.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 0-06:00:00

module purge
module load cuda/11.8
source ~/.bashrc
conda activate torch

PY=make_structural_panels.py
OUT=./struct_outputs

mkdir -p "$OUT" logs

srun python "$PY" \
  --base_dir "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions" \
  --unet_dir "/home/mingyeong/GAL2DM_ASIM_VNET/results/unet_predictions/28845/icase-both-keep2" \
  --vit_dir  "/home/mingyeong/GAL2DM_ASIM_ViT/results/vit_predictions/28846/icase-both" \
  --truth_tpl "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/{idx}.hdf5" \
  --outdir "$OUT"
