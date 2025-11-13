#!/bin/bash
#SBATCH -J cgan_train_ep50
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_GAN/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_GAN/logs/slurm/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

#set -euo pipefail
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

# ---- Absolute project paths (single source of truth) ----
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_GAN"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# All outputs live under PROJECT_ROOT only
CKPT_DIR="${PROJECT_ROOT}/results/cgan/${RUN_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
SLURM_DIR="${PROJECT_ROOT}/logs/slurm"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${SLURM_DIR}"

# ---- Environment ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
export GAL2DM_LOGDIR="${LOG_DIR}"        # logger가 참조할 유일한 로그 디렉토리
export NCCL_P2P_DISABLE=1
ulimit -n 65535

# 실행 위치에 상관없이 항상 프로젝트 루트에서 실행
cd "${PROJECT_ROOT}"

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
which python

python - <<'PY'
import torch, platform, os
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA (Torch):", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
PY

nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

# ---- Unified console log under PROJECT_ROOT/logs/RUN_ID ----
EPOCHS=50
MODEL_NAME="cgan_pix2pixcc3d"
CONSOLE_LOG="${LOG_DIR}/${MODEL_NAME}_ep${EPOCHS}.console.log"
touch "${CONSOLE_LOG}"

echo "Launching training..."
srun --ntasks=1 python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split 0.8 \
  --sample_fraction 1.0 \
  --batch_size 2 \
  --num_workers 6 \
  --epochs ${EPOCHS} \
  --min_lr 1e-4 \
  --max_lr 1e-3 \
  --cycle_length 8 \
  --ckpt_dir "${CKPT_DIR}" \
  --seed 42 \
  --device cuda \
  --keep_two_channels \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/etc/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${CONSOLE_LOG}"

echo "=== [JOB FINISHED] $(date) ==="
