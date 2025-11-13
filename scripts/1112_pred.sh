#!/bin/bash
#SBATCH -J cgan_predict
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_GAN/logs/slurm/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_GAN/logs/slurm/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

#set -euo pipefail

# -------- Environment --------
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

# -------- Paths (single source of truth) --------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_GAN"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
SLURM_DIR="${PROJECT_ROOT}/logs/slurm"

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
OUT_DIR_BASE="${PROJECT_ROOT}/results/cgan_predictions/${RUN_ID}"
mkdir -p "${SLURM_DIR}" "${LOG_DIR}" "${OUT_DIR_BASE}"

# Logger will only write under PROJECT_ROOT/logs/RUN_ID
export GAL2DM_LOGDIR="${LOG_DIR}"

# Always run from project root
cd "${PROJECT_ROOT}"

# -------- System info --------
echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch, os, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA (Torch):", getattr(torch.version, "cuda", None))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
print("GAL2DM_LOGDIR:", os.environ.get("GAL2DM_LOGDIR"))
PY
nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

# -------- Checkpoint autodetect --------
# You can export MODEL_PATH or CKPT_DIR before sbatch to override this logic.
if [ -z "${MODEL_PATH:-}" ]; then
  # If CKPT_DIR is provided, use it. Otherwise pick the newest run under results/cgan/*
  if [ -z "${CKPT_DIR:-}" ]; then
    CKPT_DIR="$(ls -dt ${PROJECT_ROOT}/results/cgan/* 2>/dev/null | head -n 1 || true)"
    if [ -z "${CKPT_DIR}" ]; then
      echo "[ERROR] No checkpoint directory found under ${PROJECT_ROOT}/results/cgan/"
      exit 1
    fi
  fi
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR not found: ${CKPT_DIR}"
    exit 1
  fi
  # Prefer *best*.pt if present
  BEST_CKPT="$(ls -t ${CKPT_DIR}/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -n "${BEST_CKPT}" ]; then
    MODEL_PATH="${BEST_CKPT}"
  else
    MODEL_PATH="$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -n 1 || true)"
  fi
  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No .pt file found in ${CKPT_DIR}"
    exit 1
  fi
fi
echo "[INFO] Using checkpoint: ${MODEL_PATH}"

# Output subdir name mirrors checkpoint run stem
RUN_STEM="$(basename "$(dirname "${MODEL_PATH}")")"
PRED_OUT_DIR="${OUT_DIR_BASE}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

# -------- Inference configuration --------
BATCH_SIZE="${BATCH_SIZE:-1}"
AMP_FLAG="--amp"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"

# input_case/keep_two_channels는 GAN 예측 코드의 기본값이 이미 'both' & keep_two=false라
# 꼭 필요할 때만 아래 두 줄을 활성화하세요.
# EXTRA_INPUT_FLAGS="--input_case both --keep_two_channels"
EXTRA_INPUT_FLAGS=""

# Single console log under PROJECT_ROOT/logs/RUN_ID
MODEL_NAME="cgan_pix2pixcc3d"
CONSOLE_LOG="${LOG_DIR}/${MODEL_NAME}_predict.log"
touch "${CONSOLE_LOG}"

# -------- Run prediction (GAN version) --------
# If your predict script path differs, adjust to the correct module/file:
#  - module form: python -m src.predict_gan
#  - file path  : python src/predict_gan.py
srun --ntasks=1 python -u -m src.predict \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "cuda" \
  --batch_size "${BATCH_SIZE}" \
  --sample_fraction "${SAMPLE_FRACTION}" \
  ${AMP_FLAG} \
  ${EXTRA_INPUT_FLAGS} \
  2>&1 | tee -a "${CONSOLE_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
