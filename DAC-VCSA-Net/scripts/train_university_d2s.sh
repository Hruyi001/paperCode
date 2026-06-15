#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

GPU_ID=${GPU_ID:-0}
CONDA_ENV=${CONDA_ENV:?Please set CONDA_ENV to the training environment}
DATA_ROOT=${DATA_ROOT:-/root/dataset}
DATASET_NAME=${DATASET_NAME:-University-Release}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/checkpoints/university-vcsa-d2s}
CHECKPOINT_START=${CHECKPOINT_START:-${REPO_ROOT}/checkpoint/pretrained_models/U1652/weights_end.pth}

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/train_university.py" \
  --dataset U1652-D2S \
  --data_folder "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --epochs 1 \
  --batch_size 4 \
  --batch_size_eval 128 \
  --weight_alignment 1.0 \
  --use_vrm True \
  --use_sam True \
  --use_csm True \
  --vcsa_residual_scale 0.05 \
  --checkpoint_start "${CHECKPOINT_START}" \
  --lr 0.00008 \
  --record False \
  --verbose False \
  --model_path "${MODEL_PATH}"
