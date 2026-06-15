#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

GPU_ID=${GPU_ID:-0}
CONDA_ENV=${CONDA_ENV:?Please set CONDA_ENV to the training environment}
DIRECTION=${DIRECTION:-U1652-D2S}
ALTITUDE=${ALTITUDE:-200}
DATA_ROOT=${DATA_ROOT:-/root/dataset}
DATASET_NAME=${DATASET_NAME:-SUES-200-512x512}

if [[ "${DIRECTION}" == "U1652-S2D" ]]; then
  DEFAULT_CKPT_PATH="${REPO_ROOT}/checkpoints/final/sues_s2d_${ALTITUDE}/weights_end.pth"
else
  DEFAULT_CKPT_PATH="${REPO_ROOT}/checkpoints/final/sues_d2s_${ALTITUDE}/weights_end.pth"
fi
CKPT_PATH=${CKPT_PATH:-${DEFAULT_CKPT_PATH}}

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/train_sues200.py" \
  --only_test True \
  --dataset "${DIRECTION}" \
  --altitude "${ALTITUDE}" \
  --data_folder "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --ckpt_path "${CKPT_PATH}" \
  --batch_size_eval 16 \
  --use_vrm True \
  --use_sam True \
  --use_csm True \
  --vcsa_residual_scale 0.001 \
  --record False \
  --verbose False
