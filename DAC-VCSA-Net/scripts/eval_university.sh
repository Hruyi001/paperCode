#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

GPU_ID=${GPU_ID:-0}
CONDA_ENV=${CONDA_ENV:?Please set CONDA_ENV to the training environment}
DIRECTION=${DIRECTION:-U1652-D2S}
DATA_ROOT=${DATA_ROOT:-/root/dataset}
DATASET_NAME=${DATASET_NAME:-University-Release}

if [[ "${DIRECTION}" == "U1652-S2D" ]]; then
  DEFAULT_CKPT_PATH="${REPO_ROOT}/checkpoints/final/university_s2d/weights_end.pth"
else
  DEFAULT_CKPT_PATH="${REPO_ROOT}/checkpoints/final/university_d2s/weights_end.pth"
fi
CKPT_PATH=${CKPT_PATH:-${DEFAULT_CKPT_PATH}}

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/train_university.py" \
  --only_test True \
  --dataset "${DIRECTION}" \
  --data_folder "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --ckpt_path "${CKPT_PATH}" \
  --batch_size_eval 128 \
  --use_vrm True \
  --use_sam True \
  --use_csm True \
  --vcsa_residual_scale 0.05 \
  --record False \
  --verbose False
