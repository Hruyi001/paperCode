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
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/checkpoints/sues200-vcsa-${DIRECTION}-${ALTITUDE}}
CHECKPOINT_START=${CHECKPOINT_START:-}

CMD=(
  conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/train_sues200.py"
  --dataset "${DIRECTION}"
  --altitude "${ALTITUDE}"
  --data_folder "${DATA_ROOT}"
  --dataset_name "${DATASET_NAME}"
  --epochs 1
  --batch_size 2
  --batch_size_eval 16
  --weight_alignment 0.4
  --use_vrm True
  --use_sam True
  --use_csm True
  --vcsa_residual_scale 0.001
  --lr 0.00002
  --record False
  --verbose False
  --model_path "${MODEL_PATH}"
)

if [[ -n "${CHECKPOINT_START}" ]]; then
  CMD+=(--checkpoint_start "${CHECKPOINT_START}")
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} "${CMD[@]}"
