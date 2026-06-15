#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
CONDA_ENV=${CONDA_ENV:?Please set CONDA_ENV to the training environment}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CKPT_PATH=${CKPT_PATH:-${REPO_ROOT}/checkpoints/university-close-d2s-drop5/convnext_base.fb_in22k_ft_in1k_384/0613202924/weights_end.pth}
DATASET_PATH=${DATASET_PATH:-/root/dataset/University-Release}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/heatmap_img_vcsa_net}
NUM_SAMPLES=${NUM_SAMPLES:-3}
IMG_SIZE=${IMG_SIZE:-384}
NCLASSES=${NCLASSES:-701}
BLOCK=${BLOCK:-2}
TRIPLET_LOSS=${TRIPLET_LOSS:-0.3}
DEVICE=${DEVICE:-cuda}
SAMPLE_IDS=${SAMPLE_IDS:-}

mkdir -p "${OUTPUT_DIR}"
export VCSA_NET_REPO_ROOT="${REPO_ROOT}"

CMD=(
  conda run -n "${CONDA_ENV}" python "${SCRIPT_DIR}/generate_heatmap.py"
  --ckpt_path "${CKPT_PATH}"
  --dataset_path "${DATASET_PATH}"
  --num_samples "${NUM_SAMPLES}"
  --img_size "${IMG_SIZE}"
  --output_dir "${OUTPUT_DIR}"
  --nclasses "${NCLASSES}"
  --block "${BLOCK}"
  --triplet_loss "${TRIPLET_LOSS}"
  --device "${DEVICE}"
)

if [[ -n "${SAMPLE_IDS}" ]]; then
  CMD+=(--sample_ids ${SAMPLE_IDS})
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Error: checkpoint file not found: ${CKPT_PATH}"
  echo "Please set CKPT_PATH to a valid weights_end.pth file."
  exit 1
fi

echo "Starting VCSA-Net heatmap generation..."
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} "${CMD[@]}"

echo ""
echo "Heatmap generation completed successfully."
echo "Results saved to: ${OUTPUT_DIR}"
