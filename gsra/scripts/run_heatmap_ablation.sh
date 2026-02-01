#!/usr/bin/env bash
set -euo pipefail

# One-click script to generate GSRA heatmaps only.
# This script automatically finds the GSRA checkpoint.

# --- Configuration: Direct paths ---
PROJECT_ROOT="/root/code/gsra"
DATASET="U1652-S2D"   # U1652-S2D or U1652-D2S
# Input images directory: should point to the root directory containing "test" subdirectory
# The script will automatically look for test/query_* and test/gallery_* subdirectories
INPUT_IMG_DIR="/root/exp/exp3/1.0/img"
# Output images directory: where the generated heatmap images will be saved
OUTPUT_IMG_DIR="${PROJECT_ROOT}/heatpmap_img"
NUM_PAIRS=3
IMG_SIZE=384
CHECKPOINT_BASE="${PROJECT_ROOT}/checkpoints"
# GSRA checkpoint path (leave empty to auto-find, or specify full path)
GSRA_CHECKPOINT="/root/code/gsra/checkpoint/best_score.pth"

# --- Find checkpoints automatically ---
# Only look for GSRA checkpoint
METHODS=("GSRA")

# Function to find the most recent checkpoint for a method
find_checkpoint() {
    local method=$1
    local dataset=$2
    local checkpoint_base=$3
    
    # Pattern: {dataset}_ConvNeXt_{method}_SALAD/*/best_score.pth
    # Find all directories matching the pattern
    local matching_dirs=$(find "${checkpoint_base}" -type d -path "*/${dataset}_ConvNeXt_${method}_SALAD/*" 2>/dev/null)
    
    if [ -z "${matching_dirs}" ]; then
        return 1
    fi
    
    # Find the most recent checkpoint directory (by modification time)
    local latest_dir=""
    local latest_time=0
    while IFS= read -r dir; do
        if [ -f "${dir}/best_score.pth" ]; then
            local dir_time=$(stat -c %Y "${dir}" 2>/dev/null || echo 0)
            if [ "${dir_time}" -gt "${latest_time}" ]; then
                latest_time=${dir_time}
                latest_dir="${dir}"
            fi
        fi
    done <<< "${matching_dirs}"
    
    if [ -z "${latest_dir}" ]; then
        return 1
    fi
    
    local ckpt_path="${latest_dir}/best_score.pth"
    if [ -f "${ckpt_path}" ]; then
        echo "${ckpt_path}"
        return 0
    fi
    
    return 1
}

# Build JSON with available checkpoints
VARIANTS_JSON="{"
first=true
for method in "${METHODS[@]}"; do
    # If GSRA_CHECKPOINT is specified, use it; otherwise try to find automatically
    if [ "${method}" = "GSRA" ] && [ -n "${GSRA_CHECKPOINT}" ] && [ -f "${GSRA_CHECKPOINT}" ]; then
        ckpt="${GSRA_CHECKPOINT}"
        echo "Using specified GSRA checkpoint: ${ckpt}"
    else
        ckpt=$(find_checkpoint "${method}" "${DATASET}" "${CHECKPOINT_BASE}" || echo "")
    fi
    
    if [ -n "${ckpt}" ] && [ -f "${ckpt}" ]; then
        if [ "$first" = true ]; then
            first=false
        else
            VARIANTS_JSON="${VARIANTS_JSON},"
        fi
        VARIANTS_JSON="${VARIANTS_JSON}\"${method}\": \"${ckpt}\""
        echo "Found checkpoint for ${method}: ${ckpt}"
    else
        echo "Warning: Checkpoint not found for ${method}, skipping..."
    fi
done
VARIANTS_JSON="${VARIANTS_JSON}}"

if [ "$VARIANTS_JSON" = "{}" ]; then
    echo "Error: No GSRA checkpoint found!"
    echo ""
    echo "Please either:"
    echo "  1. Set GSRA_CHECKPOINT variable in this script to point to your GSRA checkpoint"
    echo "     Example: GSRA_CHECKPOINT=\"/path/to/checkpoint/best_score.pth\""
    echo "  2. Or train a GSRA model first"
    echo ""
    echo "To train GSRA model, run:"
    echo "  python3 ${PROJECT_ROOT}/train_comparative_methods.py --attention GSRA --dataset ${DATASET} --model_path ${CHECKPOINT_BASE}"
    exit 1
fi

echo ""
echo "Using checkpoints:"
echo "${VARIANTS_JSON}" | python3 -m json.tool
echo ""

# Activate conda environment if available
if [ -f "/root/miniforge3/etc/profile.d/conda.sh" ]; then
    source /root/miniforge3/etc/profile.d/conda.sh
    conda activate gsra 2>/dev/null || true
fi

# Change to project root directory and set PYTHONPATH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/visualize/heatmap_ablation.py" \
  --dataset "${DATASET}" \
  --data_root "${INPUT_IMG_DIR}" \
  --num_pairs "${NUM_PAIRS}" \
  --img_size "${IMG_SIZE}" \
  --out_dir "${OUTPUT_IMG_DIR}" \
  --variant_checkpoints "${VARIANTS_JSON}"
