#!/usr/bin/env bash
set -euo pipefail

# One-click script to generate Fig.4-style ablation heatmaps for GSRA and baselines.
# This script automatically finds checkpoints for all comparative methods.

# --- Configuration: Direct paths ---
PROJECT_ROOT="/root/code/gsra"
DATASET="U1652-S2D"   # U1652-S2D or U1652-D2S
DATA_ROOT="/root/dataset/University-Release/"
OUT_DIR="${PROJECT_ROOT}/heatpmap_img"
NUM_PAIRS=3
IMG_SIZE=384
CHECKPOINT_BASE="${PROJECT_ROOT}/checkpoints"

# --- Find checkpoints automatically ---
# Methods to look for (in order for display)
METHODS=("baseline" "CBAM" "CA" "ECA" "EMA" "GSRA")

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
    ckpt=$(find_checkpoint "${method}" "${DATASET}" "${CHECKPOINT_BASE}" || echo "")
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
    echo "Error: No checkpoints found! Please train models first."
    echo "Run: ${PROJECT_ROOT}/scripts/train_all_comparative_methods.sh"
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
  --data_root "${DATA_ROOT}" \
  --num_pairs "${NUM_PAIRS}" \
  --img_size "${IMG_SIZE}" \
  --out_dir "${OUT_DIR}" \
  --variant_checkpoints "${VARIANTS_JSON}"
