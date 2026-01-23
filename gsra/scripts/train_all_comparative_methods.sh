#!/usr/bin/env bash
set -euo pipefail

# Script to train all comparative attention methods for ablation study
# Methods: baseline, CBAM, CA, ECA, EMA

PROJECT_ROOT="/root/code/gsra"
DATASET="U1652-S2D"  # or U1652-D2S
DATA_ROOT="/root/dataset/University-Release"
EPOCHS=40
BATCH_SIZE=32
LR=0.0001
GPU_IDS=(0)

# Activate conda environment if available
if [ -f "/root/miniforge3/etc/profile.d/conda.sh" ]; then
    source /root/miniforge3/etc/profile.d/conda.sh
    conda activate gsra 2>/dev/null || true
fi

# Change to project root directory and set PYTHONPATH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Methods to train
METHODS=("baseline" "CBAM" "CA" "ECA" "EMA")

echo "=========================================="
echo "Training Comparative Attention Methods"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Data Root: ${DATA_ROOT}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Methods: ${METHODS[@]}"
echo "=========================================="
echo ""

# Train each method
for method in "${METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: ${method}"
    echo "=========================================="
    echo "Start time: $(date)"
    echo ""
    
    python3 "${PROJECT_ROOT}/train_comparative_methods.py" \
        --attention "${method}" \
        --dataset "${DATASET}" \
        --data_folder "${DATA_ROOT}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --gpu_ids "${GPU_IDS[@]}" \
        --model_path "${PROJECT_ROOT}/checkpoints"
    
    echo ""
    echo "Completed: ${method}"
    echo "End time: $(date)"
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved in: ${PROJECT_ROOT}/checkpoints/"
echo ""
echo "To generate heatmaps, run:"
echo "  ${PROJECT_ROOT}/scripts/run_heatmap_ablation.sh"
