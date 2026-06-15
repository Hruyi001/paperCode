#!/bin/bash

# Script to generate DSA ablation heatmaps
# This script generates heatmaps comparing models with and without DSA module
# Usage: bash run_generate_dsa_heatmap.sh
#
# Before running, please modify the following paths according to your setup:
# - CKPT_PATH: Path to your trained model checkpoint
# - DATASET_PATH: Path to your dataset root directory
#   Expected structure:
#     DATASET_PATH/
#       test/
#         gallery_satellite/
#           <sample_id>/
#             <image>.jpg
#         query_drone/
#           <sample_id>/
#             <image>.jpg

# ============================================================================
# Configuration - MODIFY THESE PATHS ACCORDING TO YOUR SETUP
# ============================================================================

# Path to trained model checkpoint
# Default: checkpoint/pretrained_models/U1652/weights_end.pth
# You can also use other checkpoint paths if available
CKPT_PATH="checkpoint/pretrained_models/U1652/weights_end.pth"

# Path to dataset root directory (e.g., /path/to/U1652)
# The script will automatically search for test/gallery_satellite and test/query_drone
DATASET_PATH="/root/dataset/University-Release"

# Output directory for generated heatmaps
OUTPUT_DIR="./heatmap_img"

# Number of samples to visualize (if sample_ids not specified)
# Set to a large number or use generate_all_heatmaps.py for all images
NUM_SAMPLES=999999

# Image size (should match training image size)
IMG_SIZE=384

# Model configuration (should match training configuration)
NCLASSES=701
BLOCK=2
TRIPLET_LOSS=0.3

# Device (use cuda if available, otherwise cpu)
DEVICE="cuda"

# Optional: Specify specific sample IDs to visualize (leave empty to use first N samples)
# SAMPLE_IDS=("001" "002" "003")
SAMPLE_IDS=""

# ============================================================================
# Main execution
# ============================================================================

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Build command - use generate_all_heatmaps.py to process ALL images
CMD="python3 generate_all_heatmaps.py \
    --ckpt_path ${CKPT_PATH} \
    --dataset_path ${DATASET_PATH} \
    --img_size ${IMG_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    --nclasses ${NCLASSES} \
    --block ${BLOCK} \
    --triplet_loss ${TRIPLET_LOSS} \
    --device ${DEVICE}"

# Note: generate_all_heatmaps.py processes ALL images automatically
# If you want to process specific samples, modify the script or use generate_dsa_heatmap.py instead

# Check if checkpoint exists
if [ ! -f "${CKPT_PATH}" ]; then
    echo "Error: Checkpoint file not found: ${CKPT_PATH}"
    echo ""
    echo "Available checkpoint files:"
    find . -name "*.pth" -type f 2>/dev/null | head -10
    echo ""
    echo "Please update CKPT_PATH in this script to point to a valid checkpoint file."
    exit 1
fi

# Run the heatmap generation script
echo "Starting heatmap generation for ALL test images..."
echo "Checkpoint: ${CKPT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Note: This will process ALL images in the test dataset"
echo ""

${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Heatmap generation completed successfully!"
    echo "  Results saved to: ${OUTPUT_DIR}"
else
    echo ""
    echo "✗ Heatmap generation failed. Please check the error messages above."
    exit 1
fi
