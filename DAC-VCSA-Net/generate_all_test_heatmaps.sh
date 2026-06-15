#!/bin/bash
# Script to generate heatmaps for all test images
# Usage: ./generate_all_test_heatmaps.sh [checkpoint_path]

DATASET_PATH="/root/dataset/University-Release"
OUTPUT_DIR="./heatmap_img"
CKPT_PATH="${1:-checkpoint/pretrained_models/U1652/weights_end.pth}"

echo "========================================"
echo "Generate Heatmaps for All Test Images"
echo "========================================"
echo "Dataset: $DATASET_PATH"
echo "Checkpoint: $CKPT_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    echo ""
    echo "Please provide checkpoint path as argument:"
    echo "  ./generate_all_test_heatmaps.sh /path/to/checkpoint.pth"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the script
python3 generate_all_heatmaps.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --img_size 384 \
    --nclasses 701 \
    --block 2 \
    --triplet_loss 0.3 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All heatmaps generated successfully!"
    echo "  Results saved to: $OUTPUT_DIR"
else
    echo ""
    echo "✗ Heatmap generation failed."
    exit 1
fi
