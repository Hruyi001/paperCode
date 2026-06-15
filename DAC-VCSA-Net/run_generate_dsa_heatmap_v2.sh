#!/bin/bash

# Generate DSA Ablation Heatmaps - Version 2.0
# This version implements cross-view correlation visualization as described in the paper

echo "=================================="
echo "DSA Heatmap Generator v2.0"
echo "Cross-View Correlation Analysis"
echo "=================================="
echo ""

# Configuration
CHECKPOINT="checkpoint/pretrained_models/U1652/weights_end.pth"
DATASET="/root/dataset/University-Release"
OUTPUT_DIR="./heatmap_img_v2"
NUM_SAMPLES=4
IMG_SIZE=384
NCLASSES=701
BLOCK=2
TRIPLET_LOSS=0.3

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    echo "Please update the DATASET variable in this script."
    exit 1
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of Samples: $NUM_SAMPLES"
echo ""
echo "Starting heatmap generation..."
echo ""

# Run the heatmap generation script
python generate_dsa_heatmap_v2.py \
    --ckpt_path "$CHECKPOINT" \
    --dataset_path "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --img_size $IMG_SIZE \
    --nclasses $NCLASSES \
    --block $BLOCK \
    --triplet_loss $TRIPLET_LOSS

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Heatmap generation completed!"
    echo "  Results saved to: $OUTPUT_DIR"
    echo "=================================="
else
    echo ""
    echo "=================================="
    echo "✗ Heatmap generation failed!"
    echo "  Please check the error messages above."
    echo "=================================="
    exit 1
fi
