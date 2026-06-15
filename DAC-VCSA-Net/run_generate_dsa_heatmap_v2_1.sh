#!/bin/bash

# Generate DSA Ablation Heatmaps - Version 2.1 (CORRECTED)
# Supports category filtering for test set

set -e  # Exit on error

echo "=================================="
echo "DSA Heatmap Generator v2.1"
echo "CORRECTED - Better DSA Contrast"
echo "=================================="
echo ""

# Default Configuration
CHECKPOINT="checkpoint/pretrained_models/U1652/weights_end.pth"
DATASET="/root/dataset/University-Release"
OUTPUT_DIR="./heatmap_img_v2_1"
NUM_SAMPLES=4
IMG_SIZE=384
NCLASSES=701
BLOCK=2
TRIPLET_LOSS=0.3
DEVICE="cuda"
SEED=42

# Parse command line arguments
CATEGORIES=()
SAMPLE_IDS=()
CUSTOM_OUTPUT_DIR=""
CUSTOM_CHECKPOINT=""
CUSTOM_DATASET=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH          Path to model checkpoint (default: $CHECKPOINT)"
    echo "  --dataset PATH             Path to dataset (default: $DATASET)"
    echo "  --output_dir PATH          Output directory (default: $OUTPUT_DIR)"
    echo "  --categories ID1 ID2 ...  Category IDs to filter (e.g., --categories 1 2 3)"
    echo "  --sample_ids ID1 ID2 ...   Specific sample IDs to generate (overrides --categories)"
    echo "  --num_samples N            Number of samples if not using --sample_ids (default: $NUM_SAMPLES)"
    echo "  --img_size N               Image size (default: $IMG_SIZE)"
    echo "  --nclasses N               Number of classes (default: $NCLASSES)"
    echo "  --block N                  Block number (default: $BLOCK)"
    echo "  --triplet_loss F           Triplet loss weight (default: $TRIPLET_LOSS)"
    echo "  --device DEVICE            Device to use: cuda or cpu (default: $DEVICE)"
    echo "  --seed N                   Random seed (default: $SEED)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Generate heatmaps for specific categories"
    echo "  $0 --categories 1 2 3 4"
    echo ""
    echo "  # Generate heatmaps for specific sample IDs"
    echo "  $0 --sample_ids 0107 0108 0109"
    echo ""
    echo "  # Use custom checkpoint and dataset"
    echo "  $0 --checkpoint ./checkpoint/model.pth --dataset /path/to/dataset --categories 5 6"
    echo ""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CUSTOM_CHECKPOINT="$2"
            shift 2
            ;;
        --dataset)
            CUSTOM_DATASET="$2"
            shift 2
            ;;
        --output_dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        --categories)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                CATEGORIES+=("$1")
                shift
            done
            ;;
        --sample_ids)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                SAMPLE_IDS+=("$1")
                shift
            done
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --img_size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --nclasses)
            NCLASSES="$2"
            shift 2
            ;;
        --block)
            BLOCK="$2"
            shift 2
            ;;
        --triplet_loss)
            TRIPLET_LOSS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Apply custom values
if [ -n "$CUSTOM_CHECKPOINT" ]; then
    CHECKPOINT="$CUSTOM_CHECKPOINT"
fi
if [ -n "$CUSTOM_DATASET" ]; then
    DATASET="$CUSTOM_DATASET"
fi
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT_DIR"
if [ ${#CATEGORIES[@]} -gt 0 ]; then
    echo "  Categories: ${CATEGORIES[*]}"
fi
if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    echo "  Sample IDs: ${SAMPLE_IDS[*]}"
else
    echo "  Num samples: $NUM_SAMPLES"
fi
echo "  Image size: $IMG_SIZE"
echo "  Device: $DEVICE"
echo ""

# Build command
CMD="python generate_dsa_heatmap_v2_1.py"
CMD="$CMD --ckpt_path \"$CHECKPOINT\""
CMD="$CMD --dataset_path \"$DATASET\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --img_size $IMG_SIZE"
CMD="$CMD --nclasses $NCLASSES"
CMD="$CMD --block $BLOCK"
CMD="$CMD --triplet_loss $TRIPLET_LOSS"
CMD="$CMD --device $DEVICE"
CMD="$CMD --seed $SEED"

# Add categories if specified
if [ ${#CATEGORIES[@]} -gt 0 ]; then
    CMD="$CMD --categories ${CATEGORIES[*]}"
fi

# Add sample_ids if specified, otherwise use num_samples
if [ ${#SAMPLE_IDS[@]} -gt 0 ]; then
    CMD="$CMD --sample_ids ${SAMPLE_IDS[*]}"
else
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Execute command
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Generation completed!"
    echo "  Results: $OUTPUT_DIR"
    echo "=================================="
else
    echo ""
    echo "✗ Generation failed!"
    exit 1
fi
