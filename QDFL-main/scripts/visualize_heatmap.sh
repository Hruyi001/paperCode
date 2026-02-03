#!/usr/bin/env bash
set -euo pipefail

# 热力图可视化脚本 - 直接在下面修改数据集路径
# Usage: bash scripts/visualize_heatmap.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 选择可用的 Python（优先使用当前环境中的 python，其次 python3）
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

# ========== 在这里直接设置你的数据集路径 ==========
# University-1652 数据集路径
U1652_TEST_ROOT="/root/dataset/University-Release/test/"
U1652_QUERY_DRONE="${U1652_TEST_ROOT}query_drone"  # query_drone目录
U1652_QUERY_SATELLITE="${U1652_TEST_ROOT}query_satellite"  # query_satellite目录

# SUES-200 数据集路径
SUES200_TEST_ROOT="/media/whu/Largedisk/datasets/SUES-200-512x512/Testing"
SUES200_QUERY_DRONE="${SUES200_TEST_ROOT}/query_drone"
SUES200_QUERY_SATELLITE="${SUES200_TEST_ROOT}/query_satellite"

# DenseUAV 数据集路径
DENSEUAV_TEST_ROOT="/media/whu/Largedisk/datasets/DenseUAV/test"
DENSEUAV_QUERY_DRONE="${DENSEUAV_TEST_ROOT}/query_drone"

# 选择要可视化的数据集和视图: U1652_DRONE | U1652_SATELLITE | SUES200_DRONE | SUES200_SATELLITE | DENSEUAV_DRONE
# 或者直接指定完整路径
VISUALIZE_DATASET="U1652_DRONE"  # 默认使用U1652的query_drone

# 模型checkpoint路径
CHECKPOINT="LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt"

# DINOv2 预训练权重路径
# 选项1: 使用 DINOv2 官方预训练权重（需要下载）
#   下载地址: https://github.com/facebookresearch/dinov2
#   将权重文件放在 ./pretrained_weights/ 目录下
# 选项2: 使用已有的 checkpoint 文件（从 checkpoint 中提取 backbone 权重）
#   如果设置了 DINOV2_CHECKPOINT，将使用该 checkpoint 中的 backbone 权重
DINOV2_WEIGHTS_DIR="./pretrained_weights"  # 预训练权重目录
DINOV2_CHECKPOINT="./checkpoint/DINO_QDFL_U1652.pth"  # 使用 checkpoint 文件中的 backbone 权重

# 模型配置文件
CONFIG_FILE="./model_configs/dino_b_QDFL.yaml"

# 输出目录
OUTPUT_DIR="./heatmap_visualizations"

# 要可视化的图像数量
NUM_IMAGES=20

# 输入图像大小 [height width]
IMG_HEIGHT=280
IMG_WIDTH=280
# ==================================================

# 根据选择的数据集设置对应的图像目录
case "$VISUALIZE_DATASET" in
    U1652_DRONE)
        IMAGE_DIR="$U1652_QUERY_DRONE"
        echo "使用数据集: University-1652 (query_drone)"
        ;;
    U1652_SATELLITE)
        IMAGE_DIR="$U1652_QUERY_SATELLITE"
        echo "使用数据集: University-1652 (query_satellite)"
        ;;
    SUES200_DRONE)
        IMAGE_DIR="$SUES200_QUERY_DRONE"
        echo "使用数据集: SUES-200 (query_drone)"
        ;;
    SUES200_SATELLITE)
        IMAGE_DIR="$SUES200_QUERY_SATELLITE"
        echo "使用数据集: SUES-200 (query_satellite)"
        ;;
    DENSEUAV_DRONE)
        IMAGE_DIR="$DENSEUAV_QUERY_DRONE"
        echo "使用数据集: DenseUAV (query_drone)"
        ;;
    *)
        # 如果直接指定了路径，使用该路径
        if [ -d "$VISUALIZE_DATASET" ]; then
            IMAGE_DIR="$VISUALIZE_DATASET"
            echo "使用自定义路径: $IMAGE_DIR"
        else
            echo "错误: 未知的数据集选项 $VISUALIZE_DATASET"
            echo "请选择: U1652_DRONE | U1652_SATELLITE | SUES200_DRONE | SUES200_SATELLITE | DENSEUAV_DRONE"
            echo "或者直接指定一个有效的目录路径"
            exit 1
        fi
        ;;
esac

IMG_SIZE="$IMG_HEIGHT $IMG_WIDTH"

# 检查参数
if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误: 图像目录不存在: $IMAGE_DIR"
    echo "请检查脚本顶部的路径配置，或修改 VISUALIZE_DATASET 变量"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT"
    echo "请检查脚本顶部的 CHECKPOINT 变量，确保路径正确"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置 DINOv2 backbone 预训练权重路径（和 train.sh 保持一致的优先级）
export DINOV2_WEIGHTS_DIR
if [ -n "$DINOV2_CHECKPOINT" ] && [ -f "$DINOV2_CHECKPOINT" ]; then
    export DINOV2_VITB14_WEIGHT="$DINOV2_CHECKPOINT"
    echo "使用 checkpoint 中的 backbone 权重: $DINOV2_CHECKPOINT"
elif [ -f "$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth" ]; then
    export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
    echo "使用 pretrained_weights 中的权重文件: $DINOV2_VITB14_WEIGHT"
elif [ -f "$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" ]; then
    echo "使用官方 DINOv2 预训练权重: $DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth"
else
    echo "错误: 未找到 DINOv2 预训练权重文件"
    echo "请执行以下操作之一:"
    echo "  1. 下载官方 DINOv2 权重:"
    echo "     wget -O $DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    echo "  2. 或者将 checkpoint 文件放在 $DINOV2_CHECKPOINT"
    echo "  3. 或者将权重文件放在 $DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
    exit 1
fi

echo "图像目录: $IMAGE_DIR"
echo ""

echo "=========================================="
echo "热力图可视化配置"
echo "=========================================="
echo "图像目录: $IMAGE_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "图像数量: $NUM_IMAGES"
echo "图像大小: ${IMG_HEIGHT}x${IMG_WIDTH}"
echo "=========================================="
echo ""

# 检查并修复 NumPy 版本兼容性问题
echo "检查依赖环境..."
"$PYTHON_BIN" -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || {
    echo "警告: NumPy 版本可能不兼容，正在修复..."
    "$PYTHON_BIN" -m pip install "numpy<2.0.0" --quiet || {
        echo "错误: 当前 Python 无法使用 pip（可能没激活 conda 环境）"
        echo "请先激活你的环境，例如：conda activate qfdl"
        exit 1
    }
}

# 运行可视化脚本
echo "开始生成热力图可视化..."
"$PYTHON_BIN" visualize_heatmap.py \
    --image_dir "$IMAGE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_images "$NUM_IMAGES" \
    --compare \
    --img_size $IMG_SIZE

echo ""
echo "=========================================="
echo "可视化完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="
