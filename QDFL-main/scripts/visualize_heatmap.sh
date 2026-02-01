#!/usr/bin/env bash
set -euo pipefail

# 热力图可视化脚本 - 直接在下面修改数据集路径
# Usage: bash scripts/visualize_heatmap.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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
CHECKPOINT="./checkpoint/DINO_QDFL_U1652.pth"

# 模型配置文件
CONFIG_FILE="./model_configs/dino_b_QDFL.yaml"

# 输出目录
OUTPUT_DIR="./heatmap_visualizations"

# 要可视化的图像数量
NUM_IMAGES=10

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
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || {
    echo "警告: NumPy 版本可能不兼容，正在修复..."
    pip install "numpy<2.0.0" --quiet
}

# 运行可视化脚本
echo "开始生成热力图可视化..."
python visualize_heatmap.py \
    --image_dir "$IMAGE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_images "$NUM_IMAGES" \
    --img_size $IMG_SIZE

echo ""
echo "=========================================="
echo "可视化完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="
