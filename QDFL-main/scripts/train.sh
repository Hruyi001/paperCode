#!/usr/bin/env bash
set -euo pipefail

# 训练脚本 - 直接在下面修改数据集路径
# Usage: bash scripts/train.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ========== 在这里直接设置你的数据集路径 ==========
# University-1652 数据集路径
U1652_TRAIN_ROOT="/root/dataset/University-Release/train/"
U1652_TEST_ROOT="/root/dataset/University-Release/test/"  # 测试集路径

# SUES-200 数据集路径
SUES200_TRAIN_ROOT="/media/whu/Largedisk/datasets/SUES-200-512x512/Training"
SUES200_TEST_ROOT="/media/whu/Largedisk/datasets/SUES-200-512x512/Testing"  # 测试集路径
SUES200_HEIGHT="250"  # 可选: 150, 200, 250, 300

# DenseUAV 数据集路径
DENSEUAV_TRAIN_ROOT="/media/whu/Largedisk/datasets/DenseUAV/train"
DENSEUAV_TEST_ROOT="/media/whu/Largedisk/datasets/DenseUAV/test"  # 测试集路径

# 选择要训练的数据集: U1652 | SUES200 | DENSEUAV
QDFL_DATASET="U1652"

# DINOv2 预训练权重路径
# 选项1: 使用 DINOv2 官方预训练权重（需要下载）
#   下载地址: https://github.com/facebookresearch/dinov2
#   将权重文件放在 ./pretrained_weights/ 目录下
# 选项2: 使用已有的 checkpoint 文件（从 checkpoint 中提取 backbone 权重）
#   如果设置了 DINOV2_CHECKPOINT，将使用该 checkpoint 中的 backbone 权重
DINOV2_WEIGHTS_DIR="./pretrained_weights"  # 预训练权重目录
DINOV2_CHECKPOINT="checkpoint/DINO_QDFL_U1652.pth"  # 使用 checkpoint 文件中的 backbone 权重

# 其他配置
NUM_WORKERS="4"
# ==================================================

# 根据选择的数据集设置对应的路径
case "$QDFL_DATASET" in
    U1652)
        export U1652_TRAIN_ROOT
        export U1652_TEST_ROOT
        echo "使用数据集: University-1652"
        echo "训练路径: $U1652_TRAIN_ROOT"
        echo "测试路径: $U1652_TEST_ROOT"
        ;;
    SUES200)
        export SUES200_TRAIN_ROOT
        export SUES200_TEST_ROOT
        export SUES200_HEIGHT
        echo "使用数据集: SUES-200 (height=$SUES200_HEIGHT)"
        echo "训练路径: $SUES200_TRAIN_ROOT"
        echo "测试路径: $SUES200_TEST_ROOT"
        ;;
    DENSEUAV)
        export DENSEUAV_TRAIN_ROOT
        export DENSEUAV_TEST_ROOT
        echo "使用数据集: DenseUAV"
        echo "训练路径: $DENSEUAV_TRAIN_ROOT"
        echo "测试路径: $DENSEUAV_TEST_ROOT"
        ;;
    *)
        echo "错误: 未知的数据集 $QDFL_DATASET"
        exit 1
        ;;
esac

export QDFL_DATASET
export NUM_WORKERS
export DINOV2_WEIGHTS_DIR

# 如果指定了 checkpoint，使用 checkpoint 中的 backbone 权重
if [ -n "$DINOV2_CHECKPOINT" ] && [ -f "$DINOV2_CHECKPOINT" ]; then
    # 使用 checkpoint 文件作为预训练权重（代码会自动提取 backbone 权重）
    export DINOV2_VITB14_WEIGHT="$DINOV2_CHECKPOINT"
    echo "使用 checkpoint 中的 backbone 权重: $DINOV2_CHECKPOINT"
fi

# 检查并修复 NumPy 版本兼容性问题
echo "检查依赖环境..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || {
    echo "警告: NumPy 版本可能不兼容，正在修复..."
    pip install "numpy<2.0.0" --quiet
}

echo "开始训练..."
python main.py
