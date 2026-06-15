#!/usr/bin/env bash
set -euo pipefail

# 全量热力图可视化脚本（University-1652 test 目录下所有图片）
# Usage: bash scripts/visualize_heatmap_u1652_test_all.sh
#
# 说明：
# - 该脚本会递归扫描 /root/dataset/University-Release/test/ 下的所有图像并逐张生成热力图。
# - visualize_heatmap.py 内部逻辑：当图片数 > --num_images 时会 random.sample 抽样。
#   因此这里把 --num_images 设为一个很大的值，确保不会触发随机抽样（从而处理“全部图片”）。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 选择可用的 Python（优先使用当前环境中的 python，其次 python3）
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

# ========== 在这里直接设置你的数据集路径 ==========
U1652_TEST_ROOT="/root/dataset/University-Release/test/"
IMAGE_DIR="$U1652_TEST_ROOT"

# 模型 checkpoint 路径
CHECKPOINT="LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt"

# 模型配置文件
CONFIG_FILE="./model_configs/dino_b_QDFL.yaml"

# 输出目录（建议单独放一个目录，避免和其它可视化混在一起）
OUTPUT_DIR="./heatmap_visualizations_u1652_test_all"

# 一个足够大的数：确保不会触发随机抽样，从而处理全部图片
NUM_IMAGES=1000000000

# 输入图像大小 [height width]
IMG_HEIGHT=280
IMG_WIDTH=280

# DINOv2 权重路径（与 train.sh / visualize_heatmap.sh 保持一致的优先级）
DINOV2_WEIGHTS_DIR="./pretrained_weights"
DINOV2_CHECKPOINT="./checkpoint/DINO_QDFL_U1652.pth"
# ==================================================

IMG_SIZE="$IMG_HEIGHT $IMG_WIDTH"

# 检查参数
if [ ! -d "$IMAGE_DIR" ]; then
    echo "错误: 图像目录不存在: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置 DINOv2 backbone 预训练权重路径
export DINOV2_WEIGHTS_DIR
if [ -n "$DINOV2_CHECKPOINT" ] && [ -f "$DINOV2_CHECKPOINT" ]; then
    export DINOV2_VITB14_WEIGHT="$DINOV2_CHECKPOINT"
    echo "使用 checkpoint 中的 backbone 权重: $DINOV2_CHECKPOINT"
elif [ -f "$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth" ]; then
    export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
    echo "使用 pretrained_weights 中的权重文件: $DINOV2_VITB14_WEIGHT"
elif [ -f "$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" ]; then
    export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth"
    echo "使用官方 DINOv2 预训练权重: $DINOV2_VITB14_WEIGHT"
else
    echo "错误: 未找到 DINOv2 预训练权重文件"
    echo "请执行以下操作之一:"
    echo "  1. 下载官方 DINOv2 权重:"
    echo "     wget -O $DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    echo "  2. 或者将 checkpoint 文件放在 $DINOV2_CHECKPOINT"
    echo "  3. 或者将权重文件放在 $DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
    exit 1
fi

echo ""
echo "=========================================="
echo "热力图可视化（全量）配置"
echo "=========================================="
echo "图像目录: $IMAGE_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "num_images(防抽样上限): $NUM_IMAGES"
echo "图像大小: ${IMG_HEIGHT}x${IMG_WIDTH}"
echo "=========================================="
echo ""

# 检查并修复 NumPy 版本兼容性问题（尽量不影响环境；仅在 numpy import 失败时尝试修复）
echo "检查依赖环境..."
"$PYTHON_BIN" -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || {
    echo "警告: NumPy 可能无法导入，尝试安装 numpy<2.0.0 ..."
    "$PYTHON_BIN" -m pip install "numpy<2.0.0" --quiet || {
        echo "错误: 当前 Python 无法使用 pip（可能没激活 conda 环境）"
        echo "请先激活你的环境，例如：conda activate qfdl"
        exit 1
    }
}

echo "开始生成热力图可视化（可能耗时较长）..."
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

