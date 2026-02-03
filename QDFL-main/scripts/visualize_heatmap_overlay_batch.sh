#!/usr/bin/env bash
set -euo pipefail

# 批量生成 overlay 热力图（只保存 QDFL x_fine_0 overlay）
# 本脚本已固定输入/输出目录，直接运行即可：
#   bash scripts/visualize_heatmap_overlay_batch.sh
#
# 如需修改目录，在下面的 INPUT_DIR / OUTPUT_DIR 变量处改即可。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 固定输入图像目录和输出目录
INPUT_DIR="/root/exp/exp4/4-1/img"
OUTPUT_DIR="/root/exp/exp4/4-1/qdfl"

# 选择可用的 Python（优先使用当前环境中的 python，其次 python3）
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# 默认模型路径（可通过环境变量覆盖）
CHECKPOINT="${CHECKPOINT:-LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt}"
CONFIG_FILE="${CONFIG_FILE:-./model_configs/dino_b_QDFL.yaml}"

# DINOv2 权重选择（和 train.sh/visualize_heatmap.sh 一致）
DINOV2_WEIGHTS_DIR="${DINOV2_WEIGHTS_DIR:-./pretrained_weights}"
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-./checkpoint/DINO_QDFL_U1652.pth}"
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

if [ ! -d "$INPUT_DIR" ]; then
  echo "错误: 输入图像目录不存在: $INPUT_DIR"
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

mkdir -p "$OUTPUT_DIR"

echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "配置文件: $CONFIG_FILE"

echo "检查依赖环境..."
"$PYTHON_BIN" -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>/dev/null || {
  echo "警告: NumPy 版本可能不兼容，正在修复..."
  "$PYTHON_BIN" -m pip install "numpy<2.0.0" --quiet || {
    echo "错误: 当前 Python 无法使用 pip（可能没激活 conda 环境）"
    echo "请先激活你的环境，例如：conda activate qfdl"
    exit 1
  }
}

echo "开始批量生成 overlay 热力图..."
"$PYTHON_BIN" visualize_heatmap_overlay_batch.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG_FILE" \
  --img_size 280 280 \
  --alpha 0.5 \
  --heatmap_method mean \
  --preserve_structure

echo "完成！结果保存在: $OUTPUT_DIR"

