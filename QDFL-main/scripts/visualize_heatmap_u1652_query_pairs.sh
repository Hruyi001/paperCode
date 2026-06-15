#!/usr/bin/env bash
set -euo pipefail

# 生成“无人机-卫星配对”三列结果图：
# - 第1列：Input (Drone/Satellite)
# - 第2列：Overlay(Backbone featmap)
# - 第3列：Overlay(QDFL x_fine_0)
#
# Usage: bash scripts/visualize_heatmap_u1652_query_pairs.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# ========== 数据集路径（University-1652 test）==========
U1652_TEST_ROOT="/root/dataset/University-Release/test"
DRONE_DIR="${U1652_TEST_ROOT}/query_drone"
SAT_DIR="${U1652_TEST_ROOT}/query_satellite"

# 模型与配置
CHECKPOINT="LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt"
CONFIG_FILE="./model_configs/dino_b_QDFL.yaml"

# 输出目录
OUTPUT_DIR="./heatmap_visualizations_u1652_query_pairs"

# 生成多少对（0=全部）
NUM_PAIRS=01

# 输入大小
IMG_HEIGHT=280
IMG_WIDTH=280

# overlay alpha / colormap
ALPHA=0.5
CMAP="jet"
HEATMAP_METHOD="mean"

# DINOv2 权重（与其它脚本一致）
DINOV2_WEIGHTS_DIR="./pretrained_weights"
DINOV2_CHECKPOINT="./checkpoint/DINO_QDFL_U1652.pth"

export DINOV2_WEIGHTS_DIR
if [ -n "$DINOV2_CHECKPOINT" ] && [ -f "$DINOV2_CHECKPOINT" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_CHECKPOINT"
elif [ -f "$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
elif [ -f "$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth"
else
  echo "错误: 未找到 DINOv2 权重文件"
  exit 1
fi

# 基本检查
for p in "$DRONE_DIR" "$SAT_DIR"; do
  if [ ! -d "$p" ]; then
    echo "错误: 目录不存在: $p"
    exit 1
  fi
done
if [ ! -f "$CHECKPOINT" ]; then
  echo "错误: Checkpoint文件不存在: $CHECKPOINT"
  exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

echo "开始生成配对三列热力图（可能耗时较长）..."
"$PYTHON_BIN" visualize_heatmap_pairs.py \
  --drone_dir "$DRONE_DIR" \
  --satellite_dir "$SAT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --num_pairs "$NUM_PAIRS" \
  --pair_mode by_id \
  --satellite_pick first \
  --img_size "$IMG_HEIGHT" "$IMG_WIDTH" \
  --alpha "$ALPHA" \
  --heatmap_method "$HEATMAP_METHOD" \
  --cmap "$CMAP"

echo "完成，输出目录: $OUTPUT_DIR"

