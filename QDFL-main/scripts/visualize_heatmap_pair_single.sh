#!/usr/bin/env bash
set -euo pipefail

# 手动指定一对（无人机/卫星）图片，生成 2x3 对比结果图：
# - 第1列：输入图（Drone/Satellite）
# - 第2列：Overlay(Backbone featmap)
# - 第3列：Overlay(QDFL x_fine_0)
#
# Usage: bash scripts/visualize_heatmap_pair_single.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

## 1. /root/dataset/University-Release/test/query_drone/0007/image-07.jpeg
## 2. 
# ========== 你只需要改这两个路径 ==========
# DRONE_IMAGE="/root/dataset/University-Release/test/query_drone/0011/image-12.jpeg"
# SATELLITE_IMAGE="/root/dataset/University-Release/test/gallery_satellite/0011/0011.jpg"

DRONE_IMAGE="/root/dataset/University-Release/test/gallery_drone/0026/image-42.jpeg"
SATELLITE_IMAGE="/root/dataset/University-Release/test/gallery_satellite/0026/0026.jpg"
# 模型与配置
CHECKPOINT="LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt"
CONFIG_FILE="./model_configs/dino_b_QDFL.yaml"

# 输出文件（png）
OUTPUT_DIR="./heatmap_visualizations_pairs_single"
OUTPUT_NAME="" # 留空则自动命名（根据输入文件名）

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

"$PYTHON_BIN" visualize_heatmap_pair_single.py \
  --drone_image "$DRONE_IMAGE" \
  --satellite_image "$SATELLITE_IMAGE" \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  --img_size "$IMG_HEIGHT" "$IMG_WIDTH" \
  --alpha "$ALPHA" \
  --heatmap_method "$HEATMAP_METHOD" \
  --cmap "$CMAP"

echo "完成，输出目录: $OUTPUT_DIR"

