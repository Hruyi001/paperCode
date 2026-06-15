#!/usr/bin/env bash
set -euo pipefail

# University-1652 推理/测试脚本
# Usage: bash scripts/infer_u1652.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# ========== 数据集路径（University-1652 test）==========
U1652_TEST_ROOT="${U1652_TEST_ROOT:-/root/dataset/University-Release/test}"
QUERY_DRONE_DIR="${U1652_TEST_ROOT}/query_drone"
QUERY_SAT_DIR="${U1652_TEST_ROOT}/query_satellite"
GALLERY_DRONE_DIR="${U1652_TEST_ROOT}/gallery_drone"
GALLERY_SAT_DIR="${U1652_TEST_ROOT}/gallery_satellite"

# 模型与配置
CONFIG_FILE="${CONFIG_FILE:-./model_configs/dino_b_QDFL.yaml}"
CHECKPOINT="${CHECKPOINT:-LOGS/dinov2_vitb14_QDFL/lightning_logs/version_3/checkpoints/last.ckpt}"

# 推理参数
MODES="${MODES:-sat->drone,drone->sat}"
IMG_HEIGHT="${IMG_HEIGHT:-280}"
IMG_WIDTH="${IMG_WIDTH:-280}"
BATCH_SIZE="${BATCH_SIZE:-192}"
FLIPLR="${FLIPLR:-true}"
EVALUATE_TYPE="${EVALUATE_TYPE:-Euclidean}"
SAVE_IMG_PATH="${SAVE_IMG_PATH:-false}"

# DINOv2 权重：优先直接使用当前完整 checkpoint，因为其中包含 backbone.model.*
DINOV2_WEIGHTS_DIR="${DINOV2_WEIGHTS_DIR:-./pretrained_weights}"
DINOV2_CHECKPOINT="${DINOV2_CHECKPOINT:-$CHECKPOINT}"

export U1652_TEST_ROOT
export DINOV2_WEIGHTS_DIR
if [ -n "$DINOV2_CHECKPOINT" ] && [ -f "$DINOV2_CHECKPOINT" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_CHECKPOINT"
elif [ -f "$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth"
elif [ -f "$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" ]; then
  export DINOV2_VITB14_WEIGHT="$DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth"
else
  echo "错误: 未找到 DINOv2 权重文件" >&2
  echo "检查路径: $DINOV2_CHECKPOINT" >&2
  echo "检查路径: $DINOV2_WEIGHTS_DIR/DINO_QDFL_U1652.pth" >&2
  echo "检查路径: $DINOV2_WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" >&2
  exit 1
fi

# 基本检查：一次性检查清楚，避免进入 Python 后逐个报错
missing=0
for p in "$U1652_TEST_ROOT" "$QUERY_DRONE_DIR" "$QUERY_SAT_DIR" "$GALLERY_DRONE_DIR" "$GALLERY_SAT_DIR"; do
  if [ ! -d "$p" ]; then
    echo "错误: 目录不存在: $p" >&2
    missing=1
  fi
done
for p in "$CONFIG_FILE" "$CHECKPOINT" "$DINOV2_VITB14_WEIGHT"; do
  if [ ! -f "$p" ]; then
    echo "错误: 文件不存在: $p" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "检查通过"
  echo "Python: $PYTHON_BIN"
  echo "Dataset root: $U1652_TEST_ROOT"
  echo "Config: $CONFIG_FILE"
  echo "Checkpoint: $CHECKPOINT"
  echo "DINOv2 weight: $DINOV2_VITB14_WEIGHT"
  echo "Modes: $MODES"
  echo "Image size: ${IMG_HEIGHT}x${IMG_WIDTH}"
  echo "Batch size: $BATCH_SIZE"
  exit 0
fi

echo "开始 University-1652 推理测试..."
echo "Dataset root: $U1652_TEST_ROOT"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "DINOv2 weight: $DINOV2_VITB14_WEIGHT"
echo "Modes: $MODES"
echo "Image size: ${IMG_HEIGHT}x${IMG_WIDTH}"
echo "Batch size: $BATCH_SIZE"

export CONFIG_FILE CHECKPOINT MODES IMG_HEIGHT IMG_WIDTH BATCH_SIZE FLIPLR EVALUATE_TYPE SAVE_IMG_PATH

"$PYTHON_BIN" - <<'PY'
import os

from Supervised_evaluate import evaluate_supervised_learning
from utils.commons import load_config


def parse_bool(value):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Boolean value must be true or false, got {value!r}")


config_file = os.environ["CONFIG_FILE"]
checkpoint = os.environ["CHECKPOINT"]
modes = [mode.strip() for mode in os.environ["MODES"].split(",") if mode.strip()]
img_size = (int(os.environ["IMG_HEIGHT"]), int(os.environ["IMG_WIDTH"]))
batch_size = int(os.environ["BATCH_SIZE"])
fliplr = parse_bool(os.environ["FLIPLR"])
save_img_path = parse_bool(os.environ["SAVE_IMG_PATH"])
evaluate_type = os.environ["EVALUATE_TYPE"]
configs = load_config(config_file)["model_configs"]

print("=" * 80)
print("University-1652 QDFL inference/evaluation")
print(f"Config: {config_file}")
print(f"Checkpoint: {checkpoint}")
print(f"DINOv2 weight: {os.environ['DINOV2_VITB14_WEIGHT']}")
print(f"Modes: {modes}")
print(f"Image size: {img_size}")
print(f"Batch size: {batch_size}")
print(f"Flip LR: {fliplr}")
print("=" * 80)

for mode in modes:
    print(f"\nEvaluating University-1652 mode={mode}")
    evaluate_supervised_learning(
        which_dataset="U1652",
        height=200,
        mode=mode,
        fliplr=fliplr,
        img_size=img_size,
        batch_size=batch_size,
        configs=configs,
        pth_path=checkpoint,
        evaluate_type=evaluate_type,
        save_img_path=save_img_path,
    )
PY

echo "University-1652 推理测试完成"
