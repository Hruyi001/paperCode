#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

show_help() {
    cat <<'EOF'
Usage: bash scripts/infer.sh

Configurable environment variables:
  PYTHON_BIN          Python executable. Default: python
  CONFIG_PATH         Model config path. Default: ./model_configs/dino_b_QDFL.yaml
  CKPT_PATH           Checkpoint path. Default: ./pretrained_weights/DINO_QDFL_U1652.pth
  DINOV2_VITB14_WEIGHT DINOv2 backbone weight path. Default: CKPT_PATH
  DATASET             Dataset name: U1652 | SUES200 | DENSEUAV. Default: U1652
  MODES               Comma-separated modes. Default: sat->drone,drone->sat
  HEIGHTS             Comma-separated SUES-200 heights. Default: 200
  IMG_SIZE            Test image size as H,W. Default: 280,280
  BATCH_SIZE          Inference batch size. Default: 192
  FLIPLR              Use horizontal flip augmentation: true | false. Default: true
  EVALUATE_TYPE       Distance type passed to evaluation. Default: Euclidean
  SAVE_IMG_PATH       Save matched image paths: true | false. Default: false
  U1652_TEST_ROOT     University-1652 test root.
  SUES200_TEST_ROOT   SUES-200 Testing root.
  DENSEUAV_TEST_ROOT  DenseUAV test root.

Examples:
  bash scripts/infer.sh
  DATASET=SUES200 HEIGHTS=150,200,250,300 CKPT_PATH=./LOGS/.../last.ckpt bash scripts/infer.sh
  DATASET=DENSEUAV MODES=drone->sat IMG_SIZE=224,224 BATCH_SIZE=32 bash scripts/infer.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

export CONFIG_PATH="${CONFIG_PATH:-./model_configs/dino_b_QDFL.yaml}"
export CKPT_PATH="${CKPT_PATH:-./pretrained_weights/DINO_QDFL_U1652.pth}"
export DINOV2_VITB14_WEIGHT="${DINOV2_VITB14_WEIGHT:-$CKPT_PATH}"
export DATASET="${DATASET:-U1652}"
export MODES="${MODES:-sat->drone,drone->sat}"
export HEIGHTS="${HEIGHTS:-200}"
export IMG_SIZE="${IMG_SIZE:-280,280}"
export BATCH_SIZE="${BATCH_SIZE:-192}"
export FLIPLR="${FLIPLR:-true}"
export EVALUATE_TYPE="${EVALUATE_TYPE:-Euclidean}"
export SAVE_IMG_PATH="${SAVE_IMG_PATH:-false}"

case "${DATASET^^}" in
    U1652)
        export U1652_TEST_ROOT="${U1652_TEST_ROOT:-/root/dataset/University-Release/test/}"
        DATA_ROOT="$U1652_TEST_ROOT"
        unset SUES200_TEST_ROOT DENSEUAV_TEST_ROOT
        ;;
    SUES200|SUES|SUES-200)
        export SUES200_TEST_ROOT="${SUES200_TEST_ROOT:-/media/whu/Largedisk/datasets/SUES-200-512x512/Testing}"
        DATA_ROOT="$SUES200_TEST_ROOT"
        unset U1652_TEST_ROOT DENSEUAV_TEST_ROOT
        ;;
    DENSEUAV)
        export DENSEUAV_TEST_ROOT="${DENSEUAV_TEST_ROOT:-/media/whu/Largedisk/datasets/DenseUAV/test}"
        DATA_ROOT="$DENSEUAV_TEST_ROOT"
        unset U1652_TEST_ROOT SUES200_TEST_ROOT
        ;;
    *)
        echo "Unsupported DATASET: $DATASET" >&2
        exit 1
        ;;
esac

missing=0
for item in \
    "CONFIG_PATH:file:$CONFIG_PATH" \
    "CKPT_PATH:file:$CKPT_PATH" \
    "DINOV2_VITB14_WEIGHT:file:$DINOV2_VITB14_WEIGHT" \
    "${DATASET}_TEST_ROOT:dir:$DATA_ROOT"; do
    IFS=: read -r name kind path <<< "$item"
    if [[ "$kind" == "file" && ! -f "$path" ]]; then
        echo "$name not found: $path" >&2
        missing=1
    elif [[ "$kind" == "dir" && ! -d "$path" ]]; then
        echo "$name not found: $path" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from Supervised_evaluate import evaluate_supervised_learning
from utils.commons import load_config


def parse_bool(name):
    value = os.environ[name].strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be true or false, got {os.environ[name]!r}")


def parse_int_pair(name):
    raw = os.environ[name]
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"{name} must be formatted as H,W, got {raw!r}")
    return tuple(values)


config_path = os.environ["CONFIG_PATH"]
ckpt_path = os.environ["CKPT_PATH"]
dataset = os.environ["DATASET"]
modes = [item.strip() for item in os.environ["MODES"].split(",") if item.strip()]
heights = [int(item.strip()) for item in os.environ["HEIGHTS"].split(",") if item.strip()]
img_size = parse_int_pair("IMG_SIZE")
batch_size = int(os.environ["BATCH_SIZE"])
fliplr = parse_bool("FLIPLR")
save_img_path = parse_bool("SAVE_IMG_PATH")
evaluate_type = os.environ["EVALUATE_TYPE"]

if not Path(config_path).is_file():
    raise FileNotFoundError(f"CONFIG_PATH not found: {config_path}")
if not Path(ckpt_path).is_file():
    raise FileNotFoundError(f"CKPT_PATH not found: {ckpt_path}")

configs = load_config(config_path)["model_configs"]

print("=" * 80)
print("QDFL inference/evaluation")
print(f"Config: {config_path}")
print(f"Checkpoint: {ckpt_path}")
print(f"DINOv2 weight: {os.environ['DINOV2_VITB14_WEIGHT']}")
print(f"Dataset: {dataset}")
print(f"Modes: {modes}")
print(f"Heights: {heights}")
print(f"Image size: {img_size}")
print(f"Batch size: {batch_size}")
print(f"Flip LR: {fliplr}")
print("=" * 80)

for height in heights:
    for mode in modes:
        print(f"\nEvaluating dataset={dataset}, height={height}, mode={mode}")
        evaluate_supervised_learning(
            which_dataset=dataset,
            height=height,
            mode=mode,
            fliplr=fliplr,
            img_size=img_size,
            batch_size=batch_size,
            configs=configs,
            pth_path=ckpt_path,
            evaluate_type=evaluate_type,
            save_img_path=save_img_path,
        )
PY
