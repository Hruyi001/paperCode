#!/bin/bash

# ============================================
# Safe-Net 对齐参数保存脚本
# 批量处理query_drone下的所有类别和所有图片，只保存参数到txt文件
# ============================================

# 保证无论从哪里运行，都在脚本所在目录执行（避免相对路径写到别处）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ============================================
# 配置区域 - 请根据实际情况修改
# ============================================

# 数据集路径配置
DATASET_ROOT="/root/dataset/University-Release/test"  # 数据集根目录
# 或者使用相对路径: DATASET_ROOT="./data/University-Release/test"

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 图像尺寸
IMG_HEIGHT=256
IMG_WIDTH=256

# 输出目录
OUTPUT_DIR="./alignment_comparisons1"

# 参数文件路径（可选，如果不指定则使用 OUTPUT_DIR/alignment_params.txt）
PARAMS_FILE=""  # 例如: PARAMS_FILE="./alignment_params.txt"

# 调试模式
DEBUG=false  # 设置为 true 可查看详细theta参数

# Checkpoint目录（可选，如果不指定则使用 checkpoints/${MODEL_NAME}）
# CHECKPOINT_DIR=""

# 类别ID（可选，如果不指定则处理所有类别）
CLASS_ID=""  # 例如: CLASS_ID="0001"，留空则处理所有类别

# 视图类型
VIEW_TYPE="drone"  # "drone" 或 "satellite"

# ============================================
# 脚本逻辑
# ============================================

# 创建输出目录（并解析为绝对路径）
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"

# 检查数据集路径是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "错误: 数据集路径不存在: $DATASET_ROOT"
    echo "请修改脚本中的 DATASET_ROOT 变量"
    exit 1
fi

# 构建命令参数
CMD_ARGS=(
    --dataset_root "$DATASET_ROOT"
    --view_type "$VIEW_TYPE"
    --name "$MODEL_NAME"
    --epoch "$EPOCH"
    --gpu_ids "$GPU_IDS"
    --h "$IMG_HEIGHT"
    --w "$IMG_WIDTH"
    --output_dir "$OUTPUT_DIR_ABS"
    --save_params_only
)

# 如果指定了CLASS_ID，添加该参数；否则不添加，脚本会自动处理所有类别
if [ -n "$CLASS_ID" ]; then
    CMD_ARGS+=(--class_id "$CLASS_ID")
fi

# 如果指定了参数文件路径，添加该参数
if [ -n "$PARAMS_FILE" ]; then
    CMD_ARGS+=(--params_file "$PARAMS_FILE")
fi

# 添加checkpoint目录（如果指定）
if [ -n "$CHECKPOINT_DIR" ]; then
    CMD_ARGS+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

# 添加debug选项（如果启用）
if [ "$DEBUG" = "true" ]; then
    CMD_ARGS+=(--debug)
fi

# 打印配置信息
echo "=========================================="
echo "对齐参数保存配置:"
echo "  数据集路径: $DATASET_ROOT"
echo "  模型名称: $MODEL_NAME"
echo "  Epoch: $EPOCH"
echo "  GPU IDs: $GPU_IDS"
echo "  视图类型: $VIEW_TYPE"
if [ -n "$CLASS_ID" ]; then
    echo "  类别ID: $CLASS_ID (只处理该类别)"
else
    echo "  类别ID: 所有类别 (自动遍历)"
fi
if [ -n "$PARAMS_FILE" ]; then
    echo "  参数文件: $PARAMS_FILE"
else
    echo "  参数文件: $OUTPUT_DIR_ABS/alignment_params.txt (默认)"
fi
echo "=========================================="
echo ""

# 运行批量处理脚本
python compare_alignment_batch.py "${CMD_ARGS[@]}" "$@"

if [ $? -eq 0 ]; then
    echo ""
    if [ -n "$PARAMS_FILE" ]; then
        echo "✓ 参数已保存到: $PARAMS_FILE"
    else
        echo "✓ 参数已保存到: $OUTPUT_DIR_ABS/alignment_params.txt"
        # 方便你确认文件确实生成
        if [ -f "$OUTPUT_DIR_ABS/alignment_params.txt" ]; then
            ls -lh "$OUTPUT_DIR_ABS/alignment_params.txt"
        fi
    fi
else
    echo ""
    echo "✗ 处理失败，请检查错误信息"
    exit 1
fi
