#!/bin/bash

# ============================================
# Safe-Net 原图与对齐图对比脚本
# 方便指定数据集路径和其他参数
# ============================================

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

# 模式选择: 1=drone->satellite (输入为无人机图), 2=satellite->drone (输入为卫星图)
MODE=1

# 输出目录
OUTPUT_DIR="./alignment_comparisons"

# 调试模式
DEBUG=false  # 设置为 true 可查看详细theta参数

# Checkpoint目录（可选，如果不指定则使用 checkpoints/${MODEL_NAME}）
# CHECKPOINT_DIR=""

# ============================================
# 使用方式配置
# ============================================

# 方式1: 指定单个图像路径
# IMG_PATH="${DATASET_ROOT}/query_drone/0001/0001_001.jpg"

# 方式2: 指定多个图像路径（逗号分隔）
# IMG_PATH="${DATASET_ROOT}/query_drone/0001/0001_001.jpg,${DATASET_ROOT}/query_drone/0001/0001_002.jpg"

# 方式3: 从数据集自动选择（推荐）
# 指定类别ID和图像类型，会遍历该类别下的所有图像
# 如果不指定CLASS_ID，将处理query_drone下的所有类别和所有图片
CLASS_ID=""  # 类别ID（如 "0001"），留空则处理所有类别
VIEW_TYPE="drone"  # "drone" 或 "satellite"
# 注意: 会遍历指定类别（或所有类别）下的所有图像，每张图像生成一个对比图

# ============================================
# 脚本逻辑
# ============================================

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查数据集路径是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "错误: 数据集路径不存在: $DATASET_ROOT"
    echo "请修改脚本中的 DATASET_ROOT 变量"
    exit 1
fi

# 如果指定了VIEW_TYPE，使用批量处理脚本
if [ -n "$VIEW_TYPE" ]; then
    # 使用批量处理脚本
    # 如果指定了CLASS_ID，只处理该类别；否则处理所有类别
    CMD_ARGS=(
        --dataset_root "$DATASET_ROOT"
        --view_type "$VIEW_TYPE"
        --name "$MODEL_NAME"
        --epoch "$EPOCH"
        --gpu_ids "$GPU_IDS"
        --h "$IMG_HEIGHT"
        --w "$IMG_WIDTH"
        --output_dir "$OUTPUT_DIR"
    )
    
    # 如果指定了CLASS_ID，添加该参数；否则不添加，脚本会自动处理所有类别
    if [ -n "$CLASS_ID" ]; then
        CMD_ARGS+=(--class_id "$CLASS_ID")
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
    echo "配置信息:"
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
    echo "  输出目录: $OUTPUT_DIR"
    echo "=========================================="
    echo ""
    
    # 运行批量处理脚本
    python compare_alignment_batch.py "${CMD_ARGS[@]}" "$@"
    
    exit $?
fi

# 如果没有指定IMG_PATH且没有指定VIEW_TYPE，提示用户
if [ -z "$IMG_PATH" ] && [ -z "$VIEW_TYPE" ]; then
    echo "=========================================="
    echo "请选择使用方式："
    echo ""
    echo "方式1: 在脚本中设置 IMG_PATH 变量（单张或多张图像）"
    echo "方式2: 在脚本中设置 VIEW_TYPE 变量（推荐）"
    echo "      - 设置 CLASS_ID 处理指定类别"
    echo "      - 不设置 CLASS_ID 处理所有类别"
    echo "方式3: 直接通过命令行参数指定"
    echo ""
    echo "示例命令行用法："
    echo "  $0 --img path/to/image.jpg"
    echo "  $0 --class 0001 --view drone  # 处理指定类别下所有图像"
    echo "  $0 --view drone                # 处理所有类别下所有图像"
    echo "=========================================="
    exit 1
fi

# 构建checkpoint路径
if [ -n "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_ARG="--checkpoint_dir $CHECKPOINT_DIR"
else
    CHECKPOINT_ARG=""
fi

# 构建输出文件名（如果未指定）
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/alignment_comparison_$(date +%Y%m%d_%H%M%S).png"
fi

# 打印配置信息
echo "=========================================="
echo "配置信息:"
echo "  数据集路径: $DATASET_ROOT"
echo "  模型名称: $MODEL_NAME"
echo "  Epoch: $EPOCH"
echo "  GPU IDs: $GPU_IDS"
echo "  模式: $MODE ($([ $MODE -eq 1 ] && echo 'drone->satellite' || echo 'satellite->drone'))"
echo "  图像路径: $IMG_PATH"
echo "  输出文件: $OUTPUT_FILE"
echo "=========================================="
echo ""

# 运行对比脚本
python compare_alignment.py \
    --name "$MODEL_NAME" \
    --img "$IMG_PATH" \
    --epoch "$EPOCH" \
    --gpu_ids "$GPU_IDS" \
    --h "$IMG_HEIGHT" \
    --w "$IMG_WIDTH" \
    --mode "$MODE" \
    --output "$OUTPUT_FILE" \
    $CHECKPOINT_ARG \
    "$@"  # 传递额外的命令行参数

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 对比图已保存到: $OUTPUT_FILE"
else
    echo ""
    echo "✗ 生成失败，请检查错误信息"
    exit 1
fi
