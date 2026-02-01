#!/bin/bash

# ============================================
# Safe-Net FAM对齐图像生成脚本
# 根据给定的无人机图像，生成FAM模块处理后的对齐图像
# ============================================

# 保证无论从哪里运行，都在脚本所在目录执行（避免相对路径写到别处）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ============================================
# 配置区域 - 请根据实际情况修改
# ============================================

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 图像尺寸
IMG_HEIGHT=256
IMG_WIDTH=256

# Checkpoint目录（可选，如果不指定则使用 checkpoints/${MODEL_NAME}）
# CHECKPOINT_DIR=""

# 视图模式（1: drone->satellite, 2: satellite->drone）
MODE=1

# 调试模式（打印详细theta参数）
DEBUG=false  # 设置为 true 可查看详细theta参数

# 是否生成对比图（true: 生成, false: 不生成）
GENERATE_COMPARISON=true  # 设置为 false 则不生成对比图

# ============================================
# 使用说明
# ============================================
usage() {
    echo "用法: $0 <输入图像路径> [输出图像路径] [对比图路径]"
    echo ""
    echo "参数说明:"
    echo "  输入图像路径  : 必需，要处理的无人机图像路径"
    echo "  输出图像路径  : 可选，对齐后图像的保存路径"
    echo "                 如果不指定，默认保存为: ./output/<输入图像文件名>_aligned.jpg"
    echo "  对比图路径    : 可选，对比图的保存路径（原图和对齐图并排，带参数标注）"
    echo "                 如果不指定，默认保存为: ./output/<输入图像文件名>_comparison.png"
    echo "                 设置为空字符串 \"\" 则不生成对比图"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/drone_image.jpg"
    echo "  $0 /path/to/drone_image.jpg /path/to/output_aligned.jpg"
    echo "  $0 /path/to/drone_image.jpg /path/to/output_aligned.jpg /path/to/comparison.png"
    echo "  $0 /path/to/drone_image.jpg /path/to/output_aligned.jpg \"\"  # 不生成对比图"
    echo ""
    exit 1
}

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 缺少必需参数"
    usage
fi

INPUT_IMG="$1"
OUTPUT_IMG="$2"
COMPARISON_IMG="$3"

# 检查输入图像是否存在
if [ ! -f "$INPUT_IMG" ]; then
    echo "错误: 输入图像不存在: $INPUT_IMG"
    exit 1
fi

# 如果没有指定输出路径，使用默认路径
if [ -z "$OUTPUT_IMG" ]; then
    # 默认输出路径：项目根目录下的output目录
    OUTPUT_DIR="${SCRIPT_DIR}/output"
    
    # 获取输入图像的文件名
    INPUT_BASE=$(basename "$INPUT_IMG")
    INPUT_NAME="${INPUT_BASE%.*}"
    INPUT_EXT="${INPUT_BASE##*.}"
    
    # 如果扩展名为空，使用jpg
    if [ "$INPUT_EXT" = "$INPUT_BASE" ]; then
        INPUT_EXT="jpg"
    fi
    
    OUTPUT_IMG="${OUTPUT_DIR}/${INPUT_NAME}_aligned.${INPUT_EXT}"
fi

# 设置对比图输出路径
if [ -z "$COMPARISON_IMG" ]; then
    if [ "$GENERATE_COMPARISON" = "true" ]; then
        # 默认生成对比图
        COMPARISON_IMG="${OUTPUT_DIR}/${INPUT_NAME}_comparison.png"
    else
        # 不生成对比图
        COMPARISON_IMG=""
    fi
fi

# 确保输出目录存在
OUTPUT_DIR=$(dirname "$OUTPUT_IMG")
if [ ! -z "$OUTPUT_DIR" ] && [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "已创建输出目录: $OUTPUT_DIR"
fi

echo "=========================================="
echo "Safe-Net FAM对齐图像生成"
echo "=========================================="
echo "模型名称: $MODEL_NAME"
echo "输入图像: $INPUT_IMG"
echo "输出图像: $OUTPUT_IMG"
if [ ! -z "$COMPARISON_IMG" ]; then
    echo "对比图: $COMPARISON_IMG"
else
    echo "对比图: 不生成"
fi
echo "Epoch: $EPOCH"
echo "GPU IDs: $GPU_IDS"
echo "模式: $([ $MODE -eq 1 ] && echo 'drone->satellite' || echo 'satellite->drone')"
echo "=========================================="
echo ""

# 构建Python命令参数
PYTHON_ARGS=(
    --name "$MODEL_NAME"
    --img "$INPUT_IMG"
    --output "$OUTPUT_IMG"
    --epoch $EPOCH
    --gpu_ids "$GPU_IDS"
    --h $IMG_HEIGHT
    --w $IMG_WIDTH
    --mode $MODE
)

# 如果指定了checkpoint目录，添加参数
if [ ! -z "$CHECKPOINT_DIR" ]; then
    PYTHON_ARGS+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

# 如果启用调试模式，添加参数
if [ "$DEBUG" = "true" ]; then
    PYTHON_ARGS+=(--debug)
fi

# 如果指定了对比图路径，添加参数
if [ ! -z "$COMPARISON_IMG" ]; then
    PYTHON_ARGS+=(--comparison "$COMPARISON_IMG")
else
    PYTHON_ARGS+=(--comparison "")
fi

# 运行Python脚本
python generate_aligned_image.py "${PYTHON_ARGS[@]}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 成功生成FAM对齐图像: $OUTPUT_IMG"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ 生成失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi
