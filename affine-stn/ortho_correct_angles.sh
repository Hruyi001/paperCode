#!/bin/bash

# 基于倾斜角度直接校正脚本
# 不需要参考图，直接根据给定的角度进行校正

# ==========================================
# 配置区域 - 请在这里修改参数
# ==========================================
TILTED_IMAGE="/root/dataset/University-Release/test/query_drone/0000/image-43.jpeg"      # 倾斜无人机图像路径
# 示例: TILTED_IMAGE="/path/to/drone_tilted.jpg"

PITCH_ANGLE="25.5"       # 俯仰角（度），例如: 25.5
# 示例: PITCH_ANGLE="25.5"

ROLL_ANGLE="0"        # 翻滚角（度），例如: 12.3
# 示例: ROLL_ANGLE="12.3"

OUTPUT_IMAGE="./output/corrected.jpg"      # 输出图像路径（可选）
# 示例: OUTPUT_IMAGE="./output/corrected.jpg"

FOCAL_LENGTH=""      # 焦距（像素，可选，留空则自动估计）
# 示例: FOCAL_LENGTH="2000"
# ==========================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用说明
show_usage() {
    echo "=========================================="
    echo "基于角度校正工具"
    echo "=========================================="
    echo ""
    echo "使用方法1（推荐）:"
    echo "  在脚本开头配置参数，然后运行: $0"
    echo ""
    echo "使用方法2:"
    echo "  $0 <倾斜图> <pitch角度> <roll角度> [输出路径] [焦距]"
    echo ""
    echo "参数说明:"
    echo "  倾斜图      : 需要校正的倾斜无人机图像路径"
    echo "  pitch角度   : 俯仰角（度）"
    echo "  roll角度    : 翻滚角（度）"
    echo "  输出路径    : (可选) 输出图像路径"
    echo "  焦距        : (可选) 焦距（像素），留空则自动估计"
    echo ""
    echo "示例:"
    echo "  $0 drone_tilted.jpg 25.5 12.3"
    echo "  $0 drone_tilted.jpg 25.5 12.3 output.jpg 2000"
    echo ""
}

# 处理参数
if [ $# -ge 3 ]; then
    # 使用命令行参数
    TILTED_IMAGE="$1"
    PITCH_ANGLE="$2"
    ROLL_ANGLE="$3"
    OUTPUT_IMAGE="${4:-}"
    FOCAL_LENGTH="${5:-}"
elif [ -n "$TILTED_IMAGE" ] && [ -n "$PITCH_ANGLE" ] && [ -n "$ROLL_ANGLE" ]; then
    # 使用脚本中配置的参数
    # 参数已设置
    :
else
    # 既没有命令行参数，也没有配置参数
    print_error "请配置参数！"
    echo ""
    echo "方法1: 在脚本开头修改 TILTED_IMAGE, PITCH_ANGLE, ROLL_ANGLE 变量"
    echo "方法2: 通过命令行参数传入: $0 <倾斜图> <pitch角度> <roll角度> [输出路径] [焦距]"
    echo ""
    show_usage
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$TILTED_IMAGE" ]; then
    print_error "倾斜无人机图像不存在: $TILTED_IMAGE"
    exit 1
fi

# 检查角度参数
if ! [[ "$PITCH_ANGLE" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
    print_error "无效的俯仰角: $PITCH_ANGLE"
    exit 1
fi

if ! [[ "$ROLL_ANGLE" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
    print_error "无效的翻滚角: $ROLL_ANGLE"
    exit 1
fi

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    print_error "未找到 python3，请先安装 Python"
    exit 1
fi

# 检查依赖包
print_info "检查依赖包..."
if ! python3 -c "import cv2" 2>/dev/null; then
    print_warning "未找到 opencv-python，正在安装..."
    pip3 install opencv-python numpy
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    print_warning "未找到 numpy，正在安装..."
    pip3 install numpy
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/ortho_correction.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "未找到处理脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 显示处理信息
echo ""
print_info "开始处理..."
echo "  倾斜无人机图: $TILTED_IMAGE"
echo "  俯仰角 (Pitch): ${PITCH_ANGLE}°"
echo "  翻滚角 (Roll): ${ROLL_ANGLE}°"
if [ -n "$OUTPUT_IMAGE" ]; then
    echo "  输出图像    : $OUTPUT_IMAGE"
fi
if [ -n "$FOCAL_LENGTH" ]; then
    echo "  焦距        : $FOCAL_LENGTH 像素"
fi
echo ""

# 构建命令
CMD="python3 \"$PYTHON_SCRIPT\" --angles \"$TILTED_IMAGE\" $PITCH_ANGLE $ROLL_ANGLE"
if [ -n "$OUTPUT_IMAGE" ]; then
    CMD="$CMD \"$OUTPUT_IMAGE\""
fi
if [ -n "$FOCAL_LENGTH" ]; then
    CMD="$CMD $FOCAL_LENGTH"
fi

# 执行处理
eval $CMD

# 检查处理结果
if [ $? -eq 0 ]; then
    echo ""
    print_success "处理完成！"
    if [ -n "$OUTPUT_IMAGE" ]; then
        if [ -f "$OUTPUT_IMAGE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_IMAGE" | cut -f1)
            print_info "输出图像已保存到: $OUTPUT_IMAGE (大小: $FILE_SIZE)"
        fi
    fi
else
    print_error "处理失败，请检查错误信息"
    exit 1
fi
