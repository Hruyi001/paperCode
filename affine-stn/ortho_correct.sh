#!/bin/bash

# 倾斜无人机视图转正射视图校正脚本
# 
# 使用方法1（推荐）: 直接在下面配置路径，然后运行 ./ortho_correct.sh
# 使用方法2: 通过命令行参数传入路径 ./ortho_correct.sh <倾斜图> <参考图> [输出路径]

# ==========================================
# 配置区域 - 请在这里修改图像路径
# ==========================================
# 方法：取消下面三行的注释，并填入你的图像路径

TILTED_IMAGE="/root/dataset/University-Release/test/query_drone/0000/image-43.jpeg"      # 倾斜无人机图像路径
# 示例: TILTED_IMAGE="/home/user/images/drone_tilted.jpg"
# 示例: TILTED_IMAGE="./data/drone_tilted.jpg"

REFERENCE_IMAGE="/root/dataset/University-Release/test/gallery_satellite/0000/0000.jpg"   # 参考正射卫星图像路径
# 示例: REFERENCE_IMAGE="/home/user/images/satellite_reference.jpg"
# 示例: REFERENCE_IMAGE="./data/satellite_reference.jpg"

OUTPUT_IMAGE="./output"      # 输出图像路径（可选，留空则默认为 corrected_output.jpg）
# 示例: OUTPUT_IMAGE="./output/corrected_result.jpg"
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
    echo "倾斜无人机视图转正射视图校正工具"
    echo "=========================================="
    echo ""
    echo "使用方法1（推荐）:"
    echo "  在脚本开头配置路径，然后运行: $0"
    echo ""
    echo "使用方法2:"
    echo "  $0 <倾斜无人机图> <参考正射图> [输出路径]"
    echo ""
    echo "参数说明:"
    echo "  倾斜无人机图  : 需要校正的倾斜无人机图像路径"
    echo "  参考正射图    : 参考的正射卫星图像路径"
    echo "  输出路径      : (可选) 输出图像路径，默认为 corrected_output.jpg"
    echo ""
    echo "示例:"
    echo "  $0 drone_tilted.jpg satellite_reference.jpg"
    echo "  $0 drone_tilted.jpg satellite_reference.jpg output.jpg"
    echo ""
}

# 处理参数：如果通过命令行传入参数，则使用命令行参数；否则使用脚本中配置的路径
if [ $# -ge 2 ]; then
    # 使用命令行参数
    TILTED_IMAGE="$1"
    REFERENCE_IMAGE="$2"
    OUTPUT_IMAGE="${3:-corrected_output.jpg}"
elif [ -n "$TILTED_IMAGE" ] && [ -n "$REFERENCE_IMAGE" ]; then
    # 使用脚本中配置的路径
    OUTPUT_IMAGE="${OUTPUT_IMAGE:-corrected_output.jpg}"
else
    # 既没有命令行参数，也没有配置路径
    print_error "请配置图像路径！"
    echo ""
    echo "方法1: 在脚本开头修改 TILTED_IMAGE 和 REFERENCE_IMAGE 变量"
    echo "方法2: 通过命令行参数传入: $0 <倾斜图> <参考图> [输出路径]"
    echo ""
    show_usage
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$TILTED_IMAGE" ]; then
    print_error "倾斜无人机图像不存在: $TILTED_IMAGE"
    exit 1
fi

if [ ! -f "$REFERENCE_IMAGE" ]; then
    print_error "参考正射图像不存在: $REFERENCE_IMAGE"
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
echo "  参考正射图  : $REFERENCE_IMAGE"
echo "  输出图像    : $OUTPUT_IMAGE"
echo ""

# 执行处理
python3 "$PYTHON_SCRIPT" "$TILTED_IMAGE" "$REFERENCE_IMAGE" "$OUTPUT_IMAGE"

# 检查处理结果
if [ $? -eq 0 ]; then
    echo ""
    print_success "处理完成！"
    print_info "输出图像已保存到: $OUTPUT_IMAGE"
    
    # 检查输出文件是否存在
    if [ -f "$OUTPUT_IMAGE" ]; then
        # 获取文件大小
        FILE_SIZE=$(du -h "$OUTPUT_IMAGE" | cut -f1)
        print_info "输出文件大小: $FILE_SIZE"
    fi
else
    print_error "处理失败，请检查错误信息"
    exit 1
fi
