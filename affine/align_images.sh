#!/bin/bash

# 图像对齐脚本 - 基于特征点匹配
# 用法: ./align_images.sh <源图像> <参考图像> [选项]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 配置区域：在这里直接设置图像路径和参数
# ============================================
# 图像路径（如果命令行没有提供，将使用这里设置的路径）
SOURCE_IMAGE=""      # 源图像路径（需要对齐的图像）
REFERENCE_IMAGE=""    # 参考图像路径（对齐目标）

# 默认参数
DETECTOR="ORB"        # 特征检测器: SIFT, ORB, AKAZE
OUTPUT_DIR="output"   # 输出目录
SHOW_MATCHES=false    # 是否显示匹配点
MIN_MATCHES=4         # 最少匹配点数量
# ============================================

# 帮助信息
show_help() {
    echo "图像对齐工具 - 基于特征点匹配"
    echo ""
    echo "用法:"
    echo "  $0 <源图像> <参考图像> [选项]"
    echo ""
    echo "参数:"
    echo "  <源图像>      需要对齐的源图像路径"
    echo "  <参考图像>    参考图像路径（对齐目标）"
    echo ""
    echo "选项:"
    echo "  -d, --detector DETECTOR    特征检测器 (SIFT|ORB|AKAZE) [默认: ORB]"
    echo "  -o, --output DIR           输出目录 [默认: output]"
    echo "  -m, --show-matches         显示特征点匹配可视化"
    echo "  -n, --min-matches NUM      最少匹配点数量 [默认: 4]"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 source.jpg reference.jpg"
    echo "  $0 source.jpg reference.jpg -d SIFT -m"
    echo "  $0 source.jpg reference.jpg -o results --show-matches"
    echo ""
}

# 解析命令行参数
# 如果脚本中已设置路径，命令行参数会覆盖脚本中的设置
CMD_SOURCE_IMAGE=""
CMD_REFERENCE_IMAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detector)
            DETECTOR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--show-matches)
            SHOW_MATCHES=true
            shift
            ;;
        -n|--min-matches)
            MIN_MATCHES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$CMD_SOURCE_IMAGE" ]]; then
                CMD_SOURCE_IMAGE="$1"
            elif [[ -z "$CMD_REFERENCE_IMAGE" ]]; then
                CMD_REFERENCE_IMAGE="$1"
            else
                echo -e "${RED}错误: 多余的参数 $1${NC}"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 确定最终使用的图像路径（命令行参数优先，否则使用脚本中的设置）
if [[ -n "$CMD_SOURCE_IMAGE" ]]; then
    SOURCE_IMAGE="$CMD_SOURCE_IMAGE"
fi

if [[ -n "$CMD_REFERENCE_IMAGE" ]]; then
    REFERENCE_IMAGE="$CMD_REFERENCE_IMAGE"
fi

# 检查必需参数
if [[ -z "$SOURCE_IMAGE" ]] || [[ -z "$REFERENCE_IMAGE" ]]; then
    echo -e "${RED}错误: 缺少必需参数${NC}"
    echo ""
    echo "请使用以下方式之一："
    echo "  1. 命令行参数: $0 <源图像> <参考图像> [选项]"
    echo "  2. 在脚本中设置: 编辑脚本顶部的 SOURCE_IMAGE 和 REFERENCE_IMAGE 变量"
    echo ""
    show_help
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$SOURCE_IMAGE" ]]; then
    echo -e "${RED}错误: 源图像文件不存在: $SOURCE_IMAGE${NC}"
    exit 1
fi

if [[ ! -f "$REFERENCE_IMAGE" ]]; then
    echo -e "${RED}错误: 参考图像文件不存在: $REFERENCE_IMAGE${NC}"
    exit 1
fi

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3 命令${NC}"
    exit 1
fi

# 检查Python脚本是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/solution1_feature_matching.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}错误: Python脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 生成输出文件名
SOURCE_BASENAME=$(basename "$SOURCE_IMAGE" | sed 's/\.[^.]*$//')
REFERENCE_BASENAME=$(basename "$REFERENCE_IMAGE" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

ALIGNED_OUTPUT="$OUTPUT_DIR/aligned_${SOURCE_BASENAME}_${TIMESTAMP}.jpg"
COMPARISON_OUTPUT="$OUTPUT_DIR/comparison_${SOURCE_BASENAME}_${TIMESTAMP}.jpg"
MATCHES_OUTPUT="$OUTPUT_DIR/matches_${SOURCE_BASENAME}_${TIMESTAMP}.jpg"
TRANSFORM_OUTPUT="$OUTPUT_DIR/transform_${SOURCE_BASENAME}_${TIMESTAMP}.txt"

# 显示配置信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}图像对齐工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "源图像:     ${GREEN}$SOURCE_IMAGE${NC}"
echo -e "参考图像:   ${GREEN}$REFERENCE_IMAGE${NC}"
echo -e "检测器:     ${YELLOW}$DETECTOR${NC}"
echo -e "输出目录:   ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "显示匹配:  ${YELLOW}$SHOW_MATCHES${NC}"
echo ""

# 构建Python命令
PYTHON_CMD="python3 \"$PYTHON_SCRIPT\" \"$SOURCE_IMAGE\" \"$REFERENCE_IMAGE\""
PYTHON_CMD="$PYTHON_CMD -d $DETECTOR"
PYTHON_CMD="$PYTHON_CMD -o \"$ALIGNED_OUTPUT\""
PYTHON_CMD="$PYTHON_CMD --min-matches $MIN_MATCHES"

if [[ "$SHOW_MATCHES" == true ]]; then
    PYTHON_CMD="$PYTHON_CMD --show-matches"
fi

# 执行对齐
echo -e "${BLUE}开始对齐图像...${NC}"
echo ""

if eval "$PYTHON_CMD"; then
    echo ""
    echo -e "${GREEN}✅ 对齐完成！${NC}"
    echo ""
    
    # 检查输出文件
    if [[ -f "$ALIGNED_OUTPUT" ]]; then
        echo -e "输出文件:"
        echo -e "  ${GREEN}对齐图像:${NC} $ALIGNED_OUTPUT"
        
        # 生成对比图（如果Python脚本没有生成）
        if [[ ! -f "$COMPARISON_OUTPUT" ]]; then
            echo ""
            echo -e "${BLUE}生成对比图...${NC}"
            python3 << EOF
import cv2
import numpy as np
import sys

try:
    source = cv2.imread("$SOURCE_IMAGE")
    reference = cv2.imread("$REFERENCE_IMAGE")
    aligned = cv2.imread("$ALIGNED_OUTPUT")
    
    if source is None or reference is None or aligned is None:
        print("无法读取图像文件")
        sys.exit(1)
    
    # 调整图像大小以便并排显示
    h1, w1 = source.shape[:2]
    h2, w2 = reference.shape[:2]
    h3, w3 = aligned.shape[:2]
    
    max_h = max(h1, h2, h3)
    
    # 缩放图像以适合显示
    scale = min(800 / w1, 600 / h1, 1.0)
    if scale < 1.0:
        new_w1 = int(w1 * scale)
        new_h1 = int(h1 * scale)
        source = cv2.resize(source, (new_w1, new_h1))
        w1, h1 = new_w1, new_h1
    
    scale = min(800 / w2, 600 / h2, 1.0)
    if scale < 1.0:
        new_w2 = int(w2 * scale)
        new_h2 = int(h2 * scale)
        reference = cv2.resize(reference, (new_w2, new_h2))
        w2, h2 = new_w2, new_h2
    
    scale = min(800 / w3, 600 / h3, 1.0)
    if scale < 1.0:
        new_w3 = int(w3 * scale)
        new_h3 = int(h3 * scale)
        aligned = cv2.resize(aligned, (new_w3, new_h3))
        w3, h3 = new_w3, new_h3
    
    # 创建对比图
    max_h = max(h1, h2, h3)
    total_w = w1 + w2 + w3
    
    comparison = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    
    # 放置图像
    comparison[:h1, :w1] = source
    comparison[:h2, w1:w1+w2] = reference
    comparison[:h3, w1+w2:w1+w2+w3] = aligned
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w1, w2, w3) / 400.0
    thickness = max(1, int(font_scale * 2))
    
    cv2.putText(comparison, 'Source', (10, 30), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(comparison, 'Reference', (w1 + 10, 30), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(comparison, 'Aligned', (w1 + w2 + 10, 30), font, font_scale, (255, 0, 0), thickness)
    
    cv2.imwrite("$COMPARISON_OUTPUT", comparison)
    print("对比图已保存")
except Exception as e:
    print(f"生成对比图时出错: {e}")
    sys.exit(1)
EOF
        fi
        
        if [[ -f "$COMPARISON_OUTPUT" ]]; then
            echo -e "  ${GREEN}对比图:${NC}   $COMPARISON_OUTPUT"
        fi
    else
        echo -e "${YELLOW}警告: 未找到对齐输出文件${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}处理完成！${NC}"
    echo -e "${BLUE}========================================${NC}"
else
    echo ""
    echo -e "${RED}❌ 对齐失败！${NC}"
    echo ""
    echo "可能的原因:"
    echo "  - 两张图像差异太大"
    echo "  - 特征点不足"
    echo "  - 尝试使用不同的检测器: -d SIFT 或 -d AKAZE"
    exit 1
fi
