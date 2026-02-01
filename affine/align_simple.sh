#!/bin/bash

# 简化版图像对齐脚本
# 用法: ./align_simple.sh [源图像] [参考图像] [输出图像]
# 或者直接在脚本中设置图像路径（见下方配置区域）

# ============================================
# 配置区域：在这里直接设置图像路径
# ============================================
# 如果命令行没有提供参数，将使用这里设置的路径
SOURCE_IMAGE="/root/dataset/University-Release/test/query_drone/0000/image-43.jpeg"      # 源图像路径（需要对齐的图像）
REFERENCE_IMAGE="/root/dataset/University-Release/test/gallery_satellite/0000/0000.jpg"    # 参考图像路径（对齐目标）
OUTPUT_IMAGE="output"       # 输出路径（可选）：
                           #   - 留空：自动生成文件名
                           #   - 目录路径：在目录中自动生成文件名（如 "output" 或 "./results"）
                           #   - 文件路径：保存到指定文件（如 "result.jpg" 或 "./output/aligned.jpg"）
# ============================================

# 如果命令行提供了参数，则使用命令行参数
if [[ $# -ge 2 ]]; then
    SOURCE="$1"
    REFERENCE="$2"
    OUTPUT="${3:-}"
elif [[ -n "$SOURCE_IMAGE" ]] && [[ -n "$REFERENCE_IMAGE" ]]; then
    # 使用脚本中设置的路径
    SOURCE="$SOURCE_IMAGE"
    REFERENCE="$REFERENCE_IMAGE"
    OUTPUT="$OUTPUT_IMAGE"
else
    echo "用法: $0 [源图像] [参考图像] [输出图像]"
    echo ""
    echo "方式1: 命令行参数"
    echo "  $0 source.jpg reference.jpg"
    echo "  $0 source.jpg reference.jpg result.jpg"
    echo ""
    echo "方式2: 在脚本中设置路径（编辑脚本顶部的配置区域）"
    echo "  设置 SOURCE_IMAGE 和 REFERENCE_IMAGE 变量"
    exit 1
fi

# 处理输出路径
if [[ -z "$OUTPUT" ]]; then
    # 如果没有指定输出，自动生成文件名
    OUTPUT="aligned_$(date +%Y%m%d_%H%M%S).jpg"
elif [[ -d "$OUTPUT" ]] || [[ ! "$OUTPUT" =~ \.[^.]+$ ]]; then
    # 如果OUTPUT是目录或没有扩展名，在目录中生成文件名
    if [[ -d "$OUTPUT" ]]; then
        OUTPUT_DIR="$OUTPUT"
    else
        # 可能是目录路径但没有尾随斜杠
        OUTPUT_DIR="$OUTPUT"
        mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="."
    fi
    # 从源文件名生成输出文件名
    SOURCE_BASENAME=$(basename "$SOURCE" | sed 's/\.[^.]*$//')
    OUTPUT="${OUTPUT_DIR}/aligned_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
else
    # OUTPUT是文件路径，确保目录存在
    OUTPUT_DIR=$(dirname "$OUTPUT")
    if [[ -n "$OUTPUT_DIR" ]] && [[ "$OUTPUT_DIR" != "." ]]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    # 确保有扩展名
    if [[ ! "$OUTPUT" =~ \.[^.]+$ ]]; then
        OUTPUT="${OUTPUT}.jpg"
    fi
fi

# 检查文件
if [[ ! -f "$SOURCE" ]]; then
    echo "错误: 源图像不存在: $SOURCE"
    exit 1
fi

if [[ ! -f "$REFERENCE" ]]; then
    echo "错误: 参考图像不存在: $REFERENCE"
    exit 1
fi

# 执行对齐
echo "对齐图像: $SOURCE -> $REFERENCE"
# 使用 --no-display 避免在无GUI环境中出错，--save-comparison 保存对比图
python3 solution1_feature_matching.py "$SOURCE" "$REFERENCE" -o "$OUTPUT" --no-display --save-comparison

if [[ $? -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "✅ 成功！结果已保存到: $OUTPUT"
else
    echo ""
    echo "❌ 失败！请检查图像是否相似或有足够的特征点"
    exit 1
fi
