#!/bin/bash

# 简化版图像对齐脚本 - 示例配置
# 这个文件展示了如何在脚本中直接设置图像路径

# ============================================
# 配置区域：在这里直接设置图像路径
# ============================================
# 修改下面的路径为你的实际图像路径
SOURCE_IMAGE="path/to/your/source_image.jpg"      # 源图像路径（需要对齐的图像）
REFERENCE_IMAGE="path/to/your/reference_image.jpg"  # 参考图像路径（对齐目标）
OUTPUT_IMAGE=""                                     # 输出图像路径（可选，留空则自动生成）
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
    echo ""
    echo "方式2: 在脚本中设置路径（编辑脚本顶部的配置区域）"
    echo "  设置 SOURCE_IMAGE 和 REFERENCE_IMAGE 变量"
    exit 1
fi

# 如果没有指定输出文件名，自动生成
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="aligned_$(date +%Y%m%d_%H%M%S).jpg"
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
python3 solution1_feature_matching.py "$SOURCE" "$REFERENCE" -o "$OUTPUT"

if [[ $? -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "✅ 成功！结果已保存到: $OUTPUT"
else
    echo ""
    echo "❌ 失败！请检查图像是否相似或有足够的特征点"
    exit 1
fi
