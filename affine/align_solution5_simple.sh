#!/bin/bash

# 方案5简化版: scikit-image仿射变换
# 用法: ./align_solution5_simple.sh [源图像] [旋转角度] [输出图像]

# ============================================
# 配置区域：在这里直接设置参数
# ============================================
SOURCE_IMAGE=""      # 源图像路径
ROTATION_ANGLE=""     # 旋转角度（度），例如: 15
OUTPUT_IMAGE=""       # 输出路径（可选）
# ============================================

# 如果命令行提供了参数，则使用命令行参数
if [[ $# -ge 1 ]]; then
    SOURCE="$1"
    ROTATION="${2:-0}"
    OUTPUT="${3:-}"
else
    SOURCE="$SOURCE_IMAGE"
    ROTATION="${ROTATION_ANGLE:-0}"
    OUTPUT="$OUTPUT_IMAGE"
fi

# 检查必需参数
if [[ -z "$SOURCE" ]]; then
    echo "用法: $0 [源图像] [旋转角度] [输出图像]"
    echo ""
    echo "示例:"
    echo "  $0 image.jpg 15"
    echo "  $0 image.jpg 15 result.jpg"
    echo ""
    echo "或在脚本中设置 SOURCE_IMAGE 和 ROTATION_ANGLE"
    exit 1
fi

# 检查文件
if [[ ! -f "$SOURCE" ]]; then
    echo "错误: 源图像不存在: $SOURCE"
    exit 1
fi

# 处理输出路径
if [[ -z "$OUTPUT" ]]; then
    SOURCE_BASENAME=$(basename "$SOURCE" | sed 's/\.[^.]*$//')
    OUTPUT="transformed_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
elif [[ -d "$OUTPUT" ]] || [[ ! "$OUTPUT" =~ \.[^.]+$ ]]; then
    OUTPUT_DIR="$OUTPUT"
    mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="."
    SOURCE_BASENAME=$(basename "$SOURCE" | sed 's/\.[^.]*$//')
    OUTPUT="${OUTPUT_DIR}/transformed_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
else
    OUTPUT_DIR=$(dirname "$OUTPUT")
    if [[ -n "$OUTPUT_DIR" ]] && [[ "$OUTPUT_DIR" != "." ]]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    if [[ ! "$OUTPUT" =~ \.[^.]+$ ]]; then
        OUTPUT="${OUTPUT}.jpg"
    fi
fi

# 执行变换
echo "变换图像: $SOURCE (旋转 ${ROTATION}度)"
python3 << EOF
import sys
import numpy as np
from skimage import transform as tf
from skimage import io, img_as_float
import cv2

source = cv2.imread("$SOURCE")
if source is None:
    print(f"错误: 无法读取图像: $SOURCE")
    sys.exit(1)

# 转换为RGB
if len(source.shape) == 3:
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
else:
    source_rgb = source

# 转换为浮点数
source_float = img_as_float(source_rgb)

# 创建旋转变换
angle_rad = np.deg2rad(float($ROTATION))
transform = tf.AffineTransform(rotation=angle_rad)

# 应用变换
transformed = tf.warp(
    source_float,
    transform,
    output_shape=source.shape[:2],
    order=3,
    mode='constant',
    cval=0.0
)

# 转换回uint8和BGR
transformed = (transformed * 255).astype(np.uint8)
if len(transformed.shape) == 3:
    transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
else:
    transformed_bgr = transformed

# 保存
success = cv2.imwrite("$OUTPUT", transformed_bgr)
if success:
    print("✅ 成功！结果已保存到: $OUTPUT")
else:
    print("❌ 保存失败")
    sys.exit(1)
EOF

if [[ $? -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "✅ 处理完成！"
else
    echo ""
    echo "❌ 处理失败！"
    exit 1
fi
