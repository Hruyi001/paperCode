#!/bin/bash

# 方案5: 使用scikit-image的仿射变换脚本
# 用法: ./align_solution5.sh <源图像> [选项]
# 或者直接在脚本中设置图像路径和变换参数

# ============================================
# 配置区域：在这里直接设置图像路径和变换参数
# ============================================
SOURCE_IMAGE=""      # 源图像路径

# 变换参数（选择一种方式）
# 方式1: 使用参数变换（旋转、平移、缩放、剪切）
ROTATION=""          # 旋转角度（度），例如: 15
TRANSLATION_X=""     # X方向平移，例如: 20
TRANSLATION_Y=""     # Y方向平移，例如: 10
SCALE=""             # 缩放因子，例如: 1.1 或 "1.1,1.2" (分别指定x和y)
SHEAR=""             # 剪切角度（度），例如: 5

# 方式2: 从对应点估计（需要提供点坐标文件）
POINTS_FILE=""       # 包含对应点的文件路径（JSON格式）

# 方式3: 从变换矩阵文件
MATRIX_FILE=""       # 包含变换矩阵的文件路径

# 输出设置
OUTPUT_IMAGE=""      # 输出路径（可选）
OUTPUT_WIDTH=""      # 输出图像宽度（可选）
OUTPUT_HEIGHT=""     # 输出图像高度（可选）
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 帮助信息
show_help() {
    echo "方案5: scikit-image仿射变换工具"
    echo ""
    echo "用法:"
    echo "  $0 <源图像> [选项]"
    echo ""
    echo "变换方式（三选一）:"
    echo "  方式1: 参数变换"
    echo "    -r, --rotation ANGLE       旋转角度（度）"
    echo "    -t, --translation X,Y      平移 (X, Y)"
    echo "    -s, --scale FACTOR         缩放因子（或 X,Y）"
    echo "    --shear ANGLE              剪切角度（度）"
    echo ""
    echo "  方式2: 从对应点估计"
    echo "    -p, --points FILE          对应点文件（JSON格式）"
    echo ""
    echo "  方式3: 从变换矩阵"
    echo "    -m, --matrix FILE          变换矩阵文件"
    echo ""
    echo "输出选项:"
    echo "  -o, --output PATH            输出路径"
    echo "  -w, --width WIDTH           输出图像宽度"
    echo "  -h, --height HEIGHT         输出图像高度"
    echo "  --help                       显示帮助"
    echo ""
    echo "示例:"
    echo "  # 旋转15度"
    echo "  $0 image.jpg -r 15 -o result.jpg"
    echo ""
    echo "  # 旋转+平移+缩放"
    echo "  $0 image.jpg -r 15 -t 20,10 -s 1.1 -o result.jpg"
    echo ""
    echo "  # 从对应点估计"
    echo "  $0 image.jpg -p points.json -o result.jpg"
    echo ""
}

# 解析命令行参数
CMD_SOURCE_IMAGE=""
USE_PARAMS=false
USE_POINTS=false
USE_MATRIX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rotation)
            ROTATION="$2"
            USE_PARAMS=true
            shift 2
            ;;
        -t|--translation)
            TRANSLATION="$2"
            USE_PARAMS=true
            shift 2
            ;;
        -s|--scale)
            SCALE="$2"
            USE_PARAMS=true
            shift 2
            ;;
        --shear)
            SHEAR="$2"
            USE_PARAMS=true
            shift 2
            ;;
        -p|--points)
            POINTS_FILE="$2"
            USE_POINTS=true
            shift 2
            ;;
        -m|--matrix)
            MATRIX_FILE="$2"
            USE_MATRIX=true
            shift 2
            ;;
        -o|--output)
            OUTPUT_IMAGE="$2"
            shift 2
            ;;
        -w|--width)
            OUTPUT_WIDTH="$2"
            shift 2
            ;;
        --height)
            OUTPUT_HEIGHT="$2"
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
            else
                echo -e "${RED}错误: 多余的参数 $1${NC}"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 确定源图像路径
if [[ -n "$CMD_SOURCE_IMAGE" ]]; then
    SOURCE_IMAGE="$CMD_SOURCE_IMAGE"
fi

# 检查必需参数
if [[ -z "$SOURCE_IMAGE" ]]; then
    echo -e "${RED}错误: 缺少源图像路径${NC}"
    echo ""
    echo "请使用以下方式之一："
    echo "  1. 命令行参数: $0 <源图像> [选项]"
    echo "  2. 在脚本中设置: 编辑脚本顶部的 SOURCE_IMAGE 变量"
    echo ""
    show_help
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$SOURCE_IMAGE" ]]; then
    echo -e "${RED}错误: 源图像文件不存在: $SOURCE_IMAGE${NC}"
    exit 1
fi

# 确定使用的变换方式
if [[ -n "$POINTS_FILE" ]] || [[ -n "$MATRIX_FILE" ]] || [[ "$USE_PARAMS" == true ]]; then
    # 命令行指定了方式
    :
elif [[ -n "$POINTS_FILE" ]] || [[ -n "$MATRIX_FILE" ]] || [[ -n "$ROTATION" ]] || [[ -n "$TRANSLATION_X" ]] || [[ -n "$SCALE" ]]; then
    # 脚本中设置了参数
    if [[ -n "$POINTS_FILE" ]]; then
        USE_POINTS=true
    elif [[ -n "$MATRIX_FILE" ]]; then
        USE_MATRIX=true
    else
        USE_PARAMS=true
    fi
else
    echo -e "${RED}错误: 必须指定一种变换方式${NC}"
    echo ""
    echo "请选择以下方式之一："
    echo "  1. 参数变换: 使用 -r, -t, -s, --shear 选项"
    echo "  2. 从对应点: 使用 -p 选项指定点文件"
    echo "  3. 从矩阵: 使用 -m 选项指定矩阵文件"
    echo ""
    show_help
    exit 1
fi

# 处理平移参数
if [[ -n "$TRANSLATION" ]]; then
    IFS=',' read -r TRANSLATION_X TRANSLATION_Y <<< "$TRANSLATION"
fi

# 处理输出路径
if [[ -z "$OUTPUT_IMAGE" ]]; then
    SOURCE_BASENAME=$(basename "$SOURCE_IMAGE" | sed 's/\.[^.]*$//')
    OUTPUT_IMAGE="transformed_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
elif [[ -d "$OUTPUT_IMAGE" ]] || [[ ! "$OUTPUT_IMAGE" =~ \.[^.]+$ ]]; then
    OUTPUT_DIR="$OUTPUT_IMAGE"
    mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="."
    SOURCE_BASENAME=$(basename "$SOURCE_IMAGE" | sed 's/\.[^.]*$//')
    OUTPUT_IMAGE="${OUTPUT_DIR}/transformed_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
else
    OUTPUT_DIR=$(dirname "$OUTPUT_IMAGE")
    if [[ -n "$OUTPUT_DIR" ]] && [[ "$OUTPUT_DIR" != "." ]]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    if [[ ! "$OUTPUT_IMAGE" =~ \.[^.]+$ ]]; then
        OUTPUT_IMAGE="${OUTPUT_IMAGE}.jpg"
    fi
fi

# 显示配置信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}方案5: scikit-image仿射变换${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "源图像:     ${GREEN}$SOURCE_IMAGE${NC}"
echo -e "输出路径:   ${YELLOW}$OUTPUT_IMAGE${NC}"

if [[ "$USE_PARAMS" == true ]]; then
    echo -e "变换方式:   ${YELLOW}参数变换${NC}"
    [[ -n "$ROTATION" ]] && echo -e "  旋转:     ${YELLOW}${ROTATION}度${NC}"
    [[ -n "$TRANSLATION_X" ]] && echo -e "  平移:     ${YELLOW}(${TRANSLATION_X}, ${TRANSLATION_Y})${NC}"
    [[ -n "$SCALE" ]] && echo -e "  缩放:     ${YELLOW}${SCALE}${NC}"
    [[ -n "$SHEAR" ]] && echo -e "  剪切:     ${YELLOW}${SHEAR}度${NC}"
elif [[ "$USE_POINTS" == true ]]; then
    echo -e "变换方式:   ${YELLOW}从对应点估计${NC}"
    echo -e "  点文件:   ${YELLOW}$POINTS_FILE${NC}"
elif [[ "$USE_MATRIX" == true ]]; then
    echo -e "变换方式:   ${YELLOW}从变换矩阵${NC}"
    echo -e "  矩阵文件: ${YELLOW}$MATRIX_FILE${NC}"
fi

[[ -n "$OUTPUT_WIDTH" ]] && echo -e "输出宽度:   ${YELLOW}${OUTPUT_WIDTH}${NC}"
[[ -n "$OUTPUT_HEIGHT" ]] && echo -e "输出高度:   ${YELLOW}${OUTPUT_HEIGHT}${NC}"
echo ""

# 构建Python命令
PYTHON_SCRIPT=$(cat << 'PYTHON_EOF'
import sys
import numpy as np
from skimage import transform as tf
from skimage import io, img_as_float
import cv2
import json
import os

def main():
    source_path = sys.argv[1]
    output_path = sys.argv[2]
    transform_type = sys.argv[3]
    
    # 读取图像
    source = cv2.imread(source_path)
    if source is None:
        print(f"错误: 无法读取源图像: {source_path}")
        sys.exit(1)
    
    # 转换为RGB（scikit-image使用RGB）
    if len(source.shape) == 3:
        source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    else:
        source_rgb = source
    
    # 转换为浮点数
    source_float = img_as_float(source_rgb)
    
    # 确定输出形状
    if len(sys.argv) > 5 and sys.argv[4] and sys.argv[5]:
        output_shape = (int(sys.argv[5]), int(sys.argv[4]))  # (height, width)
    else:
        output_shape = source.shape[:2]
    
    # 应用变换
    if transform_type == "params":
        # 参数变换
        transform = tf.AffineTransform()
        
        params = json.loads(sys.argv[6])
        if 'rotation' in params and params['rotation']:
            angle_rad = np.deg2rad(float(params['rotation']))
            transform = transform + tf.AffineTransform(rotation=angle_rad)
        
        if 'translation' in params and params['translation']:
            tx, ty = params['translation']
            transform = transform + tf.AffineTransform(translation=(float(tx), float(ty)))
        
        if 'scale' in params and params['scale']:
            scale = params['scale']
            if isinstance(scale, list):
                sx, sy = float(scale[0]), float(scale[1])
            else:
                sx = sy = float(scale)
            transform = transform + tf.AffineTransform(scale=(sx, sy))
        
        if 'shear' in params and params['shear']:
            shear_rad = np.deg2rad(float(params['shear']))
            transform = transform + tf.AffineTransform(shear=shear_rad)
    
    elif transform_type == "points":
        # 从对应点估计
        points_file = sys.argv[6]
        with open(points_file, 'r') as f:
            points_data = json.load(f)
        
        src_points = np.array(points_data['source_points'], dtype=np.float32)
        dst_points = np.array(points_data['destination_points'], dtype=np.float32)
        
        transform = tf.AffineTransform()
        transform.estimate(src_points, dst_points)
    
    elif transform_type == "matrix":
        # 从矩阵文件
        matrix_file = sys.argv[6]
        matrix = np.loadtxt(matrix_file)
        
        if matrix.shape == (2, 3):
            matrix_3x3 = np.vstack([matrix, [0, 0, 1]])
        else:
            matrix_3x3 = matrix
        
        transform = tf.AffineTransform(matrix=matrix_3x3)
    
    # 应用变换
    transformed = tf.warp(
        source_float,
        transform,
        output_shape=output_shape,
        order=3,  # 双三次插值
        mode='constant',
        cval=0.0
    )
    
    # 转换回BGR和uint8
    if source.dtype == np.uint8:
        transformed = (transformed * 255).astype(np.uint8)
    else:
        transformed = (transformed * 65535).astype(np.uint16)
    
    # 转换回BGR（OpenCV格式）
    if len(transformed.shape) == 3:
        transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    else:
        transformed_bgr = transformed
    
    # 保存结果
    success = cv2.imwrite(output_path, transformed_bgr)
    if success:
        print(f"✅ 变换完成！结果已保存到: {output_path}")
        print(f"\n变换矩阵:")
        print(transform.params)
    else:
        print(f"❌ 保存失败: {output_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
PYTHON_EOF
)

# 创建临时Python脚本
TEMP_SCRIPT=$(mktemp /tmp/align_solution5_XXXXXX.py)
echo "$PYTHON_SCRIPT" > "$TEMP_SCRIPT"

# 构建参数
PYTHON_ARGS=("$SOURCE_IMAGE" "$OUTPUT_IMAGE")

if [[ "$USE_PARAMS" == true ]]; then
    # 构建参数字典
    PARAMS="{"
    [[ -n "$ROTATION" ]] && PARAMS="${PARAMS}\"rotation\": $ROTATION, "
    if [[ -n "$TRANSLATION_X" ]]; then
        PARAMS="${PARAMS}\"translation\": [$TRANSLATION_X, $TRANSLATION_Y], "
    fi
    if [[ -n "$SCALE" ]]; then
        if [[ "$SCALE" =~ , ]]; then
            IFS=',' read -r SX SY <<< "$SCALE"
            PARAMS="${PARAMS}\"scale\": [$SX, $SY], "
        else
            PARAMS="${PARAMS}\"scale\": $SCALE, "
        fi
    fi
    [[ -n "$SHEAR" ]] && PARAMS="${PARAMS}\"shear\": $SHEAR, "
    PARAMS="${PARAMS%, }"  # 移除最后的逗号
    PARAMS="${PARAMS}}"
    
    PYTHON_ARGS+=("params" "$OUTPUT_WIDTH" "$OUTPUT_HEIGHT" "$PARAMS")
    
elif [[ "$USE_POINTS" == true ]]; then
    if [[ ! -f "$POINTS_FILE" ]]; then
        echo -e "${RED}错误: 点文件不存在: $POINTS_FILE${NC}"
        rm -f "$TEMP_SCRIPT"
        exit 1
    fi
    PYTHON_ARGS+=("points" "$OUTPUT_WIDTH" "$OUTPUT_HEIGHT" "$POINTS_FILE")
    
elif [[ "$USE_MATRIX" == true ]]; then
    if [[ ! -f "$MATRIX_FILE" ]]; then
        echo -e "${RED}错误: 矩阵文件不存在: $MATRIX_FILE${NC}"
        rm -f "$TEMP_SCRIPT"
        exit 1
    fi
    PYTHON_ARGS+=("matrix" "$OUTPUT_WIDTH" "$OUTPUT_HEIGHT" "$MATRIX_FILE")
fi

# 执行Python脚本
echo -e "${BLUE}执行变换...${NC}"
python3 "$TEMP_SCRIPT" "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

# 清理临时文件
rm -f "$TEMP_SCRIPT"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}处理完成！${NC}"
    echo -e "${BLUE}========================================${NC}"
else
    echo ""
    echo -e "${RED}❌ 处理失败！${NC}"
    exit 1
fi
