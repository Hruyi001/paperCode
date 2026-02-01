#!/bin/bash

# 自适应倾斜校正脚本 - 使用单个指定方法
# 用法: ./correct_skew_single_method.sh [输入图像] [方法编号] [输出图像]
# 方法编号: 1-6 (1=投影轮廓法, 2=霍夫直线, 3=最小外接矩形, 4=透视+旋转, 5=频域分析, 6=组合方法)

# ============================================
# 配置区域：在这里直接设置参数
# ============================================
INPUT_IMAGE=""      # 输入图像路径
METHOD_NUM=""       # 方法编号 (1-6)，留空则使用方案6（组合方法，最准确）
OUTPUT_IMAGE=""     # 输出图像路径（可选）
                    #   - 留空：自动生成文件名
                    #   - 目录路径：在目录中自动生成文件名
                    #   - 文件路径：保存到指定文件
# ============================================

# 如果命令行提供了参数，则使用命令行参数
if [[ $# -ge 1 ]]; then
    INPUT="$1"
    METHOD="${2:-6}"  # 默认使用方法6（组合方法）
    OUTPUT="${3:-}"
elif [[ -n "$INPUT_IMAGE" ]]; then
    # 使用脚本中设置的路径
    INPUT="$INPUT_IMAGE"
    METHOD="${METHOD_NUM:-6}"
    OUTPUT="$OUTPUT_IMAGE"
else
    echo "用法: $0 [输入图像] [方法编号] [输出图像]"
    echo ""
    echo "方法编号:"
    echo "  1 - 投影轮廓法（最快，适合文档）"
    echo "  2 - 霍夫直线检测法（通用场景）"
    echo "  3 - 最小外接矩形法（有明显边界）"
    echo "  4 - 投影变换+旋转组合法（复杂变形）"
    echo "  5 - 频域分析法（周期性纹理）"
    echo "  6 - 组合方法（最准确，推荐）"
    echo ""
    echo "示例:"
    echo "  $0 image.jpg 6"
    echo "  $0 image.jpg 1 result.jpg"
    echo "  $0 image.jpg 6 output_folder"
    echo ""
    echo "或在脚本中设置 INPUT_IMAGE 和 METHOD_NUM"
    exit 1
fi

# 检查输入文件
if [[ -z "$INPUT" ]]; then
    echo "错误: 未指定输入图像"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "错误: 输入图像不存在: $INPUT"
    exit 1
fi

# 验证方法编号
if ! [[ "$METHOD" =~ ^[1-6]$ ]]; then
    echo "错误: 方法编号必须是 1-6 之间的数字"
    exit 1
fi

# 方法名称映射
declare -A METHOD_NAMES
METHOD_NAMES[1]="投影轮廓法"
METHOD_NAMES[2]="霍夫直线检测法"
METHOD_NAMES[3]="最小外接矩形法"
METHOD_NAMES[4]="投影变换+旋转组合法"
METHOD_NAMES[5]="频域分析法"
METHOD_NAMES[6]="组合方法"

METHOD_NAME="${METHOD_NAMES[$METHOD]}"

# 处理输出路径
if [[ -z "$OUTPUT" ]]; then
    # 如果没有指定输出，自动生成文件名
    SOURCE_BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
    OUTPUT="corrected_method${METHOD}_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
elif [[ -d "$OUTPUT" ]] || [[ ! "$OUTPUT" =~ \.[^.]+$ ]]; then
    # 如果OUTPUT是目录或没有扩展名，在目录中生成文件名
    if [[ -d "$OUTPUT" ]]; then
        OUTPUT_DIR="$OUTPUT"
    else
        OUTPUT_DIR="$OUTPUT"
        mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="."
    fi
    SOURCE_BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
    OUTPUT="${OUTPUT_DIR}/corrected_method${METHOD}_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S).jpg"
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

# 检查Python脚本是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS_SCRIPT="$SCRIPT_DIR/all_skew_correction_methods.py"

if [[ ! -f "$METHODS_SCRIPT" ]]; then
    echo "错误: 找不到方法脚本: $METHODS_SCRIPT"
    echo "请确保 all_skew_correction_methods.py 在同一目录下"
    exit 1
fi

# 执行倾斜校正
echo "=========================================="
echo "自适应倾斜校正 - 方案${METHOD}: ${METHOD_NAME}"
echo "=========================================="
echo "输入图像: $INPUT"
echo "输出图像: $OUTPUT"
echo ""

# 切换到脚本目录以确保可以导入模块
cd "$SCRIPT_DIR"

# 运行Python代码
python3 << EOF
import sys
import cv2
from all_skew_correction_methods import (
    correct_skew_method1,
    correct_skew_method2,
    correct_skew_method3,
    correct_skew_method4,
    correct_skew_method5,
    correct_skew_method6,
)

# 读取图像
image = cv2.imread("$INPUT")
if image is None:
    print(f"错误: 无法读取图像: $INPUT")
    sys.exit(1)

print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
print("正在处理...")

# 根据方法编号选择方法
method_num = int("$METHOD")
methods = {
    1: correct_skew_method1,
    2: correct_skew_method2,
    3: correct_skew_method3,
    4: correct_skew_method4,
    5: correct_skew_method5,
    6: correct_skew_method6,
}

method_func = methods[method_num]
corrected, info = method_func(image)

# 获取检测角度
angle = info.get('detected_angle', 0.0)
if isinstance(angle, (int, float)):
    print(f"检测到的倾斜角度: {angle:.2f}度")
    print(f"应用的校正角度: {info.get('corrected_angle', -angle):.2f}度")
else:
    print(f"检测信息: {info}")

# 保存结果
success = cv2.imwrite("$OUTPUT", corrected)
if success:
    print(f"✅ 成功！结果已保存到: $OUTPUT")
    sys.exit(0)
else:
    print("❌ 保存失败")
    sys.exit(1)
EOF

# 检查执行结果
if [[ $? -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "=========================================="
    echo "✅ 处理完成！"
    echo "=========================================="
    echo "结果已保存到: $OUTPUT"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 处理失败！"
    echo "=========================================="
    echo "请检查:"
    echo "  1. Python环境是否正确安装"
    echo "  2. 是否安装了 opencv-python 和 numpy"
    echo "  3. 输入图像是否有效"
    exit 1
fi
