#!/bin/bash

# 自适应倾斜校正脚本 - 使用所有6种方法
# 用法: ./correct_skew_all_methods.sh [输入图像] [输出目录]
# 或者直接在脚本中设置路径（见下方配置区域）

# ============================================
# 配置区域：在这里直接设置参数
# ============================================
INPUT_IMAGE=""      # 输入图像路径
OUTPUT_DIR=""       # 输出目录路径（可选）
                    #   - 留空：自动生成目录名
                    #   - 目录路径：保存到指定目录（如 "output" 或 "./results"）
# ============================================

# 如果命令行提供了参数，则使用命令行参数
if [[ $# -ge 1 ]]; then
    INPUT="$1"
    OUTPUT="${2:-}"
elif [[ -n "$INPUT_IMAGE" ]]; then
    # 使用脚本中设置的路径
    INPUT="$INPUT_IMAGE"
    OUTPUT="$OUTPUT_DIR"
else
    echo "用法: $0 [输入图像] [输出目录]"
    echo ""
    echo "方式1: 命令行参数"
    echo "  $0 image.jpg"
    echo "  $0 image.jpg output_folder"
    echo ""
    echo "方式2: 在脚本中设置路径（编辑脚本顶部的配置区域）"
    echo "  设置 INPUT_IMAGE 变量"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/image.jpg"
    echo "  $0 /path/to/image.jpg /path/to/output"
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

# 处理输出目录
if [[ -z "$OUTPUT" ]]; then
    # 如果没有指定输出目录，自动生成目录名
    SOURCE_BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
    OUTPUT="skew_correction_results_${SOURCE_BASENAME}_$(date +%Y%m%d_%H%M%S)"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT" 2>/dev/null || {
    echo "错误: 无法创建输出目录: $OUTPUT"
    exit 1
}

# 检查Python脚本是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_all_methods.py"

if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "错误: 找不到测试脚本: $TEST_SCRIPT"
    echo "请确保 test_all_methods.py 在同一目录下"
    exit 1
fi

# 执行倾斜校正
echo "=========================================="
echo "自适应倾斜校正 - 使用所有6种方法"
echo "=========================================="
echo "输入图像: $INPUT"
echo "输出目录: $OUTPUT"
echo ""

# 切换到脚本目录以确保可以导入模块
cd "$SCRIPT_DIR"

# 运行Python脚本
python3 "$TEST_SCRIPT" "$INPUT" "$OUTPUT"

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "✅ 处理完成！"
    echo "=========================================="
    echo "所有结果已保存到: $OUTPUT"
    echo ""
    echo "生成的文件:"
    echo "  - 00_original.jpg (原图)"
    echo "  - 1_method1_*.jpg (方案1: 投影轮廓法)"
    echo "  - 2_method2_*.jpg (方案2: 霍夫直线检测法)"
    echo "  - 3_method3_*.jpg (方案3: 最小外接矩形法)"
    echo "  - 4_method4_*.jpg (方案4: 投影变换+旋转组合法)"
    echo "  - 5_method5_*.jpg (方案5: 频域分析法)"
    echo "  - 6_method6_*.jpg (方案6: 组合方法)"
    echo "  - comparison_all_methods.jpg (对比图)"
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
