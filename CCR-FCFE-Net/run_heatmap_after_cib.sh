#!/bin/bash
# 批量生成After CIB热力图叠加的便捷脚本

# ============================================
# 配置区域 - 请在这里修改默认参数
# ============================================

# 输入和输出路径（请修改为您的实际路径）
input_dir="/root/exp/exp3-1/0000-4"
output_dir="/root/exp/exp3-1/b1/ccr"

# 其他默认参数
name="FCFE_Model_University"
gpu_ids="0"
view_type="auto"
alpha=0.5

# ============================================
# 以下部分一般不需要修改
# ============================================

# 解析命令行参数（如果提供了，会覆盖上面的默认值）
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            input_dir="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --name)
            name="$2"
            shift 2
            ;;
        --gpu_ids)
            gpu_ids="$2"
            shift 2
            ;;
        --view_type)
            view_type="$2"
            shift 2
            ;;
        --alpha)
            alpha="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法:"
            echo "  $0 [options]"
            echo ""
            echo "说明:"
            echo "  输入和输出路径可以在脚本的配置区域设置，也可以通过命令行参数指定"
            echo "  如果脚本中已配置路径，可以直接运行: $0"
            echo ""
            echo "当前配置（可在脚本中修改）:"
            echo "  输入目录: $input_dir"
            echo "  输出目录: $output_dir"
            echo ""
            echo "可选参数（会覆盖脚本中的默认值）:"
            echo "  --input_dir DIR          : 输入图像目录"
            echo "  --output_dir DIR         : 输出图像目录"
            echo "  --name MODEL_NAME        : 模型名称 (默认: $name)"
            echo "  --gpu_ids GPU_ID         : GPU ID (默认: $gpu_ids)"
            echo "  --view_type TYPE         : 图像类型: drone, satellite, 或 auto (默认: $view_type)"
            echo "                             auto: 根据路径自动判断"
            echo "  --alpha VALUE            : 热力图透明度 0-1 (默认: $alpha)"
            echo ""
            echo "示例:"
            echo "  $0  # 使用脚本中配置的路径"
            echo "  $0 --input_dir /path/to/input --output_dir /path/to/output"
            echo "  $0 --view_type drone --alpha 0.6"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数（如果脚本中已配置，则不需要命令行参数）
if [ -z "$input_dir" ]; then
    echo "错误: 必须指定输入目录"
    echo "方法1: 在脚本的配置区域设置 input_dir"
    echo "方法2: 使用 --input_dir 参数指定"
    echo "使用 $0 --help 查看帮助信息"
    exit 1
fi

if [ -z "$output_dir" ]; then
    echo "错误: 必须指定输出目录"
    echo "方法1: 在脚本的配置区域设置 output_dir"
    echo "方法2: 使用 --output_dir 参数指定"
    echo "使用 $0 --help 查看帮助信息"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误: 输入目录不存在: $input_dir"
    exit 1
fi

# 确保输出目录是绝对路径
# 如果输出目录是相对路径，转换为绝对路径
if [[ "$output_dir" != /* ]]; then
    # 相对路径，转换为绝对路径
    output_dir="$(cd "$(dirname "$output_dir")" && pwd)/$(basename "$output_dir")"
fi

# 构建命令
cmd="python generate_heatmap_after_cib.py \
    --input_dir \"$input_dir\" \
    --output_dir \"$output_dir\" \
    --name \"$name\" \
    --gpu_ids \"$gpu_ids\" \
    --view_type \"$view_type\" \
    --alpha $alpha"

# 显示配置信息
echo "=========================================="
echo "After CIB热力图生成配置"
echo "=========================================="
echo "输入目录:     $input_dir"
echo "输出目录:     $output_dir"
echo "模型名称:     $name"
echo "GPU ID:       $gpu_ids"
echo "图像类型:     $view_type"
echo "透明度:       $alpha"
echo "=========================================="
echo ""

# 执行命令
echo "执行命令:"
echo "$cmd"
echo ""
eval $cmd
