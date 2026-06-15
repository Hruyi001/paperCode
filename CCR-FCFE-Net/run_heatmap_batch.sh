#!/bin/bash
# 批量生成消融实验热力图的便捷脚本

# ============================================
# 配置区域 - 请在这里修改路径和默认参数
# ============================================

# 数据集路径（根据实际数据集结构配置）
# 数据集结构:
#   query_drone/{location_id}/image-xx.jpeg
#   gallery_satellite/{location_id}/{location_id}.jpg
drone_dir="/root/dataset/University-Release/test/query_drone"
satellite_dir="/root/dataset/University-Release/test/gallery_satellite"

# 默认参数
name="FCFE_Model_University"
gpu_ids="0"
output_dir="./heatmap_results_batch2"
max_images=""  # 限制处理图像数量（用于测试），留空表示处理所有

# ============================================
# 以下部分一般不需要修改
# ============================================

# 解析命令行参数（如果提供了，会覆盖上面的默认值）
while [[ $# -gt 0 ]]; do
    case $1 in
        --drone_dir)
            drone_dir="$2"
            shift 2
            ;;
        --satellite_dir)
            satellite_dir="$2"
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
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --max_images)
            max_images="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法:"
            echo "  $0 [options]"
            echo ""
            echo "默认配置（可在脚本中修改）:"
            echo "  Drone目录:    $drone_dir"
            echo "  Satellite目录: $satellite_dir"
            echo "  模型名称:     $name"
            echo "  GPU ID:       $gpu_ids"
            echo "  输出目录:     $output_dir"
            echo ""
            echo "可选参数:"
            echo "  --drone_dir PATH         : drone图像目录"
            echo "  --satellite_dir PATH     : satellite图像目录"
            echo "  --name MODEL_NAME        : 模型名称 (默认: $name)"
            echo "  --gpu_ids GPU_ID        : GPU ID (默认: $gpu_ids)"
            echo "  --output_dir DIR        : 输出目录 (默认: $output_dir)"
            echo "  --max_images N          : 最大处理图像数量（用于测试）"
            echo ""
            echo "示例:"
            echo "  $0  # 使用脚本中配置的默认路径"
            echo "  $0 --drone_dir /path/to/drone --satellite_dir /path/to/satellite"
            echo "  $0 --name MyModel --max_images 10  # 只处理前10张图像"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示配置信息
echo "=========================================="
echo "批量热力图生成配置"
echo "=========================================="
echo "Drone目录:    $drone_dir"
echo "Satellite目录: $satellite_dir"
echo "模型名称:     $name"
echo "GPU ID:       $gpu_ids"
echo "输出目录:     $output_dir"
if [ -n "$max_images" ]; then
    echo "最大图像数:   $max_images"
fi
echo "=========================================="
echo ""

# 检查目录是否存在
if [ ! -d "$drone_dir" ]; then
    echo "错误: Drone目录不存在: $drone_dir"
    exit 1
fi

if [ ! -d "$satellite_dir" ]; then
    echo "错误: Satellite目录不存在: $satellite_dir"
    exit 1
fi

# 构建命令
cmd="python generate_heatmap_batch.py \
    --drone_dir \"$drone_dir\" \
    --satellite_dir \"$satellite_dir\" \
    --name \"$name\" \
    --gpu_ids \"$gpu_ids\" \
    --output_dir \"$output_dir\""

# 如果限制了图像数量，添加参数
if [ -n "$max_images" ]; then
    cmd="$cmd --max_images $max_images"
fi

# 执行命令
echo "执行命令:"
echo "$cmd"
echo ""
eval $cmd
