#!/bin/bash
# 生成消融实验热力图的便捷脚本

# ============================================
# 配置区域 - 请在这里修改图像路径和默认参数
# ============================================

# 图像路径（请修改为您的实际路径）
drone_img="/datasets/University-Release/test/query_drone/1483/image-51.jpeg"
satellite_img="/datasets/University-Release/test/gallery_satellite/1483/1483.jpg"

# 默认参数
name="FCFE_Model_University"
gpu_ids="0"
output_dir="./heatmap_results"

# ============================================
# 以下部分一般不需要修改
# ============================================

# 解析命令行参数（如果提供了，会覆盖上面的默认值）
while [[ $# -gt 0 ]]; do
    case $1 in
        --drone_img)
            drone_img="$2"
            shift 2
            ;;
        --satellite_img)
            satellite_img="$2"
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
        -h|--help)
            echo "使用方法:"
            echo "  $0 [options]"
            echo ""
            echo "默认图像路径（可在脚本中修改）:"
            echo "  Drone: $drone_img"
            echo "  Satellite: $satellite_img"
            echo ""
            echo "可选参数:"
            echo "  --drone_img PATH         : 无人机图像路径"
            echo "  --satellite_img PATH     : 卫星图像路径"
            echo "  --name MODEL_NAME        : 模型名称 (默认: $name)"
            echo "  --gpu_ids GPU_ID        : GPU ID (默认: $gpu_ids)"
            echo "  --output_dir DIR        : 输出目录 (默认: $output_dir)"
            echo ""
            echo "示例:"
            echo "  $0  # 使用脚本中配置的默认路径"
            echo "  $0 --drone_img /path/to/drone.jpg --satellite_img /path/to/satellite.jpg"
            echo "  $0 --name MyModel --gpu_ids 0"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
cmd="python generate_heatmap_ablation.py \
    --drone_img \"$drone_img\" \
    --satellite_img \"$satellite_img\" \
    --name \"$name\" \
    --gpu_ids \"$gpu_ids\" \
    --output_dir \"$output_dir\""

# 显示配置信息
echo "=========================================="
echo "热力图生成配置"
echo "=========================================="
echo "Drone图像:    $drone_img"
echo "Satellite图像: $satellite_img"
echo "模型名称:     $name"
echo "GPU ID:       $gpu_ids"
echo "输出目录:     $output_dir"
echo "=========================================="
echo ""

# 检查图像文件是否存在
if [ ! -f "$drone_img" ]; then
    echo "错误: 无人机图像不存在: $drone_img"
    echo "请检查路径或使用 --drone_img 参数指定正确的路径"
    exit 1
fi

if [ ! -f "$satellite_img" ]; then
    echo "错误: 卫星图像不存在: $satellite_img"
    echo "请检查路径或使用 --satellite_img 参数指定正确的路径"
    exit 1
fi

# 执行命令
echo "执行命令:"
echo "$cmd"
echo ""
eval $cmd
