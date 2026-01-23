#!/bin/bash

# Safe-Net 批量可视化实验图生成脚本
# 用法: ./batch_visualize.sh [起始类别] [结束类别] [输出目录]
# 示例: ./batch_visualize.sh 0070 0100 output_figures

# ============================================
# 配置区域
# ============================================

# 数据集路径
DATASET_ROOT="/datasets/University-Release/test"
SATELLITE_DIR="${DATASET_ROOT}/query_satellite"
DRONE_DIR="${DATASET_ROOT}/query_drone"

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 输出目录（如果未指定，使用默认值）
OUTPUT_DIR="${3:-experiment_figures}"

# 起始和结束类别ID（如果未指定，处理所有类别）
START_CLASS="${1:-}"
END_CLASS="${2:-}"

# 每个类别选择的无人机图片数量
NUM_DRONE_IMGS=4

# ============================================
# 函数定义
# ============================================

# 获取所有类别ID列表
get_all_classes() {
    if [ -d "$SATELLITE_DIR" ]; then
        ls -1 "$SATELLITE_DIR" | sort
    else
        echo "错误: 卫星图目录不存在: $SATELLITE_DIR"
        exit 1
    fi
}

# 检查类别是否有效
check_class() {
    local class_id=$1
    local satellite_path="${SATELLITE_DIR}/${class_id}/${class_id}.jpg"
    local drone_dir="${DRONE_DIR}/${class_id}"
    
    # 检查卫星图是否存在
    if [ ! -f "$satellite_path" ]; then
        return 1
    fi
    
    # 检查无人机图目录是否存在
    if [ ! -d "$drone_dir" ]; then
        return 1
    fi
    
    # 检查是否有足够的无人机图片
    local drone_count=$(ls -1 "$drone_dir"/*.{jpg,jpeg,png} 2>/dev/null | wc -l)
    if [ "$drone_count" -lt "$NUM_DRONE_IMGS" ]; then
        return 1
    fi
    
    return 0
}

# 获取无人机图片列表（选择前N张）
get_drone_images() {
    local class_id=$1
    local drone_dir="${DRONE_DIR}/${class_id}"
    
    # 获取所有图片文件，按名称排序，取前N张，用逗号连接
    local images=$(ls -1 "$drone_dir"/*.{jpg,jpeg,png} 2>/dev/null | sort | head -n "$NUM_DRONE_IMGS" | tr '\n' ',' | sed 's/,$//')
    echo "$images"
}

# 处理单个类别
process_class() {
    local class_id=$1
    
    echo ""
    echo "=========================================="
    echo "处理类别: $class_id"
    echo "=========================================="
    
    # 检查类别是否有效
    if ! check_class "$class_id"; then
        echo "⚠ 跳过类别 $class_id (缺少必要的图片文件)"
        return 1
    fi
    
    # 构建路径
    local satellite_img="${SATELLITE_DIR}/${class_id}/${class_id}.jpg"
    local drone_imgs=$(get_drone_images "$class_id")
    
    # 检查是否成功获取到足够的图片
    local img_count=$(echo "$drone_imgs" | tr ',' '\n' | wc -l)
    if [ "$img_count" -lt "$NUM_DRONE_IMGS" ]; then
        echo "⚠ 跳过类别 $class_id (只有 $img_count 张无人机图片，需要 $NUM_DRONE_IMGS 张)"
        return 1
    fi
    
    # 输出文件路径
    local output_file="${OUTPUT_DIR}/experiment_figure_${class_id}.png"
    
    echo "卫星图: $satellite_img"
    echo "无人机图: $drone_imgs"
    echo "输出文件: $output_file"
    
    # 调用Python脚本生成实验图
    python visualize_demo.py \
        --name "$MODEL_NAME" \
        --satellite_img "$satellite_img" \
        --drone_imgs "$drone_imgs" \
        --epoch "$EPOCH" \
        --gpu_ids "$GPU_IDS" \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功生成: $output_file"
        return 0
    else
        echo "✗ 生成失败: $class_id"
        return 1
    fi
}

# ============================================
# 主程序
# ============================================

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取所有类别
ALL_CLASSES=$(get_all_classes)

# 确定要处理的类别范围
if [ -z "$START_CLASS" ] && [ -z "$END_CLASS" ]; then
    # 处理所有类别
    CLASSES_TO_PROCESS="$ALL_CLASSES"
    echo "=========================================="
    echo "批量生成实验图 - 处理所有类别"
    echo "=========================================="
elif [ -n "$START_CLASS" ] && [ -n "$END_CLASS" ]; then
    # 处理指定范围的类别
    CLASSES_TO_PROCESS=$(echo "$ALL_CLASSES" | awk -v start="$START_CLASS" -v end="$END_CLASS" '
        $1 >= start && $1 <= end { print $1 }
    ')
    echo "=========================================="
    echo "批量生成实验图 - 处理类别范围: $START_CLASS 到 $END_CLASS"
    echo "=========================================="
elif [ -n "$START_CLASS" ]; then
    # 从指定类别开始处理所有
    CLASSES_TO_PROCESS=$(echo "$ALL_CLASSES" | awk -v start="$START_CLASS" '$1 >= start')
    echo "=========================================="
    echo "批量生成实验图 - 从类别 $START_CLASS 开始处理"
    echo "=========================================="
else
    echo "错误: 参数格式不正确"
    echo "用法: $0 [起始类别] [结束类别] [输出目录]"
    exit 1
fi

# 统计信息
TOTAL_CLASSES=$(echo "$CLASSES_TO_PROCESS" | wc -l)
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "总类别数: $TOTAL_CLASSES"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 处理每个类别
CURRENT=0
while IFS= read -r class_id; do
    [ -z "$class_id" ] && continue
    
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_CLASSES] 处理类别: $class_id"
    
    if process_class "$class_id"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        if check_class "$class_id"; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            SKIP_COUNT=$((SKIP_COUNT + 1))
        fi
    fi
done <<< "$CLASSES_TO_PROCESS"

# 输出统计信息
echo ""
echo "=========================================="
echo "批量处理完成"
echo "=========================================="
echo "总类别数: $TOTAL_CLASSES"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo "跳过: $SKIP_COUNT"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
