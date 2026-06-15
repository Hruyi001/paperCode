Q#!/bin/bash

# 对比热力图可视化脚本
# 用于生成 part_features 和 pfeat_align 的对比可视化

# ============================================
# 配置参数 - 请根据实际情况修改
# ============================================

# ============================================
# 数据集配置
# ============================================

# 数据集根目录（data_folder）
# 这是包含所有数据集的根目录
# 如果数据集路径直接包含train/test目录，则设置为完整路径
# 例如: /root/dataset/University-Release
DATA_FOLDER="/root/dataset/University-Release"

# 数据集名称（dataset_name）
# 如果DATA_FOLDER已经是完整的数据集路径（包含train/test），设置为空字符串 ""
# 如果DATA_FOLDER是数据集根目录，需要指定数据集文件夹名称
# University-1652标准结构: "U1652" (如果路径是 sample4geo/dataset/U1652)
# SUES-200: "SUES-200"
# 自定义数据集: 你的数据集文件夹名称
# 如果数据集路径直接是完整路径（如 /root/dataset/University-Release），设置为空字符串
DATASET_NAME=""

# 数据集类型（用于可视化脚本）
# 可选: U1652, SUES-200, custom
# 这个参数用于告诉可视化脚本如何解析数据集结构
DATASET_TYPE="U1652"

# 数据集任务类型（dataset）
# University-1652: 
#   - "U1652-D2S" (Drone to Satellite): 查询是无人机图像，图库是卫星图像
#   - "U1652-S2D" (Satellite to Drone): 查询是卫星图像，图库是无人机图像
# SUES-200: 同样支持 "U1652-D2S" 或 "U1652-S2D"
DATASET="U1652-D2S"

# SUES-200 数据集高度（altitude）
# 仅当使用 SUES-200 数据集时有效
# 可选: 150, 200, 250, 300
# 如果使用 University-1652，此参数会被忽略
ALTITUDE=300

# ============================================
# 数据集配置示例
# ============================================
# 
# 示例1: 完整路径数据集（如 /root/dataset/University-Release）
# DATA_FOLDER="/root/dataset/University-Release"
# DATASET_NAME=""  # 设置为空字符串
# DATASET_TYPE="U1652"
# DATASET="U1652-D2S"
#
# 示例2: University-1652标准结构, Drone to Satellite
# DATA_FOLDER="sample4geo/dataset"
# DATASET_NAME="U1652"
# DATASET_TYPE="U1652"
# DATASET="U1652-D2S"
#
# 示例3: University-1652, Satellite to Drone
# DATA_FOLDER="sample4geo/dataset"
# DATASET_NAME="U1652"
# DATASET_TYPE="U1652"
# DATASET="U1652-S2D"
#
# 示例4: SUES-200, 300米高度, Drone to Satellite
# DATA_FOLDER="sample4geo/dataset"
# DATASET_NAME="SUES-200"
# DATASET_TYPE="SUES-200"
# DATASET="U1652-D2S"
# ALTITUDE=300
#
# 示例5: 自定义数据集
# DATA_FOLDER="sample4geo/dataset"
# DATASET_NAME="my_custom_dataset"
# DATASET_TYPE="custom"
# DATASET="U1652-D2S"  # 这个参数对custom类型影响较小
# ============================================

# 自动构建数据集路径
# 如果DATASET_NAME为空，说明DATA_FOLDER已经是完整的数据集路径
if [ -z "$DATASET_NAME" ] || [ "$DATASET_NAME" = "" ]; then
    # DATA_FOLDER已经是完整的数据集路径（如 /root/dataset/University-Release）
    DATASET_PATH="$DATA_FOLDER"
    if [ "$DATASET" = "U1652-D2S" ]; then
        # Drone to Satellite: query是drone, gallery是satellite
        QUERY_FOLDER="$DATA_FOLDER/test/query_drone"
        GALLERY_FOLDER="$DATA_FOLDER/test/gallery_satellite"
    elif [ "$DATASET" = "U1652-S2D" ]; then
        # Satellite to Drone: query是satellite, gallery是drone
        QUERY_FOLDER="$DATA_FOLDER/test/query_satellite"
        GALLERY_FOLDER="$DATA_FOLDER/test/gallery_drone"
    fi
# University-1652 数据集路径结构（标准结构：sample4geo/dataset/U1652）
elif [ "$DATASET_NAME" = "U1652" ]; then
    if [ "$DATASET" = "U1652-D2S" ]; then
        # Drone to Satellite: query是drone, gallery是satellite
        DATASET_PATH="$DATA_FOLDER/$DATASET_NAME"
        QUERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/test/query_drone"
        GALLERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/test/gallery_satellite"
    elif [ "$DATASET" = "U1652-S2D" ]; then
        # Satellite to Drone: query是satellite, gallery是drone
        DATASET_PATH="$DATA_FOLDER/$DATASET_NAME"
        QUERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/test/query_satellite"
        GALLERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/test/gallery_drone"
    fi
# SUES-200 数据集路径结构
elif [ "$DATASET_NAME" = "SUES-200" ]; then
    if [ "$DATASET" = "U1652-D2S" ]; then
        # Drone to Satellite
        DATASET_PATH="$DATA_FOLDER/$DATASET_NAME"
        QUERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/Testing/$ALTITUDE/query_drone"
        GALLERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/Testing/$ALTITUDE/gallery_satellite"
    elif [ "$DATASET" = "U1652-S2D" ]; then
        # Satellite to Drone
        DATASET_PATH="$DATA_FOLDER/$DATASET_NAME"
        QUERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/Testing/$ALTITUDE/query_satellite"
        GALLERY_FOLDER="$DATA_FOLDER/$DATASET_NAME/Testing/$ALTITUDE/gallery_drone"
    fi
else
    # 自定义数据集路径
    DATASET_PATH="$DATA_FOLDER/$DATASET_NAME"
    QUERY_FOLDER="$DATASET_PATH/query"
    GALLERY_FOLDER="$DATASET_PATH/gallery"
fi

# ============================================
# 模型配置
# ============================================

# 模型权重路径（可选，如果不提供则使用随机初始化的模型）
# 推荐使用 weights_e2_0.9355.pth（性能最好的权重）
CHECKPOINT_PATH="checkpoint/D-S-BEST 1012201824/weights_e2_0.9355.pth"

# 模型参数
NUM_CLASSES=701
BLOCK=2

# ============================================
# 可视化配置
# ============================================

# 输出目录
OUTPUT_DIR="./heatmap_visualization_results"

# 每个类别采样的图像数量
NUM_SAMPLES=5

# 图像尺寸
IMG_SIZE=384

# GPU ID
GPU_ID=0

# ============================================
# 执行脚本
# ============================================

echo "============================================"
echo "对比热力图可视化"
echo "============================================"
echo ""
echo "数据集配置:"
echo "  - 数据集根目录: $DATA_FOLDER"
if [ -n "$DATASET_NAME" ] && [ "$DATASET_NAME" != "" ]; then
    echo "  - 数据集名称: $DATASET_NAME"
else
    echo "  - 数据集名称: (使用完整路径)"
fi
echo "  - 数据集类型: $DATASET_TYPE"
echo "  - 任务类型: $DATASET"
if [ "$DATASET_NAME" = "SUES-200" ]; then
    echo "  - 高度: $ALTITUDE 米"
fi
echo "  - 数据集路径: $DATASET_PATH"
echo "  - 查询图像路径: $QUERY_FOLDER"
echo "  - 图库图像路径: $GALLERY_FOLDER"
echo ""
echo "模型配置:"
echo "  - 模型权重: ${CHECKPOINT_PATH:-未指定（使用随机初始化）}"
echo "  - 类别数量: $NUM_CLASSES"
echo "  - Block数量: $BLOCK"
echo ""
echo "可视化配置:"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 采样数量: $NUM_SAMPLES"
echo "  - 图像尺寸: $IMG_SIZE"
echo "  - GPU ID: $GPU_ID"
echo "============================================"

# 检查数据集路径是否存在
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATASET_PATH"
    echo "请检查并修改脚本中的数据集配置参数:"
    echo "  - DATA_FOLDER: $DATA_FOLDER"
    if [ -n "$DATASET_NAME" ] && [ "$DATASET_NAME" != "" ]; then
        echo "  - DATASET_NAME: $DATASET_NAME"
    else
        echo "  - DATASET_NAME: (空，使用完整路径)"
    fi
    exit 1
fi

# 检查查询和图库路径是否存在（如果使用标准数据集结构）
if [ "$DATASET_TYPE" != "custom" ]; then
    if [ ! -d "$QUERY_FOLDER" ]; then
        echo "警告: 查询图像路径不存在: $QUERY_FOLDER"
        echo "将尝试使用数据集根路径: $DATASET_PATH"
    fi
    if [ ! -d "$GALLERY_FOLDER" ]; then
        echo "警告: 图库图像路径不存在: $GALLERY_FOLDER"
        echo "将尝试使用数据集根路径: $DATASET_PATH"
    fi
fi

# 检查模型权重是否存在（如果指定了）
if [ -n "$CHECKPOINT_PATH" ] && [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "警告: 模型权重文件不存在: $CHECKPOINT_PATH"
    echo "将使用随机初始化的模型"
    CHECKPOINT_PATH=""
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行可视化脚本
python visualize_comparison_heatmap.py \
    --dataset_path "$DATASET_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --dataset_type "$DATASET_TYPE" \
    --img_size $IMG_SIZE \
    --num_classes $NUM_CLASSES \
    --block $BLOCK \
    --gpu_id $GPU_ID

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "可视化完成！"
    echo "结果保存在: $OUTPUT_DIR"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "执行失败，请检查错误信息"
    echo "============================================"
    exit 1
fi
