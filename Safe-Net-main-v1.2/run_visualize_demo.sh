#!/bin/bash

# Safe-Net 可视化实验图生成脚本
# 用法: ./run_visualize_demo.sh
# 注意: 请修改下面的图片路径配置

# ============================================
# 配置区域 - 请根据实际情况修改图片路径
# ============================================

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"
OUTPUT="experiment_figure.png"

# 图片路径配置（请修改为你的实际图片路径）
# 卫星图像路径
SATELLITE_IMG="/datasets/University-Release/test/query_satellite/0070/0070.jpg"

# 无人机图像路径（用逗号分隔，至少4张）
DRONE_IMGS="/datasets/University-Release/test/query_drone/0070/image-01.jpeg,/datasets/University-Release/test/query_drone/0070/image-02.jpeg,/datasets/University-Release/test/query_drone/0070/image-03.jpeg,/datasets/University-Release/test/query_drone/0070/image-04.jpeg"

# ============================================
# 以下为脚本执行部分，一般不需要修改
# ============================================

# 检查图像文件是否存在
echo "检查图像文件..."
if [ ! -f "$SATELLITE_IMG" ]; then
    echo "错误: 卫星图像不存在: $SATELLITE_IMG"
    echo "请修改脚本中的 SATELLITE_IMG 变量为正确的路径"
    exit 1
fi

# 检查无人机图像
IFS=',' read -ra DRONE_ARRAY <<< "$DRONE_IMGS"
if [ ${#DRONE_ARRAY[@]} -lt 4 ]; then
    echo "错误: 至少需要4张无人机图像，当前提供: ${#DRONE_ARRAY[@]}"
    echo "请修改脚本中的 DRONE_IMGS 变量，确保包含至少4张图片路径（用逗号分隔）"
    exit 1
fi

for i in "${DRONE_ARRAY[@]}"; do
    if [ ! -f "$i" ]; then
        echo "错误: 无人机图像不存在: $i"
        echo "请修改脚本中的 DRONE_IMGS 变量为正确的路径"
        exit 1
    fi
done
echo "✓ 所有图像文件检查通过"

echo "=========================================="
echo "Safe-Net 可视化实验图生成"
echo "=========================================="
echo "模型名称: $MODEL_NAME"
echo "卫星图像: $SATELLITE_IMG"
echo "无人机图像: $DRONE_IMGS"
echo "Epoch: $EPOCH"
echo "输出文件: $OUTPUT"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="
echo ""

# 运行Python脚本
python visualize_demo.py \
    --name $MODEL_NAME \
    --satellite_img "$SATELLITE_IMG" \
    --drone_imgs "$DRONE_IMGS" \
    --epoch $EPOCH \
    --gpu_ids $GPU_IDS \
    --output "$OUTPUT"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 成功生成实验图: $OUTPUT"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ 生成失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi
