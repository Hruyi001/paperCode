#!/bin/bash

# 热力图对比生成脚本
# 用于生成每隔30轮的模型热力图对比

name="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
gpu_ids=0
mode=1  # 1:drone->satellite  2:satellite->drone
h=256
w=256

# 设置要对比的epoch（每30轮）
epochs="29,59,89,119"

# 设置测试图像路径（请修改为实际的图像路径）
img_path="/datasets/University-Release/test/query_drone/0000/image-40.jpeg"

# 检查图像是否存在
if [ ! -f "$img_path" ]; then
    echo "错误: 图像文件不存在: $img_path"
    echo "请修改脚本中的 img_path 变量为实际的图像路径"
    exit 1
fi

echo "开始生成热力图对比..."
echo "模型名称: $name"
echo "图像路径: $img_path"
echo "模式: $mode"
echo "Epoch列表: $epochs"
echo ""

python generate_heatmap_comparison.py \
    --name $name \
    --img_path "$img_path" \
    --gpu_ids $gpu_ids \
    --mode $mode \
    --h $h \
    --w $w \
    --epochs "$epochs"

echo ""
echo "完成！"
