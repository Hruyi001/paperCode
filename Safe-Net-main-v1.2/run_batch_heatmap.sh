#!/bin/bash

# 批量生成热力图脚本
# 对指定目录下的所有图像生成aligned_feature_map热力图

# ========== 用户配置区域 ==========
# 输入目录：包含所有要处理的图像
INPUT_DIR="./data/University-Release/test/query_drone"

# 输出目录：保存生成的热力图
OUTPUT_DIR="./safe-net-heatmap/query_drone"

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 模式设置
# 1 = drone->satellite (用于gallery_satellite)
# 2 = satellite->drone (用于query_drone)
MODE=2

# 是否使用query_transforms（query_drone需要设置为true，gallery_satellite设置为false）
USE_QUERY_TRANSFORM=true

# 可选：指定checkpoint目录（如果不在默认位置）
# CHECKPOINT_DIR="./checkpoints/${MODEL_NAME}"
# =================================

echo "=========================================="
echo "批量生成热力图"
echo "=========================================="
echo "输入目录: ${INPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型名称: ${MODEL_NAME}"
echo "Epoch: ${EPOCH}"
echo "模式: ${MODE} ($([ ${MODE} -eq 1 ] && echo 'drone->satellite' || echo 'satellite->drone'))"
echo "使用query_transform: ${USE_QUERY_TRANSFORM}"
echo "GPU IDs: ${GPU_IDS}"
echo "=========================================="

# 检查输入目录是否存在
if [ ! -d "${INPUT_DIR}" ]; then
    echo "错误: 输入目录不存在: ${INPUT_DIR}"
    exit 1
fi

# 构建命令
CMD="python generate_batch_heatmap.py \
    --name ${MODEL_NAME} \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mode ${MODE} \
    --epoch ${EPOCH} \
    --gpu_ids ${GPU_IDS}"

# 如果使用query_transform，添加参数
if [ "${USE_QUERY_TRANSFORM}" = "true" ]; then
    CMD="${CMD} --use_query_transform"
fi

# 如果指定了checkpoint目录，添加参数
if [ -n "${CHECKPOINT_DIR}" ]; then
    CMD="${CMD} --checkpoint_dir ${CHECKPOINT_DIR}"
fi

# 运行Python脚本
echo ""
echo "开始处理..."
${CMD}

echo ""
echo "完成！热力图已保存到: ${OUTPUT_DIR}"
