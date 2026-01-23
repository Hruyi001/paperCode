#!/bin/bash

# 生成论文实验结果图的示例脚本
# 使用方法: bash run_generate_paper_figure.sh

python3 generate_paper_figure.py \
    --name SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \
    --satellite_img /path/to/satellite/image.jpg \
    --drone_imgs /path/to/drone1.jpg,/path/to/drone2.jpg,/path/to/drone3.jpg \
    --epoch 119 \
    --gpu_ids 0
