#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成不同epoch模型的热力图对比图
每隔30轮（epoch 29, 59, 89, 119）的模型热力图对比
"""

from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.heatmap_utils import generate_heatmap_data

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Generate Heatmap Comparison')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--img_path', type=str, required=True, help='输入图像路径，用于生成热力图')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--mode', default=1, type=int, help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--epochs', default='29,59,89,119', type=str, help='要对比的epoch列表，用逗号分隔')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')

opt = parser.parse_args()

# 解析epoch列表
epoch_list = [int(e.strip()) for e in opt.epochs.split(',')]
epoch_list.sort()

# 设置checkpoint目录
if opt.checkpoint_dir is None:
    checkpoint_dir = os.path.join('checkpoints', opt.name)
else:
    checkpoint_dir = opt.checkpoint_dir

# 设置GPU
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

if len(gpu_ids) > 0:
    cudnn.benchmark = True

use_gpu = torch.cuda.is_available()

# 加载配置文件
config_path = os.path.join(checkpoint_dir, 'opts.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    opt.views = config.get('views', 2)
    opt.block = config.get('block', 4)
    opt.share = config.get('share', True)
    opt.nclasses = config.get('nclasses', 729)
    if 'h' in config:
        opt.h = config['h']
    if 'w' in config:
        opt.w = config['w']
else:
    print(f"警告: 未找到配置文件 {config_path}，使用默认值")
    opt.views = 2
    opt.block = 4
    opt.share = True
    opt.nclasses = 729

print('=' * 60)
print(f'模型名称: {opt.name}')
print(f'图像路径: {opt.img_path}')
print(f'模式: {opt.mode} (1:drone->satellite, 2:satellite->drone)')
print(f'对比的epoch: {epoch_list}')
print(f'Checkpoint目录: {checkpoint_dir}')
print('=' * 60)

# 检查图像是否存在
if not os.path.exists(opt.img_path):
    raise FileNotFoundError(f"图像路径不存在: {opt.img_path}")

# 加载原始图像用于显示
original_img = Image.open(opt.img_path).convert("RGB")

# 存储每个epoch的热力图
heatmap_images = []
epoch_labels = []

# 遍历每个epoch
for epoch in epoch_list:
    # 模型文件名格式：net_%03d.pth (如 net_029.pth)
    checkpoint_path = os.path.join(checkpoint_dir, f'net_{epoch:03d}.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"警告: 未找到模型文件 {checkpoint_path}，跳过")
        continue
    
    print(f"\n正在处理 Epoch {epoch}...")
    print(f"加载模型: {checkpoint_path}")
    
    # 加载模型
    opt.checkpoint = checkpoint_path
    model = load_network(opt, gpu_ids)
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    
    # 生成热力图
    try:
        img_array, heatmap_array = generate_heatmap_data(model, opt.img_path, opt)
        heatmap_images.append(heatmap_array)
        epoch_labels.append(f'Epoch {epoch}')
        print(f"✓ Epoch {epoch} 热力图生成成功")
    except Exception as e:
        print(f"✗ Epoch {epoch} 热力图生成失败: {str(e)}")
        continue
    
    # 清理GPU内存
    del model
    torch.cuda.empty_cache()

if len(heatmap_images) == 0:
    print("错误: 没有成功生成任何热力图")
    sys.exit(1)

# 创建对比图
print(f"\n正在生成对比图...")
num_epochs = len(heatmap_images)
cols = num_epochs + 1  # 原图 + 各个epoch的热力图
rows = 1

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4))
if cols == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# 显示原图
axes[0].imshow(original_img)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# 显示各个epoch的热力图
for i, (heatmap_img, label) in enumerate(zip(heatmap_images, epoch_labels)):
    axes[i + 1].imshow(heatmap_img)
    axes[i + 1].set_title(label, fontsize=12, fontweight='bold')
    axes[i + 1].axis('off')

plt.tight_layout()

# 保存对比图
img_basename = os.path.splitext(os.path.basename(opt.img_path))[0]
save_path = f'heatmap_comparison_{img_basename}_epochs_{"_".join(map(str, epoch_list))}.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print('=' * 60)
print(f'✓ 热力图对比图已保存至: {save_path}')
print(f'  包含 {len(heatmap_images)} 个epoch的热力图对比')
print('=' * 60)
