#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成论文实验图：4x5网格
- 4行：Input, Aligned, Partition, Heatmap
- 5列：Satellite-view + 4个Drone-view（不同高度和角度）
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
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
import cv2

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.visualize import align_image, draw_partition, generate_heatmap

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Generate Experiment Figure')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--satellite_img', type=str, required=True, help='卫星图像路径')
parser.add_argument('--drone_imgs', type=str, required=True, help='无人机图像路径列表，用逗号分隔，至少4张')
parser.add_argument('--epoch', type=int, default=119, help='使用的模型epoch')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')
parser.add_argument('--output', type=str, default='experiment_figure.png', help='输出图像路径')

opt = parser.parse_args()

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
    opt.mode = config.get('mode', 1)  # 1: drone->satellite, 2: satellite->drone
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
    opt.mode = 1

# 设置save_intermediate为True以保存中间参数
opt.save_intermediate = True

print('=' * 60)
print(f'模型名称: {opt.name}')
print(f'卫星图像: {opt.satellite_img}')
print(f'无人机图像: {opt.drone_imgs}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
print('=' * 60)

# 检查图像是否存在
if not os.path.exists(opt.satellite_img):
    raise FileNotFoundError(f"卫星图像路径不存在: {opt.satellite_img}")

drone_img_paths = [p.strip() for p in opt.drone_imgs.split(',')]
if len(drone_img_paths) < 4:
    raise ValueError(f"至少需要4张无人机图像，当前提供: {len(drone_img_paths)}")

for i, path in enumerate(drone_img_paths):
    if not os.path.exists(path):
        raise FileNotFoundError(f"无人机图像 {i+1} 路径不存在: {path}")

# 加载模型
checkpoint_path = os.path.join(checkpoint_dir, f'net_{opt.epoch:03d}.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

print(f"\n正在加载模型: {checkpoint_path}")
opt.checkpoint = checkpoint_path
model = load_network(opt, gpu_ids)
model = model.eval()
if use_gpu:
    model = model.cuda()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(model, img_path, opt, is_satellite=False):
    """
    处理单张图像，生成Input, Aligned, Partition, Heatmap四种可视化结果
    
    Args:
        model: 训练好的模型
        img_path: 图像路径
        opt: 配置参数
        is_satellite: 是否为卫星图像
    
    Returns:
        dict: 包含'input', 'aligned', 'partition', 'heatmap'的字典
    """
    # 加载图像
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    if use_gpu:
        img_tensor = img_tensor.cuda()
    
    # 原始图像（numpy格式）
    input_img = np.array(img)
    h_orig, w_orig = input_img.shape[:2]
    
    # 前向传播获取中间参数
    model.eval()
    with torch.no_grad():
        # 根据模式选择输入
        if opt.mode == 1:
            # drone->satellite：卫星图作为query_satellite
            if is_satellite:
                outputs, _ = model(img_tensor, None)
            else:
                _, outputs = model(None, img_tensor)
        else:
            # satellite->drone：无人机图作为query_drone
            if is_satellite:
                _, outputs = model(None, img_tensor)
            else:
                outputs, _ = model(img_tensor, None)
        
        # 获取中间输出
        if hasattr(model, 'module'):
            loc_model = model.module.loc_model
            intermediate_outputs = model.module.intermediate_outputs
        else:
            loc_model = model.loc_model
            intermediate_outputs = model.intermediate_outputs
        
        # 提取中间参数
        theta = intermediate_outputs.get('theta', None)
        boundaries = intermediate_outputs.get('boundaries', None)
        f_p_aligned = intermediate_outputs.get('f_p_aligned', None)
        
        # 生成Aligned图
        if theta is not None:
            aligned_img = align_image(img, theta[0], img_size=opt.h)  # 取第一个batch
            # 调整尺寸到原始图像大小
            aligned_img = Image.fromarray(aligned_img).resize((w_orig, h_orig), Image.LANCZOS)
            aligned_img = np.array(aligned_img)
        else:
            aligned_img = input_img.copy()
        
        # 生成Partition图（基于Aligned图）
        if boundaries is not None:
            partition_img = draw_partition(aligned_img, boundaries, img_size=opt.h)
            # 调整尺寸
            if partition_img.shape[:2] != (h_orig, w_orig):
                partition_img = Image.fromarray(partition_img).resize((w_orig, h_orig), Image.LANCZOS)
                partition_img = np.array(partition_img)
        else:
            partition_img = aligned_img.copy()
        
        # 生成Heatmap
        if f_p_aligned is not None:
            heatmap_img = generate_heatmap(f_p_aligned[0], original_img_size=(h_orig, w_orig))  # 取第一个batch
        else:
            # 如果无法获取特征图，使用灰度图
            heatmap_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
    
    return {
        'input': input_img,
        'aligned': aligned_img,
        'partition': partition_img,
        'heatmap': heatmap_img
    }

# 处理所有图像
print("\n正在处理图像...")

# 处理卫星图像
print("处理卫星图像...")
satellite_results = process_image(model, opt.satellite_img, opt, is_satellite=True)

# 处理无人机图像（取前4张）
drone_results = []
for i, drone_path in enumerate(drone_img_paths[:4]):
    print(f"处理无人机图像 {i+1}...")
    drone_result = process_image(model, drone_path, opt, is_satellite=False)
    drone_results.append(drone_result)

# 创建4x5网格图
print("\n正在生成对比图...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# 行标签
row_labels = ['Input', 'Aligned', 'Partition', 'Heatmap']
# 列标签
col_labels = ['Satellite-view', 'Drone-view 1', 'Drone-view 2', 'Drone-view 3', 'Drone-view 4']

# 卫星图像列（第0列）
satellite_data = [
    satellite_results['input'],
    satellite_results['aligned'],
    satellite_results['partition'],
    satellite_results['heatmap']
]
for row in range(4):
    axes[row, 0].imshow(satellite_data[row])
    axes[row, 0].axis('off')
    if row == 0:
        axes[row, 0].set_title(col_labels[0], fontsize=12, fontweight='bold', pad=10)

# 无人机图像列（第1-4列）
for col in range(1, 5):
    drone_data = [
        drone_results[col-1]['input'],
        drone_results[col-1]['aligned'],
        drone_results[col-1]['partition'],
        drone_results[col-1]['heatmap']
    ]
    for row in range(4):
        axes[row, col].imshow(drone_data[row])
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(col_labels[col], fontsize=12, fontweight='bold', pad=10)

# 添加行标签（左侧）
for row in range(4):
    fig.text(0.015, 0.75 - row * 0.25, row_labels[row], 
             fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center')

plt.tight_layout()
plt.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.02)

# 保存图像
save_path = opt.output
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print('=' * 60)
print(f'✓ 论文实验结果图已保存至: {save_path}')
print('=' * 60)
