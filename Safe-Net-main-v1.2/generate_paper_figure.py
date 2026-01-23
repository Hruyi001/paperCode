#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成论文实验结果图
显示卫星视图和无人机视图在不同处理阶段（Input, Aligned, Partition, Heatmap）的对比
"""

from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.heatmap_utils import generate_heatmap_data
from utils.visualize import align_image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Generate Paper Figure')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--satellite_img', type=str, required=True, help='卫星图像路径')
parser.add_argument('--drone_imgs', type=str, required=True, help='无人机图像路径列表，用逗号分隔，至少3张')
parser.add_argument('--epoch', type=int, default=119, help='使用的模型epoch')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')

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
print(f'卫星图像: {opt.satellite_img}')
print(f'无人机图像: {opt.drone_imgs}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
print('=' * 60)

# 检查图像是否存在
if not os.path.exists(opt.satellite_img):
    raise FileNotFoundError(f"卫星图像路径不存在: {opt.satellite_img}")

drone_img_paths = [p.strip() for p in opt.drone_imgs.split(',')]
if len(drone_img_paths) < 3:
    raise ValueError(f"至少需要3张无人机图像，当前提供: {len(drone_img_paths)}")

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

def load_and_preprocess_image(img_path):
    """加载并预处理图像"""
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    if use_gpu:
        img_tensor = img_tensor.cuda()
    return img, img_tensor

def get_aligned_image(model, img_tensor, original_img, img_size=256):
    """获取对齐后的图像 - 直接对原始图像应用affine变换
    
    注意：这里直接对原始图像应用affine变换，返回对齐后的图像，
    而不是可视化特征图并叠加原图。这样更符合论文中Aligned图的定义。
    """
    model.eval()
    with torch.no_grad():
        # 获取模型内部结构
        if hasattr(model, 'module'):
            loc_model = model.module.loc_model
        else:
            loc_model = model.loc_model
        
        # 提取特征
        features, all_features = loc_model.transformer(img_tensor)
        global_feature = features[:, 0]
        
        # 获取affine变换参数theta
        A_theta = loc_model.loc_net(global_feature)
        
        # 使用utils.visualize中的align_image函数直接对原图应用affine变换
        # 这是正确的方法：直接保存对齐后的图像，而不是可视化特征图
        aligned_img = align_image(original_img, A_theta[0].cpu(), img_size=img_size)
        
        return aligned_img

def get_partition_image(model, img_tensor, original_img):
    """获取带分区框的图像"""
    model.eval()
    with torch.no_grad():
        # 获取模型内部结构
        if hasattr(model, 'module'):
            loc_model = model.module.loc_model
        else:
            loc_model = model.loc_model
        
        # 提取特征并获取对齐后的特征图
        features, all_features = loc_model.transformer(img_tensor)
        global_feature = features[:, 0]
        patch_features = features[:, 1:]
        aligned_feat = loc_model.feat_alignment(global_feature, patch_features)
        
        if aligned_feat is None:
            return np.array(original_img)
        
        # 获取分区边界
        boundaries = loc_model.get_boundary(aligned_feat)
        boundary = boundaries[0]  # 取第一个batch
        
        # 创建图像副本用于绘制
        img_np = np.array(original_img).copy()
        h_orig, w_orig = img_np.shape[:2]
        
        H, W = aligned_feat.size(2), aligned_feat.size(3)
        # 计算缩放比例
        scale_h = h_orig / H
        scale_w = w_orig / W
        
        # 绘制分区框（绘制内层和外层两个框）
        if len(boundary) >= 2:
            # 内层框（较小的）
            inner_size = boundary[0]
            inner_size_scaled_h = int(inner_size * scale_h)
            inner_size_scaled_w = int(inner_size * scale_w)
            
            # 外层框（较大的，通常是最后一个）
            outer_size = boundary[-1] if len(boundary) > 1 else boundary[0]
            outer_size_scaled_h = int(outer_size * scale_h)
            outer_size_scaled_w = int(outer_size * scale_w)
            
            # 使用PIL/OpenCV绘制
            img_with_boxes = img_np.copy()
            
            # 绘制外层框（虚线）
            cv2.rectangle(img_with_boxes, 
                         (w_orig//2 - outer_size_scaled_w, h_orig//2 - outer_size_scaled_h),
                         (w_orig//2 + outer_size_scaled_w, h_orig//2 + outer_size_scaled_h),
                         (0, 255, 255), 2)  # 青色
            
            # 绘制内层框（虚线）
            cv2.rectangle(img_with_boxes,
                         (w_orig//2 - inner_size_scaled_w, h_orig//2 - inner_size_scaled_h),
                         (w_orig//2 + inner_size_scaled_w, h_orig//2 + inner_size_scaled_h),
                         (0, 255, 255), 2)  # 青色
            
            # 使用matplotlib添加虚线效果（OpenCV不支持虚线，所以用matplotlib）
            fig, ax = plt.subplots(1, 1, figsize=(w_orig/100, h_orig/100), dpi=100)
            ax.imshow(img_np)
            
            # 外层框
            outer_rect = patches.Rectangle(
                (w_orig//2 - outer_size_scaled_w, h_orig//2 - outer_size_scaled_h),
                2 * outer_size_scaled_w, 2 * outer_size_scaled_h,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
            )
            ax.add_patch(outer_rect)
            
            # 内层框
            inner_rect = patches.Rectangle(
                (w_orig//2 - inner_size_scaled_w, h_orig//2 - inner_size_scaled_h),
                2 * inner_size_scaled_w, 2 * inner_size_scaled_h,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
            )
            ax.add_patch(inner_rect)
            
            ax.axis('off')
            ax.set_xlim(0, w_orig)
            ax.set_ylim(h_orig, 0)
            
            # 转换为numpy数组
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            partition_img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return partition_img
    
    return img_np

def get_heatmap_image(model, img_path, opt):
    """获取热力图"""
    _, heatmap_img = generate_heatmap_data(model, img_path, opt)
    return heatmap_img

# 处理所有图像
print("\n正在处理图像...")

# 处理卫星图像
print("处理卫星图像...")
satellite_img, satellite_tensor = load_and_preprocess_image(opt.satellite_img)
satellite_input = np.array(satellite_img)
satellite_aligned = get_aligned_image(model, satellite_tensor, satellite_img, img_size=opt.h)
satellite_partition = get_partition_image(model, satellite_tensor, satellite_img)
satellite_heatmap = get_heatmap_image(model, opt.satellite_img, opt)

# 处理无人机图像
drone_results = []
for i, drone_path in enumerate(drone_img_paths[:3]):  # 只取前3张
    print(f"处理无人机图像 {i+1}...")
    drone_img, drone_tensor = load_and_preprocess_image(drone_path)
    drone_input = np.array(drone_img)
    drone_aligned = get_aligned_image(model, drone_tensor, drone_img, img_size=opt.h)
    drone_partition = get_partition_image(model, drone_tensor, drone_img)
    drone_heatmap = get_heatmap_image(model, drone_path, opt)
    
    drone_results.append({
        'input': drone_input,
        'aligned': drone_aligned,
        'partition': drone_partition,
        'heatmap': drone_heatmap
    })

# 创建4x4网格图
print("\n正在生成对比图...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# 行标签
row_labels = ['Input', 'Aligned', 'Partition', 'Heatmap']
# 列标签
col_labels = ['Satellite-view', 'Drone-view 1', 'Drone-view 2', 'Drone-view 3']

# 卫星图像列
satellite_data = [satellite_input, satellite_aligned, satellite_partition, satellite_heatmap]
for row in range(4):
    axes[row, 0].imshow(satellite_data[row])
    axes[row, 0].axis('off')
    if row == 0:
        axes[row, 0].set_title(col_labels[0], fontsize=14, fontweight='bold', pad=10)

# 无人机图像列
for col in range(1, 4):
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
            axes[row, col].set_title(col_labels[col], fontsize=14, fontweight='bold', pad=10)

# 添加行标签（左侧）
for row in range(4):
    fig.text(0.02, 0.75 - row * 0.25, row_labels[row], 
             fontsize=14, fontweight='bold', rotation=90, 
             ha='center', va='center')

# 添加分隔线
for col in range(1, 4):
    for row in range(4):
        axes[row, col].axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# 保存图像
save_path = f'paper_figure_epoch_{opt.epoch}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print('=' * 60)
print(f'✓ 论文实验结果图已保存至: {save_path}')
print('=' * 60)
