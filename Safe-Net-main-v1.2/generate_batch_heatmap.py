#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量生成热力图脚本
对指定目录下的所有图像生成aligned_feature_map热力图
基于aligned_feature_map生成，保存到指定输出目录下
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
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.heatmap_utils import generate_heatmap_data
from torchvision import transforms
from datasets.queryDataset import Query_transforms

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Batch Generate Heatmaps')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录（包含所有要处理的图像）')
parser.add_argument('--output_dir', type=str, required=True, help='输出目录（保存热力图）')
parser.add_argument('--mode', type=int, default=2, help='模式: 1=drone->satellite, 2=satellite->drone')
parser.add_argument('--use_query_transform', action='store_true', help='是否使用query_transforms（用于query_drone）')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--epoch', default=119, type=int, help='epoch number')
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
print(f'输入目录: {opt.input_dir}')
print(f'输出目录: {opt.output_dir}')
print(f'模式: {opt.mode} ({'drone->satellite' if opt.mode == 1 else 'satellite->drone'})')
print(f'使用query_transform: {opt.use_query_transform}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
print('=' * 60)

# 检查输入目录是否存在
input_dir = Path(opt.input_dir)
if not input_dir.exists():
    raise FileNotFoundError(f"输入目录不存在: {input_dir}")

# 检查checkpoint是否存在
checkpoint_path = os.path.join(checkpoint_dir, f'net_{opt.epoch:03d}.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

# 加载模型
print(f"\n正在加载模型: {checkpoint_path}")
opt.checkpoint = checkpoint_path
opt.save_intermediate = False  # 不需要保存中间输出
model = load_network(opt, gpu_ids)
model = model.eval()
if use_gpu:
    model = model.cuda()

print("模型加载完成！\n")

# 创建输出目录
output_dir = Path(opt.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

def get_all_images(directory):
    """获取目录下所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    images = []
    if directory.exists():
        for ext in image_extensions:
            images.extend(directory.rglob(f'*{ext}'))
    return sorted(images)

def generate_heatmap_for_image(model, img_path, opt, output_path, use_query_transform=False):
    """为单张图像生成热力图并保存"""
    try:
        # 创建transform（generate_heatmap_data内部会使用，但我们需要先定义好）
        if use_query_transform:
            # query_drone使用query_transforms
            transform = transforms.Compose([
                transforms.Resize((opt.h, opt.w), interpolation=3),
                Query_transforms(pad=0, size=opt.w),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # gallery_satellite使用普通transforms
            transform = transforms.Compose([
                transforms.Resize((opt.h, opt.w), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # 临时保存原始transform并设置新的
        original_transform = getattr(opt, 'transform', None)
        opt.transform = transform
        
        # 生成热力图
        original_img, heatmap_img = generate_heatmap_data(model, str(img_path), opt)
        
        # 恢复transform
        if original_transform is not None:
            opt.transform = original_transform
        elif hasattr(opt, 'transform'):
            delattr(opt, 'transform')
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存热力图（只保存热力图，不叠加原图）
        # 添加"w/ DSA"标识
        heatmap_with_label = heatmap_img.copy()
        
        # 在图像上添加文字标识
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text = "w/ DSA"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 在左上角添加文字（白色文字，黑色背景）
        text_x = 10
        text_y = 30
        cv2.rectangle(heatmap_with_label, 
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      (0, 0, 0), -1)  # 黑色背景
        cv2.putText(heatmap_with_label, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        # 保存为PNG格式
        cv2.imwrite(str(output_path), cv2.cvtColor(heatmap_with_label, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"  错误: {str(e)}")
        return False

# 获取所有图像
print(f"\n正在扫描输入目录: {input_dir}")
all_images = get_all_images(input_dir)
print(f"找到 {len(all_images)} 张图像")

if len(all_images) == 0:
    print("错误: 未找到任何图像文件")
    sys.exit(1)

# 设置mode和transform
use_query_transform = opt.use_query_transform

print(f"\n开始处理图像...")
print(f"模式: {opt.mode} ({'drone->satellite' if opt.mode == 1 else 'satellite->drone'})")
print(f"使用query_transform: {use_query_transform}")

# 处理每张图像
total_images = 0
success_count = 0
fail_count = 0

for img_path in tqdm(all_images, desc="生成热力图"):
    total_images += 1
    
    # 计算相对路径，保持目录结构
    relative_path = img_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    
    # 确保输出文件扩展名为.png
    output_path = output_path.with_suffix('.png')
    
    # 如果文件已存在，跳过
    if output_path.exists():
        continue
    
    # 生成热力图
    if generate_heatmap_for_image(model, img_path, opt, output_path, use_query_transform):
        success_count += 1
    else:
        fail_count += 1
        print(f"  失败: {img_path}")

print('\n' + '=' * 60)
print('处理完成！')
print(f'总计: {total_images} 张图像')
print(f'成功: {success_count} 张')
print(f'失败: {fail_count} 张')
print(f'输出目录: {output_dir}')
print('=' * 60)
