#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对比原图和FAM对齐后的图像
将原图和对齐图并排显示，方便直观对比对齐效果
"""

from __future__ import print_function, division

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import math

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.visualize import align_image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Compare Original and Aligned Images')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--img', type=str, required=True, help='图像路径（可以是单张图像或逗号分隔的多张图像）')
parser.add_argument('--epoch', type=int, default=119, help='使用的模型epoch')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')
parser.add_argument('--output', type=str, default='alignment_comparison.png', help='输出图像路径')
parser.add_argument('--mode', type=int, default=1, help='1: drone->satellite (输入为无人机图), 2: satellite->drone (输入为卫星图)')
parser.add_argument('--debug', action='store_true', help='打印theta参数详细信息')

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
    if 'mode' in config:
        opt.mode = config['mode']
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

# 设置save_intermediate为True以保存中间参数
opt.save_intermediate = True

print('=' * 60)
print(f'模型名称: {opt.name}')
print(f'图像路径: {opt.img}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
print(f'模式: {"drone->satellite" if opt.mode == 1 else "satellite->drone"}')
print('=' * 60)

# 解析图像路径
img_paths = [p.strip() for p in opt.img.split(',')]

# 检查图像是否存在
for i, path in enumerate(img_paths):
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像 {i+1} 路径不存在: {path}")

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

def process_single_image(model, img_path, opt, debug=False):
    """
    处理单张图像，获取原图和对齐图
    
    Args:
        model: 训练好的模型
        img_path: 图像路径
        opt: 配置参数
        debug: 是否打印调试信息
    
    Returns:
        dict: 包含'original', 'aligned', 'theta', 'info'的字典
    """
    # 加载图像
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    if use_gpu:
        img_tensor = img_tensor.cuda()
    
    # 保存原始图像（用于显示）
    original_img = np.array(img.resize((opt.h, opt.w), Image.LANCZOS))
    
    # 前向传播
    with torch.no_grad():
        if opt.mode == 1:
            # drone->satellite: 输入为无人机图
            outputs, _ = model(img_tensor, None)
        else:
            # satellite->drone: 输入为卫星图
            _, outputs = model(None, img_tensor)
    
    # 获取中间输出
    if hasattr(model, 'module'):
        intermediate_outputs = model.module.intermediate_outputs
    else:
        intermediate_outputs = model.intermediate_outputs
    
    theta = intermediate_outputs.get('theta', None)
    
    if theta is None:
        raise ValueError("未找到theta参数，请确保模型设置了save_intermediate=True")
    
    # 转换为numpy
    theta_np = theta[0].numpy() if isinstance(theta, torch.Tensor) else theta[0]
    
    # 生成对齐图
    aligned_img = align_image(original_img, theta_np, img_size=opt.h, debug=debug)
    
    # 分析theta参数
    theta_tensor = torch.tensor(theta_np, dtype=torch.float32)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=theta_tensor.dtype)
    diff = torch.abs(theta_tensor - identity).sum().item()
    
    angle = math.atan2(theta_tensor[1, 0].item(), theta_tensor[0, 0].item()) * 180 / math.pi
    scale = math.sqrt(theta_tensor[0, 0].item()**2 + theta_tensor[1, 0].item()**2)
    translation = (theta_tensor[0, 2].item(), theta_tensor[1, 2].item())
    translation_magnitude = math.sqrt(translation[0]**2 + translation[1]**2)
    
    info = {
        'theta': theta_np.tolist(),
        'diff_from_identity': diff,
        'rotation_angle': angle,
        'scale': scale,
        'translation': translation,
        'translation_magnitude': translation_magnitude,
        'effect_level': '不明显' if diff < 0.01 else ('中等' if diff < 0.1 else '明显')
    }
    
    return {
        'original': original_img,
        'aligned': aligned_img,
        'theta': theta_np,
        'info': info
    }

# 处理所有图像
print("\n正在处理图像...")
results = []
for i, img_path in enumerate(img_paths):
    print(f"处理图像 {i+1}/{len(img_paths)}: {img_path}")
    result = process_single_image(model, img_path, opt, debug=opt.debug)
    results.append(result)
    
    if opt.debug:
        print(f"  Theta矩阵:\n{np.array(result['info']['theta'])}")
        print(f"  与单位矩阵差异: {result['info']['diff_from_identity']:.6f}")
        print(f"  旋转角度: {result['info']['rotation_angle']:.2f}°")
        print(f"  缩放: {result['info']['scale']:.4f}")
        print(f"  平移: {result['info']['translation']}")
        print(f"  对齐效果: {result['info']['effect_level']}")

# 创建对比图
print("\n正在生成对比图...")

num_images = len(results)
if num_images == 1:
    # 单张图像：左右并排
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(results[0]['original'])
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(results[0]['aligned'])
    title = f"Aligned Image\n(Effect: {results[0]['info']['effect_level']})"
    axes[1].set_title(title, fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # 添加信息文本
    info_text = f"Rotation: {results[0]['info']['rotation_angle']:.2f}°\n"
    info_text += f"Scale: {results[0]['info']['scale']:.4f}\n"
    info_text += f"Translation: ({results[0]['info']['translation'][0]:.4f}, {results[0]['info']['translation'][1]:.4f})"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
else:
    # 多张图像：2列布局（原图 | 对齐图）
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))
    
    for i, result in enumerate(results):
        # 原图
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f'Original Image {i+1}', fontsize=12, fontweight='bold', pad=10)
        axes[i, 0].axis('off')
        
        # 对齐图
        title = f"Aligned Image {i+1}\n(Effect: {result['info']['effect_level']})"
        axes[i, 1].imshow(result['aligned'])
        axes[i, 1].set_title(title, fontsize=12, fontweight='bold', pad=10)
        axes[i, 1].axis('off')
        
        # 添加信息文本（在图像下方）
        info_text = f"Rot: {result['info']['rotation_angle']:.2f}° | "
        info_text += f"Scale: {result['info']['scale']:.4f} | "
        info_text += f"Trans: ({result['info']['translation'][0]:.3f}, {result['info']['translation'][1]:.3f})"
        axes[i, 1].text(0.5, -0.08, info_text, transform=axes[i, 1].transAxes,
                       ha='center', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.05 if num_images == 1 else None)

# 保存图像
save_path = opt.output
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 对比图已保存到: {save_path}")

# 显示图像（如果在交互式环境中）
try:
    plt.show()
except:
    print("(非交互式环境，跳过显示)")

plt.close()

print("\n处理完成！")
