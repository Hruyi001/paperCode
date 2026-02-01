#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成FAM对齐后的图像
根据给定的无人机图像，生成FAM模块处理后的对齐图像并保存
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
import math
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.visualize import align_image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Generate FAM Aligned Image')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--img', type=str, required=True, help='输入图像路径（无人机图像）')
parser.add_argument('--output', type=str, default=None, help='输出图像路径（默认：./output/<输入图像文件名>_aligned.jpg）')
parser.add_argument('--comparison', type=str, default=None, help='对比图输出路径（默认：./output/<输入图像文件名>_comparison.png），设置为空字符串则不生成对比图')
parser.add_argument('--epoch', type=int, default=119, help='使用的模型epoch')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')
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

# 检查输入图像是否存在
if not os.path.exists(opt.img):
    raise FileNotFoundError(f"输入图像不存在: {opt.img}")

# 获取项目根目录和输出目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')

# 获取输入图像的文件名
input_basename = os.path.basename(opt.img)
input_name, input_ext = os.path.splitext(input_basename)
if not input_ext:
    input_ext = '.jpg'

# 设置输出路径
if opt.output is None:
    # 默认输出路径：项目根目录下的output目录
    output_filename = f"{input_name}_aligned{input_ext}"
    opt.output = os.path.join(output_dir, output_filename)

# 设置对比图输出路径
if opt.comparison is None:
    # 默认生成对比图
    comparison_filename = f"{input_name}_comparison.png"
    opt.comparison = os.path.join(output_dir, comparison_filename)
elif opt.comparison == "":
    # 空字符串表示不生成对比图
    opt.comparison = None

# 确保输出目录存在
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print('=' * 60)
print(f'模型名称: {opt.name}')
print(f'输入图像: {opt.img}')
print(f'输出图像: {opt.output}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
print(f'模式: {"drone->satellite" if opt.mode == 1 else "satellite->drone"}')
print('=' * 60)

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

# 加载图像
print(f"\n正在处理图像: {opt.img}")
img = Image.open(opt.img).convert("RGB")
original_size = img.size  # 保存原始尺寸 (width, height)
img_tensor = transform(img).unsqueeze(0)
if use_gpu:
    img_tensor = img_tensor.cuda()

# 保存原始图像（用于对齐）
original_img = np.array(img.resize((opt.h, opt.w), Image.LANCZOS))

# 前向传播
print("正在进行FAM对齐...")
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
aligned_img = align_image(original_img, theta_np, img_size=opt.h, debug=opt.debug)

# 计算theta参数信息
theta_tensor = torch.tensor(theta_np, dtype=torch.float32)
identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=theta_tensor.dtype)
diff = torch.abs(theta_tensor - identity).sum().item()
angle = math.atan2(theta_tensor[1, 0].item(), theta_tensor[0, 0].item()) * 180 / math.pi
scale = math.sqrt(theta_tensor[0, 0].item()**2 + theta_tensor[1, 0].item()**2)
translation = (theta_tensor[0, 2].item(), theta_tensor[1, 2].item())
translation_magnitude = math.sqrt(translation[0]**2 + translation[1]**2)
effect_level = '不明显' if diff < 0.01 else ('中等' if diff < 0.1 else '明显')

# 如果需要，调整到原始图像尺寸
if original_size != (opt.w, opt.h):
    aligned_img_pil = Image.fromarray(aligned_img)
    aligned_img_pil = aligned_img_pil.resize(original_size, Image.LANCZOS)
    aligned_img = np.array(aligned_img_pil)
    
    # 同时调整原图尺寸以匹配
    original_img_resized = np.array(img.resize(original_size, Image.LANCZOS))
else:
    original_img_resized = original_img

# 保存对齐图像
aligned_img_pil = Image.fromarray(aligned_img)
aligned_img_pil.save(opt.output, quality=95)
print(f"\n✓ FAM对齐图像已保存到: {opt.output}")

# 生成对比图（如果指定了输出路径）
if opt.comparison:
    print("\n正在生成对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 原图
    axes[0].imshow(original_img_resized)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # 对齐图
    title = f"FAM Aligned Image\n(Effect: {effect_level})"
    axes[1].imshow(aligned_img)
    axes[1].set_title(title, fontsize=16, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # 添加详细的参数信息文本
    info_text = f"Affine Transformation Parameters:\n"
    info_text += f"Rotation Angle: {angle:.2f}°\n"
    info_text += f"Scale: {scale:.4f}\n"
    info_text += f"Translation: ({translation[0]:.4f}, {translation[1]:.4f})\n"
    info_text += f"Translation Magnitude: {translation_magnitude:.4f}\n"
    info_text += f"Difference from Identity: {diff:.6f}\n"
    info_text += f"\nTheta Matrix:\n"
    info_text += f"[{theta_np[0,0]:.4f}, {theta_np[0,1]:.4f}, {theta_np[0,2]:.4f}]\n"
    info_text += f"[{theta_np[1,0]:.4f}, {theta_np[1,1]:.4f}, {theta_np[1,2]:.4f}]"
    
    # 在图像底部添加参数信息
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存对比图
    plt.savefig(opt.comparison, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 对比图已保存到: {opt.comparison}")

# 打印theta信息（如果启用debug）
if opt.debug:
    print(f"\nTheta参数信息:")
    print(f"  Theta矩阵:\n{np.array(theta_np)}")
    print(f"  与单位矩阵差异: {diff:.6f}")
    print(f"  旋转角度: {angle:.2f}°")
    print(f"  缩放: {scale:.4f}")
    print(f"  平移: ({translation[0]:.4f}, {translation[1]:.4f})")
    print(f"  平移幅度: {translation_magnitude:.4f}")
    print(f"  对齐效果: {effect_level}")

print("\n处理完成！")
