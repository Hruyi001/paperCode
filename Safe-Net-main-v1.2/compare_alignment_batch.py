#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量对比原图和FAM对齐后的图像
为每张图像单独生成一个对比图（原图+对齐图）
可以遍历指定类别下的所有图像，或者遍历query_drone下的所有类别和所有图像
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
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from utils.utils_server import load_network
from utils.visualize import align_image

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Batch Compare Original and Aligned Images')
parser.add_argument('--name', default='SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652', type=str, help='model name')
parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录路径')
parser.add_argument('--class_id', type=str, default=None, help='类别ID（如 0001）。如果不指定，将遍历query_drone下的所有类别')
parser.add_argument('--view_type', type=str, choices=['drone', 'satellite'], default='drone', 
                   help='视图类型: drone 或 satellite')
parser.add_argument('--epoch', type=int, default=119, help='使用的模型epoch')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint目录路径，默认使用checkpoints/name')
parser.add_argument('--output_dir', type=str, default='./alignment_comparisons', help='输出目录')
parser.add_argument('--mode', type=int, default=None, help='1: drone->satellite, 2: satellite->drone (自动根据view_type设置)')
parser.add_argument('--debug', action='store_true', help='打印theta参数详细信息')
parser.add_argument('--save_params_only', action='store_true', help='只保存参数到txt文件，不生成对齐结果图')
parser.add_argument('--params_file', type=str, default=None, help='参数保存文件路径（默认：output_dir/alignment_params.txt）')

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
    if opt.mode is None:
        if 'mode' in config:
            opt.mode = config['mode']
        else:
            # 根据view_type自动设置
            opt.mode = 1 if opt.view_type == 'drone' else 2
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
    if opt.mode is None:
        opt.mode = 1 if opt.view_type == 'drone' else 2

# 设置save_intermediate为True以保存中间参数
opt.save_intermediate = True

print('=' * 60)
print(f'模型名称: {opt.name}')
print(f'数据集路径: {opt.dataset_root}')
print(f'类别ID: {opt.class_id if opt.class_id else "所有类别"}')
print(f'视图类型: {opt.view_type}')
print(f'Epoch: {opt.epoch}')
print(f'Checkpoint目录: {checkpoint_dir}')
mode_str = 'drone->satellite' if opt.mode == 1 else 'satellite->drone'
print(f'模式: {opt.mode} ({mode_str})')
print('=' * 60)

# 获取所有类别列表
dataset_root = Path(opt.dataset_root)

def get_all_classes(dataset_root, view_type):
    """获取所有类别ID列表"""
    if view_type == 'drone':
        drone_root = dataset_root / 'query_drone'
        if not drone_root.exists():
            raise FileNotFoundError(f"无人机图像根目录不存在: {drone_root}")
        # 获取所有类别目录
        class_dirs = [d for d in drone_root.iterdir() if d.is_dir()]
        class_ids = sorted([d.name for d in class_dirs])
        return class_ids
    else:
        satellite_root = dataset_root / 'query_satellite'
        if not satellite_root.exists():
            raise FileNotFoundError(f"卫星图像根目录不存在: {satellite_root}")
        class_dirs = [d for d in satellite_root.iterdir() if d.is_dir()]
        class_ids = sorted([d.name for d in class_dirs])
        return class_ids

def get_image_paths_for_class(dataset_root, class_id, view_type):
    """获取指定类别的所有图像路径"""
    img_paths = []
    
    if view_type == 'drone':
        drone_dir = dataset_root / 'query_drone' / class_id
        if not drone_dir.exists():
            return []
        
        # 获取所有图像文件
        img_files = sorted(list(drone_dir.glob("*.jpg")) + 
                          list(drone_dir.glob("*.jpeg")) + 
                          list(drone_dir.glob("*.png")))
        
        img_paths = [str(f) for f in img_files]
        
    elif view_type == 'satellite':
        satellite_file = dataset_root / 'query_satellite' / class_id / f"{class_id}.jpg"
        if satellite_file.exists():
            img_paths = [str(satellite_file)]
    
    return img_paths

# 确定要处理的类别列表
if opt.class_id:
    # 只处理指定的类别
    class_ids = [opt.class_id]
else:
    # 处理所有类别
    print("\n正在扫描所有类别...")
    class_ids = get_all_classes(dataset_root, opt.view_type)
    print(f"找到 {len(class_ids)} 个类别")

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
    
    # 保存原始图像（用于显示，如果不需要生成图像则跳过）
    if hasattr(opt, 'save_params_only') and opt.save_params_only:
        original_img = None  # 不保存原始图像，节省内存
    else:
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
    if isinstance(theta, torch.Tensor):
        theta_np = theta[0].cpu().numpy() if theta.is_cuda else theta[0].numpy()
    else:
        theta_np = theta[0] if isinstance(theta, (list, tuple)) else theta
    
    # 确保theta_np是2x3的numpy数组
    theta_np = np.array(theta_np).reshape(2, 3)
    
    # 生成对齐图（如果不需要保存参数，可以跳过）
    if hasattr(opt, 'save_params_only') and opt.save_params_only:
        aligned_img = None  # 不生成对齐图，节省时间
    else:
        aligned_img = align_image(original_img, theta_np, img_size=opt.h, debug=debug)
    
    # 分析theta参数
    # theta矩阵格式: [[a, b, tx], [c, d, ty]]
    # 对于旋转+缩放的组合变换，正确的分解方式：
    theta_tensor = torch.tensor(theta_np, dtype=torch.float32)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=theta_tensor.dtype)
    diff = torch.abs(theta_tensor - identity).sum().item()
    
    # 提取旋转、缩放、平移参数
    # 对于旋转矩阵 R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    # 如果有缩放，则 S*R = [[s*cos(θ), -s*sin(θ)], [s*sin(θ), s*cos(θ)]]
    # 所以：a = s*cos(θ), b = -s*sin(θ), c = s*sin(θ), d = s*cos(θ)
    
    a, b, tx = theta_tensor[0, 0].item(), theta_tensor[0, 1].item(), theta_tensor[0, 2].item()
    c, d, ty = theta_tensor[1, 0].item(), theta_tensor[1, 1].item(), theta_tensor[1, 2].item()
    
    # 计算旋转角度（从旋转矩阵的第一列或使用atan2(c, a)）
    # 如果只有旋转+均匀缩放，角度可以从第一列计算
    angle = math.atan2(c, a) * 180 / math.pi
    
    # 计算缩放（对于均匀缩放，使用第一列或第二列的模长）
    # 对于旋转+均匀缩放：scale = sqrt(a^2 + c^2) = sqrt(b^2 + d^2)
    scale_x = math.sqrt(a**2 + c**2)
    scale_y = math.sqrt(b**2 + d**2)
    # 使用平均缩放或主要缩放
    scale = scale_x  # 或者使用 (scale_x + scale_y) / 2
    
    # 平移参数（已经是归一化坐标）
    translation = (tx, ty)
    translation_magnitude = math.sqrt(tx**2 + ty**2)
    
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

# 创建输出目录
os.makedirs(opt.output_dir, exist_ok=True)

# 参数文件写入策略：
# - 如果开启 --save_params_only：默认写到 output_dir/alignment_params.txt（除非显式传 --params_file）
# - 如果未开启 --save_params_only：只有显式传 --params_file 才会写
params_file_enabled = opt.save_params_only or (opt.params_file is not None)
if opt.save_params_only and opt.params_file is None:
    opt.params_file = os.path.join(opt.output_dir, 'alignment_params.txt')

params_f = None
params_written = 0
if params_file_enabled:
    params_dir = os.path.dirname(opt.params_file) or "."
    os.makedirs(params_dir, exist_ok=True)
    params_f = open(opt.params_file, 'w', encoding='utf-8')
    params_f.write("图像路径\t类别ID\t图像名称\tRotation(度)\tScale\tTranslation_X\tTranslation_Y\n")

# 处理所有类别和所有图像
total_success = 0
total_failed = 0
total_images = 0

try:
    for class_idx, class_id in enumerate(class_ids):
        print("\n" + "=" * 60)
        print(f"[类别 {class_idx+1}/{len(class_ids)}] 处理类别: {class_id}")
        print("=" * 60)
        
        # 获取该类别的所有图像路径
        img_paths = get_image_paths_for_class(dataset_root, class_id, opt.view_type)
        
        if not img_paths:
            print(f"⚠ 类别 {class_id} 中没有找到图像，跳过")
            continue
        
        print(f"找到 {len(img_paths)} 张图像")
        total_images += len(img_paths)
        
        # 处理该类别的所有图像
        for i, img_path in enumerate(img_paths):
            try:
                # 获取图像文件名（不含扩展名）
                img_name = Path(img_path).stem
                
                print(f"\n  [{i+1}/{len(img_paths)}] 处理: {Path(img_path).name}")
                
                # 处理图像
                result = process_single_image(model, img_path, opt, debug=opt.debug)
                
                # 打印theta矩阵（用于调试，即使不在debug模式也显示前几个）
                if i < 3 or opt.debug:
                    theta_matrix = np.array(result['info']['theta'])
                    print(f"    Theta矩阵值: [[{theta_matrix[0,0]:.6f}, {theta_matrix[0,1]:.6f}, {theta_matrix[0,2]:.6f}],")
                    print(f"                   [{theta_matrix[1,0]:.6f}, {theta_matrix[1,1]:.6f}, {theta_matrix[1,2]:.6f}]]")
                
                if opt.debug:
                    theta_matrix = np.array(result['info']['theta'])
                    print(f"    Theta矩阵:\n{theta_matrix}")
                    print(f"    矩阵元素: a={theta_matrix[0,0]:.6f}, b={theta_matrix[0,1]:.6f}, tx={theta_matrix[0,2]:.6f}")
                    print(f"              c={theta_matrix[1,0]:.6f}, d={theta_matrix[1,1]:.6f}, ty={theta_matrix[1,2]:.6f}")
                    print(f"    与单位矩阵差异: {result['info']['diff_from_identity']:.6f}")
                    print(f"    旋转角度: {result['info']['rotation_angle']:.2f}°")
                    print(f"    缩放: {result['info']['scale']:.4f}")
                    print(f"    平移: {result['info']['translation']}")
                    print(f"    对齐效果: {result['info']['effect_level']}")
                
                # 打印参数信息
                print(f"    Rotation: {result['info']['rotation_angle']:.2f}°")
                print(f"    Scale: {result['info']['scale']:.4f}")
                print(f"    Translation: ({result['info']['translation'][0]:.4f}, {result['info']['translation'][1]:.4f})")
                
                # 立即写入参数文件（避免中途 Ctrl+C 时丢数据）
                if params_f is not None:
                    params_f.write(
                        f"{img_path}\t{class_id}\t{Path(img_path).name}\t"
                        f"{result['info']['rotation_angle']:.6f}\t{result['info']['scale']:.6f}\t"
                        f"{result['info']['translation'][0]:.6f}\t{result['info']['translation'][1]:.6f}\n"
                    )
                    params_written += 1
                    # 定期flush，确保落盘
                    if params_written % 100 == 0:
                        params_f.flush()
                
                # 如果不只保存参数，则生成对比图
                if not opt.save_params_only:
                    # 创建对比图（单张图像：左右并排）
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # 原图
                    axes[0].imshow(result['original'])
                    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
                    axes[0].axis('off')
                    
                    # 对齐图
                    title = f"Aligned Image\n(Effect: {result['info']['effect_level']})"
                    axes[1].imshow(result['aligned'])
                    axes[1].set_title(title, fontsize=14, fontweight='bold', pad=10)
                    axes[1].axis('off')
                    
                    # 添加信息文本
                    info_text = f"Rotation: {result['info']['rotation_angle']:.2f}°\n"
                    info_text += f"Scale: {result['info']['scale']:.4f}\n"
                    info_text += f"Translation: ({result['info']['translation'][0]:.4f}, {result['info']['translation'][1]:.4f})"
                    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.05)
                    
                    # 生成输出文件名
                    output_file = os.path.join(opt.output_dir, f"comparison_{class_id}_{img_name}.png")
                    
                    # 保存图像
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"    ✓ 已保存: {output_file}")
                
                total_success += 1
                
            except Exception as e:
                print(f"    ✗ 处理失败: {str(e)}")
                total_failed += 1
                continue
except KeyboardInterrupt:
    print("\n⚠ 收到 Ctrl+C，中断处理。已写入的参数不会丢失。")
finally:
    if params_f is not None:
        params_f.flush()
        params_f.close()

# 总结
print("\n" + "=" * 60)
print("批量处理完成！")
print(f"  总类别数: {len(class_ids)}")
print(f"  总图像数: {total_images}")
print(f"  成功: {total_success} 张")
print(f"  失败: {total_failed} 张")
if params_file_enabled:
    print(f"  参数文件: {opt.params_file}")
    print(f"  已写入记录数: {params_written}")
else:
    print(f"  输出目录: {opt.output_dir}")
print("=" * 60)
