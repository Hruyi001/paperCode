#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热力图可视化脚本
从指定目录读取图像，生成模型关注区域的热力图可视化对比图
"""

import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation_utils.load_network import load_network_supervised
from utils.commons import load_config


class HeatmapVisualizer:
    def __init__(self, model, device, img_size=(280, 280)):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.x_fine_0 = None
        
        # 注册hook来捕获x_fine_0
        self._register_hook()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _register_hook(self):
        """注册hook来捕获AQEU模块输出的x_fine_0特征图"""
        def hook_fn(module, input, output):
            # AQEU返回(y, x_fine_0)，我们捕获x_fine_0
            if isinstance(output, tuple) and len(output) == 2:
                self.x_fine_0 = output[1].detach().cpu()
        
        # 找到AQEU模块并注册hook
        aqeu_found = False
        for name, module in self.model.named_modules():
            if 'AQEU' in name and hasattr(module, 'forward'):
                module.register_forward_hook(hook_fn)
                print(f"Registered hook on: {name}")
                aqeu_found = True
                break
        
        if not aqeu_found:
            # 尝试直接访问components中的AQEU
            if hasattr(self.model, 'components') and hasattr(self.model.components, 'AQEU'):
                self.model.components.AQEU.register_forward_hook(hook_fn)
                print("Registered hook on: components.AQEU")
                aqeu_found = True
        
        if not aqeu_found:
            print("Warning: Could not find AQEU module for hook registration")
            print("Will try direct extraction method")
    
    def load_image(self, img_path):
        """加载并预处理图像"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            return img, img_tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
    
    def generate_heatmap(self, feature_map, method='mean'):
        """
        从特征图生成热力图
        
        Args:
            feature_map: [1, C, H, W] 特征图
            method: 'mean' - 通道平均, 'max' - 通道最大值, 'norm' - L2范数
        
        Returns:
            heatmap: [H, W] numpy array
        """
        if feature_map is None:
            return None
        
        # 确保是numpy数组
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.numpy()
        
        # 移除batch维度
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
        
        # [C, H, W] -> [H, W]
        if method == 'mean':
            heatmap = np.mean(feature_map, axis=0)
        elif method == 'max':
            heatmap = np.max(feature_map, axis=0)
        elif method == 'norm':
            heatmap = np.linalg.norm(feature_map, axis=0)
        else:
            heatmap = np.mean(feature_map, axis=0)
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def visualize(self, img_path, save_path=None):
        """
        对单张图像生成热力图可视化
        
        Args:
            img_path: 图像路径
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            fig: matplotlib figure对象
        """
        # 加载图像
        img_pil, img_tensor = self.load_image(img_path)
        if img_pil is None:
            return None
        
        # 重置特征图
        self.x_fine_0 = None
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # 检查是否成功捕获特征图
        if self.x_fine_0 is None:
            print(f"Warning: Failed to capture x_fine_0 for {img_path}, trying direct extraction...")
            # 尝试直接从组件获取
            try:
                # 手动调用模型来获取特征
                backbone_output = self.model.backbone(img_tensor)
                if isinstance(backbone_output, tuple):
                    _, feature_map = backbone_output
                else:
                    feature_map = backbone_output
                
                # 调用AQEU - QDFL中AQEU返回(aqeu_feature, x_fine)，其中x_fine就是x_fine_0
                if hasattr(self.model.components, 'AQEU'):
                    aqeu_output = self.model.components.AQEU(feature_map)
                    if isinstance(aqeu_output, tuple) and len(aqeu_output) == 2:
                        self.x_fine_0 = aqeu_output[1].detach().cpu()
                        print(f"Successfully extracted x_fine_0 via direct call, shape: {self.x_fine_0.shape}")
                    else:
                        # 如果AQEU只返回一个值，尝试从QDFL的forward中获取
                        # 通过hook在QDFL的forward中捕获
                        pass
            except Exception as e:
                print(f"Error extracting feature: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        if self.x_fine_0 is None:
            print(f"Error: Could not extract feature map for {img_path}")
            return None
        
        # 生成热力图
        heatmap = self.generate_heatmap(self.x_fine_0, method='mean')
        
        # 调整热力图大小到原图大小
        img_array = np.array(img_pil)
        h, w = img_array.shape[:2]
        heatmap_resized = np.array(Image.fromarray(heatmap).resize((w, h), Image.BICUBIC))
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原图
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # 热力图
        im = axes[1].imshow(heatmap_resized, cmap='jet', interpolation='bilinear')
        axes[1].set_title('Heatmap (x_fine_0)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # 叠加图
        axes[2].imshow(img_array)
        overlay = axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        return fig


def get_image_files(directory, num_images=10):
    """从目录中获取图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    # 随机选择指定数量的图像
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description='Generate heatmap visualizations for QDFL model')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to visualize')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--config', type=str, default='./model_configs/dino_b_QDFL.yaml',
                        help='Path to model config file')
    parser.add_argument('--output_dir', type=str, default='./heatmap_visualizations',
                        help='Directory to save visualization results')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to visualize')
    parser.add_argument('--img_size', type=int, nargs=2, default=[280, 280],
                        help='Input image size [height width]')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    print("Loading model configuration...")
    config = load_config(args.config)
    model_configs = config['model_configs']
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_network_supervised(model_configs, args.checkpoint)
    print(f"Model loaded on {device}")
    
    # 创建可视化器
    visualizer = HeatmapVisualizer(model, device, img_size=tuple(args.img_size))
    
    # 获取图像文件
    print(f"Scanning images in {args.image_dir}...")
    image_files = get_image_files(args.image_dir, args.num_images)
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_files)} images. Generating visualizations...")
    
    # 生成可视化
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # 生成保存路径
        img_name = Path(img_path).stem
        save_path = os.path.join(args.output_dir, f"{img_name}_heatmap.png")
        
        # 生成可视化
        fig = visualizer.visualize(img_path, save_path)
        
        if fig is not None:
            plt.close(fig)
    
    print(f"\nVisualization complete! Results saved to: {args.output_dir}")
    
    # 生成对比图（所有图像在一个图中）
    print("Generating comparison figure...")
    fig, axes = plt.subplots(len(image_files), 3, figsize=(18, 6 * len(image_files)))
    if len(image_files) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_files):
        img_pil, img_tensor = visualizer.load_image(img_path)
        if img_pil is None:
            continue
        
        visualizer.x_fine_0 = None
        with torch.no_grad():
            _ = model(img_tensor)
        
        if visualizer.x_fine_0 is not None:
            heatmap = visualizer.generate_heatmap(visualizer.x_fine_0, method='mean')
            img_array = np.array(img_pil)
            h, w = img_array.shape[:2]
            heatmap_resized = np.array(Image.fromarray(heatmap).resize((w, h), Image.BICUBIC))
            
            axes[idx, 0].imshow(img_array)
            axes[idx, 0].set_title(f'Original {idx+1}', fontsize=10)
            axes[idx, 0].axis('off')
            
            im = axes[idx, 1].imshow(heatmap_resized, cmap='jet', interpolation='bilinear')
            axes[idx, 1].set_title(f'Heatmap {idx+1}', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(img_array)
            axes[idx, 2].imshow(heatmap_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
            axes[idx, 2].set_title(f'Overlay {idx+1}', fontsize=10)
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, 'comparison_all.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved to: {comparison_path}")
    plt.close()


if __name__ == '__main__':
    main()
