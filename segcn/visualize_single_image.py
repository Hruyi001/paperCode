"""
单图像对比热力图可视化脚本
用于快速可视化单张图像的对比热力图

使用方法:
    python visualize_single_image.py --image_path <图像路径> --checkpoint <模型权重路径>
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample4geo.hand_convnext.model import make_model
from visualize_comparison_heatmap import load_model, generate_comparison_heatmap


def main():
    parser = argparse.ArgumentParser(description='单图像对比热力图可视化')
    parser.add_argument('--image_path', type=str, required=True,
                       help='图像路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重路径（可选）')
    parser.add_argument('--output_path', type=str, default='./comparison_heatmap.png',
                       help='输出图像路径')
    parser.add_argument('--img_size', type=int, default=384,
                       help='图像尺寸')
    parser.add_argument('--num_classes', type=int, default=701,
                       help='类别数量')
    parser.add_argument('--block', type=int, default=2,
                       help='block数量')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查图像路径
    if not os.path.exists(args.image_path):
        print(f"错误: 图像路径不存在: {args.image_path}")
        return
    
    # 加载模型
    model = load_model(args.checkpoint, device, args.num_classes, args.block)
    
    # 生成对比热力图
    print(f"处理图像: {args.image_path}")
    generate_comparison_heatmap(
        model, args.image_path, device, args.img_size, 
        save_path=args.output_path
    )
    
    print(f"对比热力图已保存到: {args.output_path}")


if __name__ == '__main__':
    main()
