"""
对比热力图生成脚本
生成 part_features 和 pfeat_align 的对比可视化，突出DSA模块的改进效果

使用方法:
    python visualize_comparison_heatmap.py --dataset_path <数据集路径> --checkpoint <模型权重路径>
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample4geo.hand_convnext.model import make_model


def load_model(checkpoint_path, device, num_classes=701, block=2, resnet=False):
    """加载模型"""
    class Config:
        def __init__(self):
            self.nclasses = num_classes
            self.block = block
            self.triplet_loss = True
            self.resnet = resnet
            self.views = 2  # 双视图网络（satellite和drone）
    
    config = Config()
    model = make_model(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理可能的字典包装（如 {'model': state_dict, ...}）
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 处理DataParallel保存的权重
        if len(checkpoint.keys()) > 0:
            first_key = list(checkpoint.keys())[0]
            if 'module.' in first_key:
                # 如果权重是用DataParallel保存的，需要去掉'module.'前缀
                new_checkpoint = {}
                for k, v in checkpoint.items():
                    new_key = k.replace('module.', '')
                    new_checkpoint[new_key] = v
                checkpoint = new_checkpoint
        
        # 移除分类器权重（如果存在）- 这些权重可能不匹配
        # 根据train.py中的处理方式，移除分类器的weight和bias
        keys_to_remove = []
        for key in checkpoint.keys():
            # 匹配格式: model_1.classifier1.classifier.0.weight/bias
            # 或: classifier1.classifier.0.weight/bias
            if ('classifier1.classifier.0.' in key or 
                'classifier_mcb' in key and '.classifier.0.' in key):
                if key.endswith('.weight') or key.endswith('.bias'):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in checkpoint:
                del checkpoint[key]
                print(f"移除分类器权重: {key}")
        
        model.load_state_dict(checkpoint, strict=False)
        print("模型权重加载完成")
    else:
        print("警告: 未找到模型权重，使用随机初始化的模型")
    
    model = model.to(device)
    return model


def preprocess_image(image_path, img_size=384):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # 获取resize后的图像（用于可视化）
    # PIL返回RGB格式，需要转换为BGR（OpenCV格式）
    img_resized = img.resize((img_size, img_size), Image.LANCZOS)
    original_img = np.array(img_resized)  # RGB格式 (H, W, 3)
    
    # 转换为BGR格式（OpenCV格式）
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # 确保是uint8类型
    if original_img.dtype != np.uint8:
        original_img = original_img.astype(np.uint8)
    
    return img_tensor, original_img


def generate_heatmap_from_features(features, img_size=384):
    """从特征图生成热力图"""
    # features: (B, C, H, W) 或 (B, C, H*W)
    if len(features.shape) == 3:
        # (B, C, H*W) 需要reshape
        B, C, HW = features.shape
        H = W = int(np.sqrt(HW))
        features = features.view(B, C, H, W)
    
    # 对通道维度求平均
    heatmap = features[0].mean(dim=0).detach().cpu().numpy()  # (H, W)
    
    # 归一化到[0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 上采样到原图大小
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    # 应用colormap
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap_resized, heatmap_colored


def generate_comparison_heatmap(model, image_path, device, img_size=384, save_path=None):
    """
    生成对比热力图：同时展示 part_features 和 pfeat_align
    突出DSA模块的改进效果
    """
    # 预处理图像
    img_tensor, original_img = preprocess_image(image_path, img_size)
    img_tensor = img_tensor.to(device)
    
    # 设置为训练模式以获取pfeat_align
    model.train()
    # 但是BatchNorm在batch_size=1时会报错，所以临时将BatchNorm设置为eval模式
    # 这样可以使用running statistics而不是计算batch statistics
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
        
        # 1. 获取 part_features (原始特征)
        if len(output) == 2:
            # 评估模式（不应该发生，因为设置了train模式）
            part_features = output[1]  # (B, 768, H, W)
        else:
            # 训练模式
            part_features = output[-1]  # (B, 768, H, W)
        
        # 获取part_features的空间维度，用于后续reshape pfeat_align
        _, _, h_feat, w_feat = part_features.shape
        
        heatmap_part, heatmap_part_colored = generate_heatmap_from_features(part_features, img_size)
        # 确保尺寸匹配（两者都应该是(img_size, img_size, 3)）
        if original_img.shape[:2] != heatmap_part_colored.shape[:2]:
            original_img_resized = cv2.resize(original_img, (heatmap_part_colored.shape[1], heatmap_part_colored.shape[0]))
        else:
            original_img_resized = original_img.copy()
        # 确保两者都是BGR格式且尺寸完全匹配
        assert original_img_resized.shape == heatmap_part_colored.shape, \
            f"尺寸不匹配: original_img {original_img_resized.shape} vs heatmap {heatmap_part_colored.shape}"
        blended_part = cv2.addWeighted(original_img_resized, 0.5, heatmap_part_colored, 0.5, 0)
        
        # 2. 获取 pfeat_align (DSA处理后的特征) - 仅在训练模式下可用
        if len(output) > 2:
            pfeat_align = output[0]  # (B, 512, H*W)
            # 使用part_features的空间维度来reshape pfeat_align
            B, C, HW = pfeat_align.shape
            # 验证HW是否等于h_feat * w_feat
            if HW != h_feat * w_feat:
                print(f"警告: pfeat_align的空间维度 {HW} 与 part_features {h_feat * w_feat} 不匹配，使用sqrt计算")
                h_align = w_align = int(np.sqrt(HW))
            else:
                h_align, w_align = h_feat, w_feat
            
            # Reshape为空间特征图
            pfeat_align_reshaped = pfeat_align.view(B, C, h_align, w_align)
            
            # 改进DSA热力图可视化：使用多种聚合方式
            # 方法1: 对通道维度求平均（原始方法）
            heatmap_dsa_mean = pfeat_align_reshaped[0].mean(dim=0).detach().cpu().numpy()  # (H, W)
            
            # 方法2: 对通道维度求最大值（突出最强激活）
            heatmap_dsa_max = pfeat_align_reshaped[0].max(dim=0)[0].detach().cpu().numpy()  # (H, W)
            
            # 方法3: 使用L2范数（突出激活强度）
            heatmap_dsa_norm = torch.norm(pfeat_align_reshaped[0], dim=0).detach().cpu().numpy()  # (H, W)
            
            # 选择最直观的可视化方式（可以尝试不同的方法）
            # 使用L2范数通常能更好地展示特征激活
            heatmap_dsa = heatmap_dsa_norm
            
            # 归一化到[0, 1]
            heatmap_dsa = (heatmap_dsa - heatmap_dsa.min()) / (heatmap_dsa.max() - heatmap_dsa.min() + 1e-8)
            
            # 上采样到原图大小
            heatmap_dsa_resized = cv2.resize(heatmap_dsa, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            
            # 应用colormap
            heatmap_dsa_colored = cv2.applyColorMap((heatmap_dsa_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 确保尺寸匹配（复用上面的original_img_resized）
            blended_dsa = cv2.addWeighted(original_img_resized, 0.5, heatmap_dsa_colored, 0.5, 0)
            
            # 创建对比图（包含DSA）
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # 第一行：原始特征
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(blended_part, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Backbone Features (part_features)\n原始特征提取', 
                               fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 第二行：DSA处理后的特征
            axes[1, 0].imshow(cv2.cvtColor(heatmap_dsa_colored, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('DSA Heatmap (pfeat_align)\n注意力热力图', 
                               fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cv2.cvtColor(blended_dsa, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('DSA Features Overlay (pfeat_align)\nDSA对齐后特征叠加', 
                               fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            
            plt.suptitle('Feature Visualization Comparison\n对比热力图可视化', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
        else:
            # 只显示part_features（不应该发生）
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image', fontsize=14)
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(blended_part, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Backbone Features (part_features)', fontsize=14)
            axes[1].axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"对比热力图已保存到: {save_path}")
        
        plt.close()
        
        return blended_part, blended_dsa if len(output) > 2 else None


def process_dataset_images(model, dataset_path, output_dir, device, img_size=384, 
                          num_samples=10, dataset_type='U1652'):
    """
    处理数据集中的图像，生成对比热力图
    
    Args:
        model: 模型
        dataset_path: 数据集路径
        output_dir: 输出目录
        num_samples: 每个类别采样数量
        dataset_type: 数据集类型 ('U1652' 或 'SUES-200')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据数据集类型确定路径
    if dataset_type == 'U1652':
        # University-1652 结构
        query_path = os.path.join(dataset_path, 'test', 'query_drone')
        gallery_path = os.path.join(dataset_path, 'test', 'gallery_satellite')
    elif dataset_type == 'SUES-200':
        # SUES-200 结构（需要指定altitude）
        # 这里假设使用300米高度
        query_path = os.path.join(dataset_path, 'Testing', '300', 'query_drone')
        gallery_path = os.path.join(dataset_path, 'Testing', '300', 'gallery_satellite')
    else:
        # 直接使用提供的路径
        query_path = os.path.join(dataset_path, 'query')
        gallery_path = os.path.join(dataset_path, 'gallery')
    
    print(f"查询图像路径: {query_path}")
    print(f"图库图像路径: {gallery_path}")
    
    # 收集图像对
    image_pairs = []
    
    if os.path.exists(query_path) and os.path.exists(gallery_path):
        # 按类别组织
        query_classes = [d for d in os.listdir(query_path) 
                        if os.path.isdir(os.path.join(query_path, d))]
        
        for class_id in query_classes[:min(10, len(query_classes))]:  # 最多处理10个类别
            query_class_path = os.path.join(query_path, class_id)
            gallery_class_path = os.path.join(gallery_path, class_id)
            
            if not os.path.exists(gallery_class_path):
                continue
            
            query_images = [f for f in os.listdir(query_class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            gallery_images = [f for f in os.listdir(gallery_class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 采样图像对
            num_pairs = min(num_samples, len(query_images), len(gallery_images))
            sampled_query = random.sample(query_images, min(num_pairs, len(query_images)))
            sampled_gallery = random.sample(gallery_images, min(num_pairs, len(gallery_images)))
            
            for q_img, g_img in zip(sampled_query, sampled_gallery):
                image_pairs.append({
                    'query': os.path.join(query_class_path, q_img),
                    'gallery': os.path.join(gallery_class_path, g_img),
                    'class_id': class_id
                })
    else:
        # 如果路径不存在，尝试直接读取图像文件
        print("警告: 标准数据集路径不存在，尝试直接读取图像文件...")
        if os.path.isdir(dataset_path):
            all_images = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append(os.path.join(root, file))
            
            # 随机采样
            sampled_images = random.sample(all_images, min(num_samples * 2, len(all_images)))
            for img_path in sampled_images:
                image_pairs.append({
                    'query': img_path,
                    'gallery': img_path,  # 使用同一张图像
                    'class_id': 'unknown'
                })
    
    print(f"找到 {len(image_pairs)} 个图像对")
    
    # 处理每个图像对
    for idx, pair in enumerate(image_pairs):
        print(f"\n处理图像对 {idx+1}/{len(image_pairs)}: {pair['class_id']}")
        
        # 处理查询图像
        query_save_path = os.path.join(output_dir, f"{pair['class_id']}_query_{idx+1}_comparison.png")
        try:
            generate_comparison_heatmap(
                model, pair['query'], device, img_size, 
                save_path=query_save_path
            )
        except Exception as e:
            print(f"处理查询图像失败: {e}")
        
        # 处理图库图像（如果不同）
        if pair['query'] != pair['gallery']:
            gallery_save_path = os.path.join(output_dir, f"{pair['class_id']}_gallery_{idx+1}_comparison.png")
            try:
                generate_comparison_heatmap(
                    model, pair['gallery'], device, img_size, 
                    save_path=gallery_save_path
                )
            except Exception as e:
                print(f"处理图库图像失败: {e}")
    
    print(f"\n所有对比热力图已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='生成对比热力图可视化')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='数据集根路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重路径（可选）')
    parser.add_argument('--output_dir', type=str, default='./heatmap_visualization',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个类别采样的图像数量')
    parser.add_argument('--dataset_type', type=str, default='U1652',
                       choices=['U1652', 'SUES-200', 'custom'],
                       help='数据集类型')
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
    
    # 加载模型
    model = load_model(args.checkpoint, device, args.num_classes, args.block)
    
    # 处理数据集图像
    process_dataset_images(
        model, args.dataset_path, args.output_dir, device, 
        args.img_size, args.num_samples, args.dataset_type
    )
    
    print("\n完成！")


if __name__ == '__main__':
    main()
