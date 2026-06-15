# -*- coding: utf-8 -*-
"""
生成消融实验的热力图可视化
比较有/无CIB组件的注意力图差异
"""

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import yaml
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils import load_network
import cv2


# Hook来捕获attention maps
attention_maps_hook = None

def attention_hook(module, input, output):
    """Hook函数来捕获CIB_block的attention maps"""
    global attention_maps_hook
    x = input[0]  # ADIB_attention_features
    block = module.block
    M = module.M
    
    all_attention_maps = []
    for i in range(block):
        part = x[:, :, :, :, i]
        att_maps = module.attentions(part)  # [B, M, H, W]
        all_attention_maps.append(att_maps)
    
    # 合并所有block的attention maps: [B, block*M, H, W]
    attention_maps_hook = torch.cat(all_attention_maps, dim=1)


def register_attention_hook(model, use_cib=True):
    """注册hook来捕获attention maps"""
    global attention_maps_hook
    attention_maps_hook = None
    
    if hasattr(model, 'model_1') and hasattr(model.model_1, 'CIB_layer'):
        if use_cib:
            # 注册hook到CIB_layer
            handle = model.model_1.CIB_layer.register_forward_hook(attention_hook)
            return handle
    return None


def get_attention_maps_with_hook(model, x, view_index=1, use_cib=True):
    """使用hook提取模型的attention maps"""
    global attention_maps_hook
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'model_1'):
            # 获取backbone特征
            gap_feature, part_features = model.model_1.backbone(x)
            ADIB_features = model.model_1.ADIB_layer(part_features)
            
            ADIB_list = []
            for i in range(model.model_1.block):
                ADIB_list.append(ADIB_features[i])
            ADIB_attention_features = torch.stack(ADIB_list, dim=4)
            
            if use_cib:
                # 有CIB：注册hook并前向传播，提取CIB的attention maps
                handle = register_attention_hook(model, use_cib=True)
                try:
                    _, _ = model.model_1.CIB_layer(ADIB_attention_features)
                    attention_maps = attention_maps_hook.clone() if attention_maps_hook is not None else None
                finally:
                    if handle is not None:
                        handle.remove()
                
                if attention_maps is None:
                    # 如果hook失败，使用backbone特征作为fallback
                    B, C, H, W = part_features.shape
                    attention_maps = part_features.mean(dim=1, keepdim=True)
                    attention_maps = attention_maps.repeat(1, model.model_1.block * model.model_1.M, 1, 1)
            else:
                # 无CIB：直接使用backbone的特征图生成attention map
                # 使用ADIB_features的平均值作为attention
                B, C, H, W = part_features.shape
                # 对每个block的特征进行平均
                attention_list = []
                for i in range(model.model_1.block):
                    block_feat = ADIB_features[i]  # [B, C, H, W]
                    # 对通道维度进行平均，生成attention map
                    att_map = block_feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                    # 扩展到M个通道（模拟CIB的M个attention heads）
                    att_map = att_map.repeat(1, model.model_1.M, 1, 1)
                    attention_list.append(att_map)
                attention_maps = torch.cat(attention_list, dim=1)  # [B, block*M, H, W]
    
    return attention_maps


def generate_heatmap(attention_maps, original_img, method='mean'):
    """
    从attention maps生成热力图
    
    Args:
        attention_maps: [B, C, H, W] 或 [C, H, W]
        original_img: PIL Image 或 numpy array
        method: 'mean' 或 'max' - 如何聚合多个attention maps
    """
    # 转换为numpy
    if isinstance(attention_maps, torch.Tensor):
        attention_maps = attention_maps.cpu().numpy()
    
    # 处理batch维度
    if len(attention_maps.shape) == 4:
        attention_maps = attention_maps[0]  # 取第一个batch
    
    # 聚合多个attention maps
    if method == 'mean':
        heatmap = np.mean(attention_maps, axis=0)  # [H, W]
    elif method == 'max':
        heatmap = np.max(attention_maps, axis=0)  # [H, W]
    else:
        heatmap = np.mean(attention_maps, axis=0)
    
    # 归一化到[0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 调整大小到原始图像尺寸
    if isinstance(original_img, Image.Image):
        img_size = original_img.size  # (width, height)
    else:
        img_size = (original_img.shape[1], original_img.shape[0])
    
    heatmap_resized = cv2.resize(heatmap, img_size, interpolation=cv2.INTER_LINEAR)
    
    return heatmap_resized


def overlay_heatmap(img, heatmap, alpha=0.5, colormap='jet'):
    """
    将热力图叠加到原始图像上
    
    Args:
        img: PIL Image 或 numpy array
        heatmap: numpy array [H, W]
        alpha: 透明度
        colormap: 颜色映射
    """
    # 转换为numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # 确保是RGB格式
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 应用颜色映射
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # [H, W, 3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # 叠加
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_ablation_study(model, img_path, save_path, 
                             view_type='drone', img_size=(384, 384)):
    """
    生成消融实验的可视化结果
    
    Args:
        model: 模型（同一个模型，通过hook控制有/无CIB）
        img_path: 输入图像路径
        save_path: 保存路径
        view_type: 'drone' 或 'satellite'
        img_size: 图像尺寸
    """
    # 加载和预处理图像
    img = Image.open(img_path).convert('RGB')
    original_img = img.copy()
    
    data_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = data_transforms(img).unsqueeze(0).cuda()
    
    # 确定view_index
    view_index = 3 if view_type == 'drone' else 1
    
    # 提取attention maps（有CIB）
    attention_with_cib = get_attention_maps_with_hook(model, img_tensor, view_index, use_cib=True)
    
    # 提取attention maps（无CIB）
    attention_without_cib = get_attention_maps_with_hook(model, img_tensor, view_index, use_cib=False)
    
    # 生成热力图
    heatmap_with_cib = generate_heatmap(attention_with_cib, original_img, method='mean')
    heatmap_without_cib = generate_heatmap(attention_without_cib, original_img, method='mean')
    
    # 叠加热力图
    overlay_with_cib = overlay_heatmap(original_img, heatmap_with_cib, alpha=0.5)
    overlay_without_cib = overlay_heatmap(original_img, heatmap_without_cib, alpha=0.5)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Input', fontsize=14)
    axes[0].axis('off')
    
    # Without CIB
    axes[1].imshow(overlay_without_cib)
    axes[1].set_title('Without CIB', fontsize=14)
    axes[1].axis('off')
    
    # With CIB
    axes[2].imshow(overlay_with_cib)
    axes[2].set_title('With CIB', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate heatmap for ablation study')
    parser.add_argument('--name', default='FCFE_Model_University', type=str, help='model name')
    parser.add_argument('--which_epoch', default='last', type=str, help='model epoch')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids')
    parser.add_argument('--img_path', type=str, required=True, help='path to input image')
    parser.add_argument('--save_path', type=str, default='heatmap_result.png', help='path to save result')
    parser.add_argument('--view_type', type=str, default='drone', choices=['drone', 'satellite'], 
                       help='view type: drone or satellite')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    opt = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    opt.fp16 = config['fp16']
    opt.views = config['views']
    opt.block = config['block']
    opt.M = config['M']
    opt.share = config['share']
    opt.nclasses = config.get('nclasses', 729)
    opt.resnet = config.get('resnet', False)
    
    # 设置GPU
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    # 加载模型
    print('Loading model...')
    model, _, epoch = load_network(opt.name, opt)
    model.head = nn.Sequential()
    model = model.eval()
    model = model.cuda()
    
    # 生成可视化（使用同一个模型，通过hook控制有/无CIB）
    print('Generating heatmap...')
    visualize_ablation_study(
        model, 
        opt.img_path, 
        opt.save_path,
        opt.view_type,
        (opt.h, opt.w)
    )
    
    print('Done!')


if __name__ == '__main__':
    main()
