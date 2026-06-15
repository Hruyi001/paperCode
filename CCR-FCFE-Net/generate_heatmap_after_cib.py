# -*- coding: utf-8 -*-
"""
批量生成After CIB热力图叠加
将指定输入目录下的所有图像，生成After CIB的热力图叠加并保存到输出目录

使用方法:
python generate_heatmap_after_cib.py \
    --input_dir /path/to/input/images \
    --output_dir /path/to/output/images \
    --name FCFE_Model_University \
    --view_type auto

注意：模型必须包含CIB组件
"""

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import yaml
import cv2
from tqdm import tqdm
from generate_heatmap_ablation import (
    AttentionExtractor,
    generate_heatmap_overlay,
    load_and_preprocess_image
)
from utils import load_network
from datasets.queryDataset import Query_transforms


def determine_view_type(image_path):
    """
    根据图像路径自动判断view_type
    
    Args:
        image_path: 图像路径
    
    Returns:
        'drone' 或 'satellite' 或 None（无法判断）
    """
    path_lower = image_path.lower()
    
    # 检查路径中是否包含关键词
    if 'drone' in path_lower or 'query' in path_lower:
        return 'drone'
    elif 'satellite' in path_lower or 'gallery' in path_lower:
        return 'satellite'
    
    # 检查文件名
    filename = os.path.basename(path_lower)
    if 'drone' in filename:
        return 'drone'
    elif 'satellite' in filename:
        return 'satellite'
    
    return None


def process_single_image(image_path, model, opt, output_dir, view_type='auto', alpha=0.5):
    """
    处理单张图像，生成After CIB的热力图叠加
    
    Args:
        image_path: 输入图像路径
        model: 模型
        opt: 配置参数
        output_dir: 输出目录
        view_type: 'drone', 'satellite', 或 'auto'（自动判断）
        alpha: 热力图透明度（0-1）
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 自动判断view_type
        if view_type == 'auto':
            detected_type = determine_view_type(image_path)
            if detected_type is None:
                print(f"警告: 无法自动判断图像类型，默认使用drone: {image_path}")
                view_type = 'drone'
            else:
                view_type = detected_type
        
        # 确定view_index
        view_index = 3 if view_type == 'drone' else 1
        
        # 加载和预处理图像
        img_tensor, img_array = load_and_preprocess_image(image_path, opt)
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            img_tensor = img_tensor.cuda()
        
        # 提取After CIB的注意力图
        # 使用与 create_ablation_visualization 完全相同的方法
        extractor_after = AttentionExtractor(model)
        extractor_after.register_hooks(hook_before_cib=False)  # Hook到CIB之后（CIB attentions输出）
        
        try:
            attention_after = extractor_after.extract_attention(img_tensor, view_index=view_index, opt=opt)
        except Exception as e:
            print(f"警告: 提取After CIB注意力图时出错: {e}")
            import traceback
            traceback.print_exc()
            attention_after = None
        
        extractor_after.remove_hooks()
        
        if attention_after is None:
            print(f"错误: 无法提取After CIB注意力图: {image_path}")
            return False
        
        # 调整图像大小
        img_resized = cv2.resize(img_array, (opt.w, opt.h))
        
        # 确保图像是BGR格式（cv2使用BGR）
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            # 如果是从PIL转换的RGB，需要转换为BGR
            img_resized_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        else:
            img_resized_bgr = img_resized
        
        # 生成热力图叠加
        # 注意：与 create_ablation_visualization 保持一致
        # jet colormap: 蓝色(低值) -> 红色(高值)
        # 从图像对比看，批处理脚本的结果是红色表示高注意力
        # 如果我的结果是蓝色表示高注意力，说明注意力值需要反转
        # 尝试反转注意力值以匹配批处理脚本的结果
        attention_after_inverted = 1.0 - attention_after
        overlay = generate_heatmap_overlay(img_resized_bgr, attention_after_inverted, alpha=alpha)
        
        # 生成输出文件名
        # 保持原文件名，但添加后缀
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1] or '.png'
        output_filename = f"{name_without_ext}_after_cib{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存图像
        # 将BGR转换为RGB（因为cv2使用BGR，而PIL使用RGB）
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(output_path)
        
        return True
        
    except Exception as e:
        print(f"错误: 处理图像 {image_path} 时失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(input_dir, output_dir, model, opt, view_type='auto', alpha=0.5):
    """
    批量处理输入目录下的所有图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        model: 模型
        opt: 配置参数
        view_type: 'drone', 'satellite', 或 'auto'（自动判断）
        alpha: 热力图透明度（0-1）
    """
    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"错误: 在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    fail_list = []
    
    # 批量处理
    print("\n开始批量处理...")
    for image_path in tqdm(image_files, desc="Processing"):
        # 如果文件已存在，询问是否跳过（这里默认跳过）
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1] or '.png'
        output_filename = f"{name_without_ext}_after_cib{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            # 跳过已存在的文件
            success_count += 1
            continue
        
        # 处理图像
        if process_single_image(image_path, model, opt, output_dir, view_type, alpha):
            success_count += 1
        else:
            fail_count += 1
            fail_list.append(image_path)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    if fail_list:
        print(f"\n失败的图像列表（前10个）:")
        for img in fail_list[:10]:
            print(f"  - {img}")
        if len(fail_list) > 10:
            print(f"  ... 还有 {len(fail_list) - 10} 个失败")


def main():
    parser = argparse.ArgumentParser(description='Generate After CIB heatmap overlay for images')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
    parser.add_argument('--name', default='FCFE_Model_University', type=str, help='model name')
    parser.add_argument('--which_epoch', default='last', type=str, help='model epoch')
    parser.add_argument('--input_dir', type=str, required=True, help='input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output image directory')
    parser.add_argument('--view_type', default='auto', type=str, 
                       choices=['drone', 'satellite', 'auto'],
                       help='image view type: drone, satellite, or auto (auto-detect from path)')
    parser.add_argument('--alpha', default=0.5, type=float, 
                       help='heatmap overlay transparency (0-1, default: 0.5)')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--model', default='convnext_small', type=str, help='model type')
    
    opt = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(opt.input_dir):
        raise FileNotFoundError(f"Input directory not found: {opt.input_dir}")
    
    if opt.alpha < 0 or opt.alpha > 1:
        raise ValueError(f"Alpha must be between 0 and 1, got: {opt.alpha}")
    
    # 加载配置
    yaml.warnings({'YAMLLoadWarning': False})
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        opt.fp16 = config.get('fp16', False)
        opt.views = config.get('views', 2)
        opt.block = config.get('block', 4)
        opt.M = config.get('M', 4)
        opt.share = config.get('share', False)
        if 'h' in config:
            opt.h = config['h']
            opt.w = config['w']
        if 'nclasses' in config:
            opt.nclasses = config['nclasses']
        else:
            opt.nclasses = 729
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using default values...")
        opt.fp16 = False
        opt.views = 2
        opt.block = 4
        opt.M = 4
        opt.share = False
        opt.nclasses = 729
    
    # 设置GPU
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    # 加载模型（模型一定包含CIB）
    print(f"\nLoading model: {opt.name}")
    model, _, _ = load_network(opt.name, opt)
    model.eval()
    
    # 将模型移到GPU（如果可用）
    use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
    if use_gpu:
        model = model.cuda()
        print("Model moved to GPU")
    else:
        print("Using CPU")
    
    # 批量处理
    process_batch(
        opt.input_dir,
        opt.output_dir,
        model,
        opt,
        opt.view_type,
        opt.alpha
    )


if __name__ == '__main__':
    main()
