# -*- coding: utf-8 -*-
"""
批量生成消融实验的热力图
为数据集中的所有drone图像生成对应的消融热力图

使用方法:
python generate_heatmap_batch.py \
    --drone_dir /datasets/University-Release/test/query_drone \
    --satellite_dir /datasets/University-Release/test/gallery_satellite \
    --name FCFE_Model_University \
    --output_dir ./heatmap_results_batch

注意：模型必须包含CIB组件，通过hook不同位置来展示CIB前后的差异
"""

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import os
import yaml
import glob
from tqdm import tqdm
from generate_heatmap_ablation import (
    AttentionExtractor,
    generate_heatmap_overlay,
    load_and_preprocess_image,
    create_ablation_visualization
)
from utils import load_network


def find_matching_satellite_image(drone_img_path, satellite_dir):
    """
    根据drone图像路径找到对应的satellite图像
    
    Args:
        drone_img_path: drone图像路径，如 /path/to/query_drone/0001/image-06.jpeg
        satellite_dir: satellite图像目录，如 /path/to/gallery_satellite
    
    Returns:
        satellite图像路径，如果找不到返回None
    """
    # 提取地点ID（文件夹名）
    # drone路径格式: .../query_drone/{id}/image-xx.jpeg
    drone_dir = os.path.dirname(drone_img_path)
    location_id = os.path.basename(drone_dir)
    
    # satellite路径格式: .../gallery_satellite/{id}/{id}.jpg
    satellite_path = os.path.join(satellite_dir, location_id, f"{location_id}.jpg")
    
    if os.path.exists(satellite_path):
        return satellite_path
    
    # 如果找不到{id}.jpg，尝试找该目录下的第一个jpg文件
    satellite_folder = os.path.join(satellite_dir, location_id)
    if os.path.exists(satellite_folder):
        jpg_files = glob.glob(os.path.join(satellite_folder, "*.jpg"))
        if jpg_files:
            return jpg_files[0]
        jpeg_files = glob.glob(os.path.join(satellite_folder, "*.jpeg"))
        if jpeg_files:
            return jpeg_files[0]
    
    return None


def process_batch(drone_dir, satellite_dir, model, opt, output_dir):
    """
    批量处理所有图像
    
    Args:
        drone_dir: drone图像目录
        satellite_dir: satellite图像目录
        model_with_cib: 有CIB的模型
        model_without_cib: 无CIB的模型
        opt: 配置参数
        output_dir: 输出目录
    """
    # 获取所有drone图像
    drone_images = []
    for root, dirs, files in os.walk(drone_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                drone_images.append(os.path.join(root, file))
    
    drone_images.sort()
    
    # 如果设置了最大图像数量，限制处理数量
    if hasattr(opt, 'max_images') and opt.max_images is not None:
        drone_images = drone_images[:opt.max_images]
        print(f"\n找到 {len(drone_images)} 张drone图像（限制处理前{opt.max_images}张）")
    else:
        print(f"\n找到 {len(drone_images)} 张drone图像")
    
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    fail_list = []
    
    # 批量处理
    print("\n开始批量处理...")
    for drone_img_path in tqdm(drone_images, desc="Processing"):
        # 找到对应的satellite图像
        satellite_img_path = find_matching_satellite_image(drone_img_path, satellite_dir)
        
        if satellite_img_path is None:
            print(f"\n警告: 找不到对应的satellite图像: {drone_img_path}")
            fail_count += 1
            fail_list.append(drone_img_path)
            continue
        
        # 生成输出文件名
        # 从drone路径提取: query_drone/0001/image-06.jpeg -> 0001_image-06
        # 或者完整路径: /datasets/.../query_drone/0001/image-06.jpeg -> 0001_image-06
        rel_path = os.path.relpath(drone_img_path, drone_dir)
        # rel_path格式: 0001/image-06.jpeg
        location_id = os.path.dirname(rel_path)
        if not location_id or location_id == '.':
            # 如果rel_path没有目录，从完整路径提取
            location_id = os.path.basename(os.path.dirname(drone_img_path))
        image_name = os.path.splitext(os.path.basename(rel_path))[0]
        output_filename = f"{location_id}_{image_name}_ablation.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(output_path):
            print(f"\n跳过已存在的文件: {output_filename}")
            success_count += 1
            continue
        
        try:
            # 为单张图像生成热力图（静默模式，不打印详细信息）
            import sys
            from io import StringIO
            # 临时重定向stdout以减少输出
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            create_ablation_visualization(
                drone_img_path,
                satellite_img_path,
                model,
                opt,
                output_dir=output_dir,
                output_filename=output_filename
            )
            
            sys.stdout = old_stdout
            success_count += 1
        except Exception as e:
            print(f"\n错误: 处理 {drone_img_path} 时失败: {e}")
            fail_count += 1
            fail_list.append(drone_img_path)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    if fail_list:
        print(f"\n失败的图像列表:")
        for img in fail_list[:10]:  # 只显示前10个
            print(f"  - {img}")
        if len(fail_list) > 10:
            print(f"  ... 还有 {len(fail_list) - 10} 个失败")


def main():
    parser = argparse.ArgumentParser(description='Batch generate ablation heatmap visualization')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
    parser.add_argument('--name', default='FCFE_Model_University', type=str, help='model name')
    parser.add_argument('--which_epoch', default='last', type=str, help='model epoch')
    parser.add_argument('--drone_dir', type=str, required=True, 
                       help='directory containing drone images (e.g., /path/to/query_drone)')
    parser.add_argument('--satellite_dir', type=str, required=True,
                       help='directory containing satellite images (e.g., /path/to/gallery_satellite)')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--output_dir', default='./heatmap_results_batch', type=str, help='output directory')
    parser.add_argument('--model', default='convnext_small', type=str, help='model type')
    parser.add_argument('--max_images', type=int, default=None, help='maximum number of images to process (for testing)')
    
    opt = parser.parse_args()
    
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
    
    # 检查目录是否存在
    if not os.path.exists(opt.drone_dir):
        raise FileNotFoundError(f"Drone directory not found: {opt.drone_dir}")
    if not os.path.exists(opt.satellite_dir):
        raise FileNotFoundError(f"Satellite directory not found: {opt.satellite_dir}")
    
    # 批量处理
    process_batch(
        opt.drone_dir,
        opt.satellite_dir,
        model,
        opt,
        opt.output_dir
    )


if __name__ == '__main__':
    main()
