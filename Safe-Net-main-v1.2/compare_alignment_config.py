#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对比脚本配置文件和便捷运行脚本
可以通过修改配置或命令行参数来指定数据集路径和其他参数
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ============================================
# 配置区域 - 请根据实际情况修改
# ============================================

class Config:
    """配置类，集中管理所有参数"""
    
    # ========== 数据集路径配置 ==========
    DATASET_ROOT = "/root/dataset/University-Release/test"
    # 或者使用相对路径: DATASET_ROOT = "./data/University-Release/test"
    
    # ========== 模型配置 ==========
    MODEL_NAME = "SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
    EPOCH = 119
    GPU_IDS = "0"
    
    # ========== 图像配置 ==========
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    
    # ========== 模式配置 ==========
    # 1: drone->satellite (输入为无人机图)
    # 2: satellite->drone (输入为卫星图)
    MODE = 1
    
    # ========== 输出配置 ==========
    OUTPUT_DIR = "./alignment_comparisons"
    
    # ========== Checkpoint配置 ==========
    # 如果为None，则使用 checkpoints/{MODEL_NAME}
    CHECKPOINT_DIR = None
    
    # ========== 自动选择图像配置 ==========
    # 如果设置了CLASS_ID和VIEW_TYPE，会自动从数据集选择图像
    CLASS_ID = None  # 例如: "0001"
    VIEW_TYPE = None  # "drone" 或 "satellite"
    NUM_IMAGES = 3  # 选择几张图像
    
    # ========== 直接指定图像路径 ==========
    # 如果设置了IMG_PATH，将直接使用该路径（可以是单张或多张，逗号分隔）
    IMG_PATH = None  # 例如: "/path/to/image1.jpg,/path/to/image2.jpg"
    
    # ========== 调试模式 ==========
    DEBUG = False

def get_image_paths_from_dataset(config):
    """从数据集自动获取图像路径"""
    if config.CLASS_ID is None or config.VIEW_TYPE is None:
        return None
    
    dataset_root = Path(config.DATASET_ROOT)
    
    if config.VIEW_TYPE == "drone":
        drone_dir = dataset_root / "query_drone" / config.CLASS_ID
        if not drone_dir.exists():
            raise FileNotFoundError(f"无人机图像目录不存在: {drone_dir}")
        
        # 获取所有图像文件
        img_files = sorted(list(drone_dir.glob("*.jpg")) + 
                          list(drone_dir.glob("*.jpeg")) + 
                          list(drone_dir.glob("*.png")))
        
        if not img_files:
            raise FileNotFoundError(f"在 {drone_dir} 中未找到图像文件")
        
        # 选择前N张
        selected_files = img_files[:config.NUM_IMAGES]
        return ",".join([str(f) for f in selected_files])
    
    elif config.VIEW_TYPE == "satellite":
        satellite_file = dataset_root / "query_satellite" / config.CLASS_ID / f"{config.CLASS_ID}.jpg"
        if not satellite_file.exists():
            raise FileNotFoundError(f"卫星图像不存在: {satellite_file}")
        
        config.MODE = 2  # 卫星图使用mode=2
        return str(satellite_file)
    
    else:
        raise ValueError(f"VIEW_TYPE 必须是 'drone' 或 'satellite'，当前为: {config.VIEW_TYPE}")

def run_comparison(config):
    """运行对比脚本"""
    # 确定图像路径
    if config.IMG_PATH:
        img_path = config.IMG_PATH
    elif config.CLASS_ID and config.VIEW_TYPE:
        img_path = get_image_paths_from_dataset(config)
    else:
        print("错误: 请指定图像路径或设置 CLASS_ID 和 VIEW_TYPE")
        print("\n使用方式:")
        print("  1. 设置 config.IMG_PATH = 'path/to/image.jpg'")
        print("  2. 设置 config.CLASS_ID = '0001' 和 config.VIEW_TYPE = 'drone'")
        print("  3. 使用命令行参数: --img path/to/image.jpg")
        print("  4. 使用命令行参数: --class 0001 --view drone")
        return False
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 生成输出文件名
    if config.CLASS_ID:
        output_file = os.path.join(
            config.OUTPUT_DIR, 
            f"comparison_{config.CLASS_ID}_{config.VIEW_TYPE}.png"
        )
    else:
        output_file = os.path.join(
            config.OUTPUT_DIR,
            "alignment_comparison.png"
        )
    
    # 构建命令
    cmd = [
        "python", "compare_alignment.py",
        "--name", config.MODEL_NAME,
        "--img", img_path,
        "--epoch", str(config.EPOCH),
        "--gpu_ids", config.GPU_IDS,
        "--h", str(config.IMG_HEIGHT),
        "--w", str(config.IMG_WIDTH),
        "--mode", str(config.MODE),
        "--output", output_file,
    ]
    
    if config.CHECKPOINT_DIR:
        cmd.extend(["--checkpoint_dir", config.CHECKPOINT_DIR])
    
    if config.DEBUG:
        cmd.append("--debug")
    
    # 打印配置信息
    print("=" * 60)
    print("配置信息:")
    print(f"  数据集路径: {config.DATASET_ROOT}")
    print(f"  模型名称: {config.MODEL_NAME}")
    print(f"  Epoch: {config.EPOCH}")
    print(f"  GPU IDs: {config.GPU_IDS}")
    print(f"  模式: {config.MODE} ({'drone->satellite' if config.MODE == 1 else 'satellite->drone'})")
    print(f"  图像路径: {img_path}")
    print(f"  输出文件: {output_file}")
    print("=" * 60)
    print()
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ 对比图已保存到: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 生成失败，错误代码: {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n✗ 错误: 找不到 compare_alignment.py 脚本")
        print("请确保在项目根目录运行此脚本")
        return False

def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(
        description='Safe-Net 原图与对齐图对比工具（配置版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 方式1: 使用配置文件中的设置
  python compare_alignment_config.py
  
  # 方式2: 通过命令行指定图像路径
  python compare_alignment_config.py --img path/to/image.jpg
  
  # 方式3: 从数据集自动选择（推荐）
  python compare_alignment_config.py --class 0001 --view drone --num 3
  
  # 方式4: 指定多个图像
  python compare_alignment_config.py --img img1.jpg,img2.jpg,img3.jpg
  
  # 方式5: 修改数据集路径
  python compare_alignment_config.py --dataset /path/to/dataset --class 0001 --view drone
        """
    )
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集根目录路径')
    
    # 图像路径配置
    parser.add_argument('--img', type=str, default=None,
                       help='图像路径（单张或多张，逗号分隔）')
    
    # 自动选择配置
    parser.add_argument('--class', dest='class_id', type=str, default=None,
                       help='类别ID（如 0001）')
    parser.add_argument('--view', type=str, choices=['drone', 'satellite'], default=None,
                       help='视图类型: drone 或 satellite')
    parser.add_argument('--num', type=int, default=None,
                       help='选择几张图像（仅用于drone视图）')
    
    # 模型配置
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称')
    parser.add_argument('--epoch', type=int, default=None,
                       help='模型epoch')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU IDs')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名')
    
    # 其他配置
    parser.add_argument('--mode', type=int, choices=[1, 2], default=None,
                       help='模式: 1=drone->satellite, 2=satellite->drone')
    parser.add_argument('--debug', action='store_true',
                       help='打印详细调试信息')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = Config()
    
    # 应用命令行参数
    if args.dataset:
        config.DATASET_ROOT = args.dataset
    if args.img:
        config.IMG_PATH = args.img
    if args.class_id:
        config.CLASS_ID = args.class_id
    if args.view:
        config.VIEW_TYPE = args.view
    if args.num:
        config.NUM_IMAGES = args.num
    if args.model:
        config.MODEL_NAME = args.model
    if args.epoch:
        config.EPOCH = args.epoch
    if args.gpu:
        config.GPU_IDS = args.gpu
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.mode:
        config.MODE = args.mode
    if args.debug:
        config.DEBUG = True
    
    # 如果指定了output，直接传递给compare_alignment.py
    if args.output:
        # 临时修改输出文件
        import tempfile
        config.OUTPUT_DIR = os.path.dirname(args.output) or "."
        output_file = args.output
    else:
        output_file = None
    
    # 运行对比
    success = run_comparison(config)
    
    if output_file and success:
        # 如果指定了输出文件，重命名
        import shutil
        default_output = os.path.join(config.OUTPUT_DIR, "alignment_comparison.png")
        if os.path.exists(default_output):
            shutil.move(default_output, output_file)
            print(f"✓ 文件已重命名为: {output_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    # 如果直接运行此脚本（不通过命令行），使用配置文件中的设置
    if len(sys.argv) == 1:
        print("=" * 60)
        print("使用配置文件中的设置")
        print("=" * 60)
        print()
        print("提示: 可以通过命令行参数覆盖配置，例如:")
        print("  python compare_alignment_config.py --class 0001 --view drone")
        print()
        print("或者直接修改脚本中的 Config 类")
        print()
        
        config = Config()
        success = run_comparison(config)
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
