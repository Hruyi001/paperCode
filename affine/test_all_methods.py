#!/usr/bin/env python3
"""
测试所有倾斜校正方法并生成结果图
用法: python test_all_methods.py <输入图像> [输出目录]
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime
from typing import Optional
from all_skew_correction_methods import correct_skew_all_methods, METHODS


def create_comparison_image(original: np.ndarray, results: dict, output_path: str):
    """
    创建对比图像，显示原图和所有校正结果
    
    Args:
        original: 原始图像
        results: 所有方法的校正结果
        output_path: 输出路径
    """
    # 计算布局
    num_methods = len(results)
    cols = 3
    rows = (num_methods + 2 + cols - 1) // cols  # +2 for original and title
    
    # 获取图像尺寸
    h, w = original.shape[:2]
    
    # 调整图像大小以便显示（如果太大）
    max_display_size = 600
    if max(h, w) > max_display_size:
        scale = max_display_size / max(h, w)
        display_h, display_w = int(h * scale), int(w * scale)
    else:
        display_h, display_w = h, w
        scale = 1.0
    
    # 创建画布
    canvas_h = rows * (display_h + 60)  # 60 for text
    canvas_w = cols * (display_w + 20)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 绘制原图
    orig_resized = cv2.resize(original, (display_w, display_h))
    y_offset = 40
    x_offset = 10
    canvas[y_offset:y_offset+display_h, x_offset:x_offset+display_w] = orig_resized
    cv2.putText(canvas, 'Original Image', (x_offset, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 绘制所有方法的结果
    idx = 1
    for method_key, (method_name, _) in METHODS.items():
        if method_key not in results:
            continue
        
        corrected, info = results[method_key]
        
        # 计算位置
        row = idx // cols
        col = idx % cols
        
        y = row * (display_h + 60) + 40
        x = col * (display_w + 20) + 10
        
        # 调整校正后的图像大小
        corrected_resized = cv2.resize(corrected, (display_w, display_h))
        
        # 绘制图像
        canvas[y:y+display_h, x:x+display_w] = corrected_resized
        
        # 绘制标题和角度信息
        title = f"{method_name}"
        angle = info.get('detected_angle', 0.0)
        if isinstance(angle, (int, float)):
            title += f" ({angle:.2f}deg)"
        
        cv2.putText(canvas, title, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        idx += 1
    
    # 保存对比图
    cv2.imwrite(output_path, canvas)
    print(f"对比图已保存: {output_path}")


def save_individual_results(original: np.ndarray, results: dict, output_dir: str):
    """
    保存每个方法的单独结果
    
    Args:
        original: 原始图像
        results: 所有方法的校正结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原图
    original_path = os.path.join(output_dir, "00_original.jpg")
    cv2.imwrite(original_path, original)
    print(f"原图已保存: {original_path}")
    
    # 保存每个方法的结果
    for method_key, (method_name, _) in METHODS.items():
        if method_key not in results:
            continue
        
        corrected, info = results[method_key]
        
        # 生成文件名
        method_num = method_key.replace('method', '')
        filename = f"{method_num}_{method_key}_{method_name.replace('法', '').replace('+', '_')}.jpg"
        filename = filename.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, filename)
        
        # 保存图像
        cv2.imwrite(output_path, corrected)
        
        # 打印信息
        angle = info.get('detected_angle', 0.0)
        if isinstance(angle, (int, float)):
            print(f"{method_name}: 角度={angle:.2f}度, 已保存: {output_path}")
        else:
            print(f"{method_name}: 已保存: {output_path}")


def print_summary(results: dict):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)
    
    for method_key, (method_name, _) in METHODS.items():
        if method_key not in results:
            continue
        
        corrected, info = results[method_key]
        
        print(f"\n{method_name} ({method_key}):")
        print(f"  成功: {info.get('success', True)}")
        
        angle = info.get('detected_angle', 0.0)
        if isinstance(angle, (int, float)):
            print(f"  检测角度: {angle:.2f}度")
        
        if 'error' in info:
            print(f"  错误: {info['error']}")
        
        if 'methods_used' in info:
            print(f"  使用的方法: {', '.join(info['methods_used'])}")
            if 'all_angles' in info:
                angles_str = ', '.join([f"{a:.2f}" for a in info['all_angles']])
                print(f"  所有角度: [{angles_str}]")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_all_methods.py <输入图像> [输出目录]")
        print("\n示例:")
        print("  python test_all_methods.py image.jpg")
        print("  python test_all_methods.py image.jpg output_results")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 读取图像
    print(f"读取图像: {input_image_path}")
    if not os.path.exists(input_image_path):
        print(f"错误: 文件不存在: {input_image_path}")
        sys.exit(1)
    
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"错误: 无法读取图像: {input_image_path}")
        sys.exit(1)
    
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # 生成输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        output_dir = f"skew_correction_results_{base_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print()
    
    # 使用所有方法进行校正
    print("开始使用所有方法进行校正...")
    print("-" * 60)
    results = correct_skew_all_methods(image)
    print("-" * 60)
    
    # 打印摘要
    print_summary(results)
    
    # 保存单独的结果
    print("\n保存单独的结果图像...")
    save_individual_results(image, results, output_dir)
    
    # 创建对比图
    print("\n创建对比图...")
    comparison_path = os.path.join(output_dir, "comparison_all_methods.jpg")
    create_comparison_image(image, results, comparison_path)
    
    print("\n✅ 所有结果已保存到:", output_dir)
    print(f"   - 对比图: {comparison_path}")
    print(f"   - 单独结果: {output_dir}/")


if __name__ == '__main__':
    main()
