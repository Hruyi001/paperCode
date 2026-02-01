#!/usr/bin/env python3
"""
简单的使用示例
演示如何使用所有倾斜校正方法
"""

import cv2
import sys
import os

# 导入所有方法
from all_skew_correction_methods import (
    correct_skew_all_methods,
    correct_skew_method1,
    correct_skew_method2,
    correct_skew_method3,
    correct_skew_method4,
    correct_skew_method5,
    correct_skew_method6,
    METHODS
)


def example_single_method():
    """示例1: 使用单个方法"""
    print("=" * 60)
    print("示例1: 使用单个方法（方案1 - 投影轮廓法）")
    print("=" * 60)
    
    # 创建测试图像
    img = create_test_image(angle=5.0)
    
    # 使用方案1
    corrected, info = correct_skew_method1(img)
    
    print(f"检测到的角度: {info.get('detected_angle', 0.0):.2f}度")
    print(f"应用的校正角度: {info.get('corrected_angle', 0.0):.2f}度")
    
    # 保存结果
    cv2.imwrite('example1_result.jpg', corrected)
    print("结果已保存到: example1_result.jpg")


def example_all_methods():
    """示例2: 使用所有方法"""
    print("\n" + "=" * 60)
    print("示例2: 使用所有方法")
    print("=" * 60)
    
    # 创建测试图像
    img = create_test_image(angle=8.0)
    
    # 使用所有方法
    results = correct_skew_all_methods(img)
    
    # 打印结果
    print("\n所有方法的检测结果:")
    for method_key, (method_name, _) in METHODS.items():
        if method_key in results:
            corrected, info = results[method_key]
            angle = info.get('detected_angle', 0.0)
            print(f"  {method_name}: {angle:.2f}度")
            
            # 保存结果
            filename = f'example2_{method_key}.jpg'
            cv2.imwrite(filename, corrected)
            print(f"    已保存: {filename}")


def example_real_image(image_path: str):
    """示例3: 处理真实图像"""
    print("\n" + "=" * 60)
    print(f"示例3: 处理真实图像 - {image_path}")
    print("=" * 60)
    
    # 读取图像
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像: {image_path}")
        return
    
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 使用所有方法
    results = correct_skew_all_methods(img)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"results_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n保存结果到: {output_dir}/")
    for method_key, (method_name, _) in METHODS.items():
        if method_key in results:
            corrected, info = results[method_key]
            angle = info.get('detected_angle', 0.0)
            
            filename = os.path.join(output_dir, f"{method_key}_{method_name}.jpg")
            cv2.imwrite(filename, corrected)
            print(f"  {method_name}: {angle:.2f}度 -> {filename}")


def create_test_image(angle: float = 5.0):
    """创建测试图像（带倾斜的文档）"""
    # 创建白色背景
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制一些文字行（模拟文档）
    for i in range(10):
        y = 50 + i * 55
        cv2.line(img, (50, y), (750, y), (0, 0, 0), 3)
        cv2.putText(img, f"Line {i+1}: This is a test line", (60, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 添加一个矩形框
    cv2.rectangle(img, (40, 40), (760, 560), (0, 0, 0), 2)
    
    # 旋转图像模拟倾斜
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    skewed = cv2.warpAffine(img, M, (new_w, new_h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    
    return skewed


def main():
    """主函数"""
    print("倾斜校正方法使用示例")
    print("=" * 60)
    
    # 示例1: 使用单个方法
    example_single_method()
    
    # 示例2: 使用所有方法
    example_all_methods()
    
    # 示例3: 如果有命令行参数，处理真实图像
    if len(sys.argv) > 1:
        example_real_image(sys.argv[1])
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    import numpy as np
    main()
