#!/usr/bin/env python3
"""
方案1快速开始示例
最简单的使用方式
"""

import cv2
import numpy as np
from solution1_feature_matching import align_image_with_features

def quick_align(source_path: str, reference_path: str, output_path: str = None):
    """
    快速对齐两张图像
    
    Args:
        source_path: 源图像路径
        reference_path: 参考图像路径
        output_path: 输出图像路径（可选）
    
    Returns:
        aligned_img: 对齐后的图像，如果失败返回None
    """
    print(f"读取源图像: {source_path}")
    source = cv2.imread(source_path)
    if source is None:
        print(f"❌ 无法读取源图像: {source_path}")
        return None
    
    print(f"读取参考图像: {reference_path}")
    reference = cv2.imread(reference_path)
    if reference is None:
        print(f"❌ 无法读取参考图像: {reference_path}")
        return None
    
    print("\n开始对齐...")
    print("使用ORB检测器（快速且免费）...")
    
    # 尝试ORB（最快）
    aligned, transform = align_image_with_features(
        source, reference,
        detector_type='ORB'
    )
    
    # 如果ORB失败，尝试SIFT
    if transform is None:
        print("\nORB失败，尝试SIFT检测器...")
        aligned, transform = align_image_with_features(
            source, reference,
            detector_type='SIFT'
        )
    
    if transform is None:
        print("\n❌ 对齐失败！")
        print("可能的原因:")
        print("  - 两张图像差异太大")
        print("  - 图像中没有足够的特征点")
        print("  - 尝试预处理图像（增强对比度、去噪等）")
        return None
    
    print("✅ 对齐成功！")
    
    if output_path:
        cv2.imwrite(output_path, aligned)
        print(f"结果已保存到: {output_path}")
    
    return aligned


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python quick_start_solution1.py <源图像> <参考图像> [输出图像]")
        print("\n示例:")
        print("  python quick_start_solution1.py source.jpg reference.jpg aligned.jpg")
        sys.exit(1)
    
    source_path = sys.argv[1]
    reference_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'aligned_result.jpg'
    
    result = quick_align(source_path, reference_path, output_path)
    
    if result is not None:
        print("\n显示结果（按任意键关闭）...")
        cv2.imshow('Aligned Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
