"""
推荐方案：根据场景自动选择或提供综合解决方案
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import sys

# 导入各个方案
try:
    from solution2_edge_detection import correct_skew, detect_skew_angle
    from solution3_manual_points import get_affine_transform_from_points, apply_affine_transform
    from solution4_perspective_correction import correct_perspective, detect_quadrilateral
except ImportError:
    print("请确保所有方案文件都在同一目录下")
    sys.exit(1)


def smart_align_image(
    image: np.ndarray,
    mode: str = 'auto',
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    智能图像对齐函数，根据模式自动选择最佳方案
    
    Args:
        image: 输入图像
        mode: 对齐模式
            - 'auto': 自动检测（推荐）
            - 'skew': 倾斜校正（方案2）
            - 'perspective': 透视校正（方案4）
            - 'manual': 手动指定点（方案3）
        **kwargs: 额外参数
            - source_points: 源点（manual模式需要）
            - target_points: 目标点（manual模式需要）
            - angle: 旋转角度（skew模式可选）
    
    Returns:
        aligned_img: 对齐后的图像
        info: 包含变换信息的字典
    """
    if mode == 'auto':
        # 自动检测模式：先尝试透视校正，再尝试倾斜校正
        try:
            corners = detect_quadrilateral(image)
            if corners is not None and len(corners) == 4:
                aligned_img, transform_matrix = correct_perspective(image, corners=corners)
                return aligned_img, {
                    'method': 'perspective',
                    'transform_matrix': transform_matrix,
                    'corners': corners.tolist()
                }
        except:
            pass
        
        # 如果透视校正失败，尝试倾斜校正
        try:
            angle = detect_skew_angle(image)
            if abs(angle) > 0.5:  # 如果检测到明显倾斜
                aligned_img, used_angle = correct_skew(image, angle=angle)
                return aligned_img, {
                    'method': 'skew',
                    'angle': used_angle,
                    'transform_matrix': None
                }
        except:
            pass
        
        # 如果都失败，返回原图
        return image.copy(), {'method': 'none', 'message': '无法自动检测对齐方式'}
    
    elif mode == 'skew':
        angle = kwargs.get('angle', None)
        aligned_img, used_angle = correct_skew(image, angle=angle, auto_detect=(angle is None))
        return aligned_img, {
            'method': 'skew',
            'angle': used_angle
        }
    
    elif mode == 'perspective':
        corners = kwargs.get('corners', None)
        aligned_img, transform_matrix = correct_perspective(
            image, 
            corners=corners, 
            auto_detect=(corners is None)
        )
        return aligned_img, {
            'method': 'perspective',
            'transform_matrix': transform_matrix,
            'corners': corners.tolist() if corners is not None else None
        }
    
    elif mode == 'manual':
        source_points = kwargs.get('source_points')
        target_points = kwargs.get('target_points')
        if source_points is None or target_points is None:
            raise ValueError("manual模式需要提供source_points和target_points")
        
        transform_matrix = get_affine_transform_from_points(source_points, target_points)
        output_size = kwargs.get('output_size', None)
        aligned_img = apply_affine_transform(image, transform_matrix, output_size)
        
        return aligned_img, {
            'method': 'manual',
            'transform_matrix': transform_matrix.tolist(),
            'source_points': source_points,
            'target_points': target_points
        }
    
    else:
        raise ValueError(f"未知的模式: {mode}")


def quick_skew_correction(image: np.ndarray) -> np.ndarray:
    """
    快速倾斜校正（最常用的场景）
    这是最简单、最实用的方案
    
    Args:
        image: 输入图像
    
    Returns:
        corrected_img: 校正后的图像
    """
    corrected, _ = correct_skew(image, auto_detect=True)
    return corrected


def quick_perspective_correction(image: np.ndarray) -> np.ndarray:
    """
    快速透视校正（文档扫描常用）
    
    Args:
        image: 输入图像
    
    Returns:
        corrected_img: 校正后的图像
    """
    corrected, _ = correct_perspective(image, auto_detect=True)
    return corrected


def align_with_known_angle(
    image: np.ndarray,
    angle: float,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0
) -> np.ndarray:
    """
    已知角度和参数的快速对齐（最灵活）
    
    Args:
        image: 输入图像
        angle: 旋转角度（度）
        scale: 缩放因子
        tx: x方向平移
        ty: y方向平移
    
    Returns:
        aligned_img: 对齐后的图像
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # 构建仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # 应用变换
    aligned = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return aligned


def demo():
    """演示不同场景的使用"""
    print("=" * 60)
    print("推荐使用方案")
    print("=" * 60)
    print()
    
    print("【场景1: 文档扫描倾斜校正】")
    print("推荐: solution2_edge_detection.py 或 quick_skew_correction()")
    print("代码示例:")
    print("  from recommended_solution import quick_skew_correction")
    print("  corrected = quick_skew_correction(image)")
    print()
    
    print("【场景2: 已知变换参数】")
    print("推荐: align_with_known_angle() 或 solution3_manual_points.py")
    print("代码示例:")
    print("  from recommended_solution import align_with_known_angle")
    print("  aligned = align_with_known_angle(image, angle=15, scale=1.1)")
    print()
    
    print("【场景3: 文档透视校正】")
    print("推荐: solution4_perspective_correction.py 或 quick_perspective_correction()")
    print("代码示例:")
    print("  from recommended_solution import quick_perspective_correction")
    print("  corrected = quick_perspective_correction(image)")
    print()
    
    print("【场景4: 不确定场景，想自动检测】")
    print("推荐: smart_align_image(mode='auto')")
    print("代码示例:")
    print("  from recommended_solution import smart_align_image")
    print("  aligned, info = smart_align_image(image, mode='auto')")
    print("  print(f'使用的方案: {info[\"method\"]}')")
    print()
    
    # 实际示例
    print("\n" + "=" * 60)
    print("实际测试")
    print("=" * 60)
    
    # 创建测试图像
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 2)
    
    # 测试倾斜校正
    center = (300, 200)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    skewed = cv2.warpAffine(img, M, (600, 400))
    
    print("\n1. 测试自动倾斜检测和校正...")
    corrected, info = smart_align_image(skewed, mode='skew')
    print(f"   检测到的倾斜角度: {info['angle']:.2f}度")
    
    print("\n2. 测试已知角度的快速校正...")
    test_img = align_with_known_angle(skewed, angle=-5)
    print("   校正完成")
    
    print("\n✅ 推荐方案总结:")
    print("   - 最简单: quick_skew_correction() - 一键倾斜校正")
    print("   - 最灵活: align_with_known_angle() - 已知参数时使用")
    print("   - 最智能: smart_align_image(mode='auto') - 自动选择方案")


if __name__ == '__main__':
    demo()
