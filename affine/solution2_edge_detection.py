"""
方案2: 基于边缘检测的倾斜校正
使用Canny边缘检测和霍夫变换检测直线，计算倾斜角度进行旋转校正
适用于文档扫描等场景
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def detect_skew_angle(image: np.ndarray) -> float:
    """
    检测图像的倾斜角度
    
    Args:
        image: 输入图像（灰度图或彩色图）
    
    Returns:
        angle: 倾斜角度（度）
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # 计算所有直线的角度
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        # 只考虑接近水平的直线（-45到45度）
        if -45 <= angle <= 45:
            angles.append(angle)
    
    if len(angles) == 0:
        return 0.0
    
    # 使用中位数作为倾斜角度（更鲁棒）
    median_angle = np.median(angles)
    
    return median_angle


def correct_skew(
    image: np.ndarray,
    angle: Optional[float] = None,
    auto_detect: bool = True
) -> Tuple[np.ndarray, float]:
    """
    校正图像的倾斜
    
    Args:
        image: 输入图像
        angle: 旋转角度（度），如果为None则自动检测
        auto_detect: 是否自动检测倾斜角度
    
    Returns:
        corrected_img: 校正后的图像
        used_angle: 使用的旋转角度
    """
    if angle is None and auto_detect:
        angle = detect_skew_angle(image)
    
    if angle is None:
        angle = 0.0
    
    # 如果角度很小，不需要校正
    if abs(angle) < 0.5:
        return image.copy(), 0.0
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的图像尺寸
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转中心
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # 应用旋转
    corrected_img = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return corrected_img, angle


def align_with_affine_transform(
    image: np.ndarray,
    angle: float,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0
) -> np.ndarray:
    """
    使用仿射变换对齐图像（支持旋转、缩放、平移）
    
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
    # 先旋转
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 添加平移
    transform_matrix = rotation_matrix.copy()
    transform_matrix[0, 2] += tx
    transform_matrix[1, 2] += ty
    
    # 应用变换
    aligned_img = cv2.warpAffine(
        image, transform_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return aligned_img


def demo():
    """演示函数"""
    # 创建示例图像（带倾斜的文本）
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制一些文本线条
    for i in range(10):
        y = 50 + i * 50
        cv2.line(img, (50, y), (750, y), (0, 0, 0), 2)
    
    # 旋转图像模拟倾斜
    center = (400, 300)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    skewed_img = cv2.warpAffine(img, M, (800, 600))
    
    # 自动检测并校正
    corrected, angle = correct_skew(skewed_img, auto_detect=True)
    
    print(f"Detected skew angle: {angle:.2f} degrees")
    
    # 显示结果
    cv2.imshow('Original', img)
    cv2.imshow('Skewed', skewed_img)
    cv2.imshow('Corrected', corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
