"""
方案3: 手动指定控制点的仿射变换
手动选择源图像和目标图像的对应点，计算仿射变换矩阵
适用于已知对应点的情况
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_affine_transform_from_points(
    src_points: List[Tuple[float, float]],
    dst_points: List[Tuple[float, float]]
) -> Optional[np.ndarray]:
    """
    从对应点计算仿射变换矩阵
    
    Args:
        src_points: 源图像中的点列表，至少需要3个点
        dst_points: 目标图像中的对应点列表，至少需要3个点
    
    Returns:
        transform_matrix: 2x3仿射变换矩阵，如果失败返回None
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError("At least 3 point pairs are required for affine transformation")
    
    if len(src_points) != len(dst_points):
        raise ValueError("Number of source and destination points must match")
    
    # 转换为numpy数组
    src_pts = np.float32(src_points).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_points).reshape(-1, 1, 2)
    
    # 如果只有3个点，使用精确计算
    if len(src_points) == 3:
        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    else:
        # 如果有更多点，使用最小二乘法
        transform_matrix, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
    
    return transform_matrix


def apply_affine_transform(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    应用仿射变换到图像
    
    Args:
        image: 输入图像
        transform_matrix: 2x3仿射变换矩阵
        output_size: 输出图像尺寸 (width, height)，如果为None则使用输入图像尺寸
    
    Returns:
        transformed_img: 变换后的图像
    """
    if output_size is None:
        h, w = image.shape[:2]
        output_size = (w, h)
    
    transformed_img = cv2.warpAffine(
        image, transform_matrix, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return transformed_img


def align_image_manual(
    source_img: np.ndarray,
    source_points: List[Tuple[float, float]],
    target_points: List[Tuple[float, float]],
    output_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    手动指定点对齐图像
    
    Args:
        source_img: 源图像
        source_points: 源图像中的点坐标列表
        target_points: 目标图像中的对应点坐标列表
        output_size: 输出图像尺寸
    
    Returns:
        aligned_img: 对齐后的图像
        transform_matrix: 2x3仿射变换矩阵
    """
    transform_matrix = get_affine_transform_from_points(source_points, target_points)
    
    if transform_matrix is None:
        raise ValueError("Failed to compute affine transformation")
    
    aligned_img = apply_affine_transform(source_img, transform_matrix, output_size)
    
    return aligned_img, transform_matrix


def create_affine_transform(
    angle: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    创建自定义的仿射变换矩阵
    
    Args:
        angle: 旋转角度（度）
        scale: 缩放因子
        tx: x方向平移
        ty: y方向平移
        center: 旋转中心，如果为None则使用(0, 0)
    
    Returns:
        transform_matrix: 2x3仿射变换矩阵
    """
    if center is None:
        center = (0, 0)
    
    # 创建旋转和缩放矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 添加平移
    transform_matrix = rotation_matrix.copy()
    transform_matrix[0, 2] += tx
    transform_matrix[1, 2] += ty
    
    return transform_matrix


def demo():
    """演示函数"""
    # 创建示例图像
    source = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(source, (100, 100), (300, 300), (0, 0, 0), 2)
    cv2.circle(source, (200, 200), 50, (255, 0, 0), -1)
    
    # 定义源图像中的三个点（矩形的三个角）
    src_points = [
        (100, 100),  # 左上角
        (300, 100),  # 右上角
        (100, 300),  # 左下角
    ]
    
    # 定义目标位置（旋转和平移后的位置）
    dst_points = [
        (150, 50),   # 左上角的新位置
        (350, 100),  # 右上角的新位置
        (50, 250),   # 左下角的新位置
    ]
    
    # 计算变换矩阵
    transform = get_affine_transform_from_points(src_points, dst_points)
    
    # 应用变换
    aligned = apply_affine_transform(source, transform, (400, 400))
    
    # 可视化点
    vis_source = source.copy()
    for pt in src_points:
        cv2.circle(vis_source, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    
    vis_aligned = aligned.copy()
    for pt in dst_points:
        cv2.circle(vis_aligned, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    
    print("Transform matrix:")
    print(transform)
    
    # 显示结果
    cv2.imshow('Source with points', vis_source)
    cv2.imshow('Aligned with points', vis_aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
