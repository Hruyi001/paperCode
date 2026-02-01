"""
方案5: 使用scikit-image的仿射变换
使用scikit-image的transform模块，提供更高级的变换接口
支持多种插值方法
"""

import numpy as np
from skimage import transform as tf
from skimage import io, img_as_float
import cv2
from typing import Tuple, Optional


def align_with_skimage_affine(
    image: np.ndarray,
    transform_params: dict,
    output_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, tf.AffineTransform]:
    """
    使用scikit-image进行仿射变换
    
    Args:
        image: 输入图像
        transform_params: 变换参数字典，包含：
            - 'rotation': 旋转角度（度）
            - 'translation': (tx, ty) 平移
            - 'scale': 缩放因子或(sx, sy)
            - 'shear': 剪切角度（度）
        output_shape: 输出图像形状 (height, width)
    
    Returns:
        transformed_img: 变换后的图像
        transform: AffineTransform对象
    """
    # 转换为浮点数格式（scikit-image推荐）
    if image.dtype != np.float64:
        image_float = img_as_float(image)
    else:
        image_float = image
    
    # 创建仿射变换对象
    transform = tf.AffineTransform()
    
    # 应用变换参数
    if 'rotation' in transform_params:
        angle_rad = np.deg2rad(transform_params['rotation'])
        transform = transform + tf.AffineTransform(rotation=angle_rad)
    
    if 'translation' in transform_params:
        tx, ty = transform_params['translation']
        transform = transform + tf.AffineTransform(translation=(tx, ty))
    
    if 'scale' in transform_params:
        scale = transform_params['scale']
        if isinstance(scale, (tuple, list)):
            sx, sy = scale
        else:
            sx = sy = scale
        transform = transform + tf.AffineTransform(scale=(sx, sy))
    
    if 'shear' in transform_params:
        shear_rad = np.deg2rad(transform_params['shear'])
        transform = transform + tf.AffineTransform(shear=shear_rad)
    
    # 确定输出形状
    if output_shape is None:
        output_shape = image.shape[:2]
    
    # 应用变换
    transformed_img = tf.warp(
        image_float,
        transform,
        output_shape=output_shape,
        order=3,  # 双三次插值
        mode='constant',
        cval=1.0 if image_float.max() <= 1.0 else 255.0
    )
    
    # 转换回原始数据类型
    if image.dtype == np.uint8:
        transformed_img = (transformed_img * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        transformed_img = (transformed_img * 65535).astype(np.uint16)
    
    return transformed_img, transform


def align_from_matrix(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    从变换矩阵应用仿射变换
    
    Args:
        image: 输入图像
        transform_matrix: 2x3或3x3变换矩阵
        output_shape: 输出图像形状 (height, width)
    
    Returns:
        transformed_img: 变换后的图像
    """
    # 转换为浮点数
    if image.dtype != np.float64:
        image_float = img_as_float(image)
    else:
        image_float = image
    
    # 如果是2x3矩阵，转换为3x3
    if transform_matrix.shape == (2, 3):
        matrix_3x3 = np.vstack([transform_matrix, [0, 0, 1]])
    else:
        matrix_3x3 = transform_matrix
    
    # 创建AffineTransform对象
    transform = tf.AffineTransform(matrix=matrix_3x3)
    
    # 确定输出形状
    if output_shape is None:
        output_shape = image.shape[:2]
    
    # 应用变换
    transformed_img = tf.warp(
        image_float,
        transform,
        output_shape=output_shape,
        order=3,
        mode='constant',
        cval=1.0 if image_float.max() <= 1.0 else 255.0
    )
    
    # 转换回原始数据类型
    if image.dtype == np.uint8:
        transformed_img = (transformed_img * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        transformed_img = (transformed_img * 65535).astype(np.uint16)
    
    return transformed_img


def estimate_affine_transform_skimage(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> tf.AffineTransform:
    """
    从对应点估计仿射变换
    
    Args:
        src_points: 源点坐标，形状为(N, 2)
        dst_points: 目标点坐标，形状为(N, 2)
    
    Returns:
        transform: AffineTransform对象
    """
    transform = tf.AffineTransform()
    transform.estimate(src_points, dst_points)
    return transform


def demo():
    """演示函数"""
    # 创建示例图像
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 0), 2)
    cv2.circle(img, (200, 200), 50, (255, 0, 0), -1)
    
    # 方法1: 使用参数字典
    transform_params = {
        'rotation': 15,
        'translation': (20, 10),
        'scale': 1.1
    }
    
    transformed1, transform1 = align_with_skimage_affine(
        img, transform_params, output_shape=(400, 400)
    )
    
    print("Transform matrix (from params):")
    print(transform1.params)
    
    # 方法2: 从OpenCV矩阵转换
    center = (200, 200)
    M_cv = cv2.getRotationMatrix2D(center, 15, 1.0)
    transformed2 = align_from_matrix(img, M_cv, output_shape=(400, 400))
    
    # 方法3: 从对应点估计
    src_pts = np.array([[100, 100], [300, 100], [100, 300]], dtype=np.float32)
    dst_pts = np.array([[120, 80], [320, 120], [80, 320]], dtype=np.float32)
    transform3 = estimate_affine_transform_skimage(src_pts, dst_pts)
    transformed3 = align_from_matrix(img, transform3.params, output_shape=(400, 400))
    
    print("\nTransform matrix (from points):")
    print(transform3.params)
    
    # 显示结果
    cv2.imshow('Original', img)
    cv2.imshow('Transformed (params)', transformed1)
    cv2.imshow('Transformed (matrix)', transformed2)
    cv2.imshow('Transformed (points)', transformed3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
