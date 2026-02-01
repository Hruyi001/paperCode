"""
方案4: 基于投影变换的透视校正
检测文档/矩形的四个角点，使用透视变换进行校正
适用于文档扫描等场景
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def detect_quadrilateral(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测图像中的四边形（如文档边界）
    
    Args:
        image: 输入图像（灰度图或彩色图）
    
    Returns:
        corners: 四个角点的坐标，形状为(4, 2)，如果未检测到返回None
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算轮廓的近似多边形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 如果近似多边形有4个顶点，返回它们
    if len(approx) == 4:
        corners = approx.reshape(4, 2)
        return corners
    
    # 如果近似多边形不是4个顶点，尝试找到最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    corners = np.int0(box)
    
    return corners.astype(np.float32)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    对四个点进行排序：左上、右上、右下、左下
    
    Args:
        pts: 四个点的坐标，形状为(4, 2)
    
    Returns:
        ordered_pts: 排序后的点
    """
    # 初始化排序后的点
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 左上角点有最小的x+y和
    # 右下角点有最大的x+y和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 右上角点有最小的x-y差
    # 左下角点有最大的x-y差
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def get_perspective_transform(
    src_points: np.ndarray,
    dst_points: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算透视变换矩阵
    
    Args:
        src_points: 源图像中的四个角点，形状为(4, 2)
        dst_points: 目标图像中的四个角点，如果为None则自动计算
        width: 输出图像宽度
        height: 输出图像高度
    
    Returns:
        transform_matrix: 3x3透视变换矩阵
        dst_points: 目标点坐标
    """
    # 对源点进行排序
    src_ordered = order_points(src_points)
    
    # 如果没有提供目标点，根据源点计算
    if dst_points is None:
        if width is None or height is None:
            # 计算源四边形的宽度和高度
            (tl, tr, br, bl) = src_ordered
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            width = maxWidth
            height = maxHeight
        
        # 创建目标点（矩形）
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
    else:
        dst_points = order_points(dst_points)
    
    # 计算透视变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(src_ordered, dst_points)
    
    return transform_matrix, dst_points


def correct_perspective(
    image: np.ndarray,
    corners: Optional[np.ndarray] = None,
    auto_detect: bool = True,
    output_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    校正图像的透视变形
    
    Args:
        image: 输入图像
        corners: 四个角点坐标，如果为None则自动检测
        auto_detect: 是否自动检测角点
        output_size: 输出图像尺寸 (width, height)
    
    Returns:
        corrected_img: 校正后的图像
        transform_matrix: 3x3透视变换矩阵
    """
    if corners is None and auto_detect:
        corners = detect_quadrilateral(image)
    
    if corners is None:
        raise ValueError("Could not detect quadrilateral corners")
    
    # 计算透视变换
    transform_matrix, dst_points = get_perspective_transform(
        corners,
        width=output_size[0] if output_size else None,
        height=output_size[1] if output_size else None
    )
    
    # 确定输出尺寸
    if output_size is None:
        width = int(dst_points[1, 0] + 1)
        height = int(dst_points[2, 1] + 1)
    else:
        width, height = output_size
    
    # 应用透视变换
    corrected_img = cv2.warpPerspective(
        image, transform_matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return corrected_img, transform_matrix


def demo():
    """演示函数"""
    # 创建示例图像（模拟倾斜的文档）
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制一个矩形（模拟文档）
    cv2.rectangle(img, (100, 100), (700, 500), (0, 0, 0), 2)
    
    # 添加一些文本线条
    for i in range(5):
        y = 150 + i * 70
        cv2.line(img, (150, y), (650, y), (0, 0, 0), 2)
    
    # 创建透视变形（模拟倾斜拍摄）
    src_pts = np.float32([[100, 100], [700, 120], [680, 500], [120, 480]])
    dst_pts = np.float32([[0, 0], [600, 0], [600, 400], [0, 400]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    skewed_img = cv2.warpPerspective(img, M, (600, 400))
    
    # 自动检测并校正
    corrected, transform = correct_perspective(skewed_img, auto_detect=True)
    
    print("Perspective transform matrix:")
    print(transform)
    
    # 显示结果
    cv2.imshow('Original', img)
    cv2.imshow('Skewed', skewed_img)
    cv2.imshow('Corrected', corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()
