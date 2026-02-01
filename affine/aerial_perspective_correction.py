"""
航拍图像透视校正
专门针对无人机/航拍图像的透视校正
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from solution4_perspective_correction import correct_perspective, order_points


def detect_building_region(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测航拍图像中的主要建筑物区域（用于透视校正）
    
    策略：
    1. 检测最大的矩形轮廓（可能是建筑物）
    2. 或检测多个建筑物，选择最大的
    3. 或使用边缘检测找到主要结构
    
    Args:
        image: 输入航拍图像
    
    Returns:
        corners: 四个角点坐标，形状为(4, 2)，如果未检测到返回None
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 边缘检测（使用自适应阈值）
    edges = cv2.Canny(blurred, 50, 150)
    
    # 形态学操作，连接断开的边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 按面积排序，选择最大的几个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # 尝试找到四边形
    for contour in contours:
        # 计算轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果近似多边形有4个顶点，返回它们
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            # 检查是否合理（面积不能太小）
            area = cv2.contourArea(corners)
            if area > image.shape[0] * image.shape[1] * 0.1:  # 至少占图像10%
                return corners
    
    # 如果没找到4边形，尝试使用最小外接矩形
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > image.shape[0] * image.shape[1] * 0.1:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = box.astype(np.float32)
            return corners
    
    return None


def detect_ground_plane(image: np.ndarray) -> Optional[np.ndarray]:
    """
    检测地面平面区域（用于透视校正）
    
    适用于有明显地面区域（如道路、广场）的图像
    
    Args:
        image: 输入航拍图像
    
    Returns:
        corners: 四个角点坐标，如果未检测到返回None
    """
    # 转换为HSV，检测地面颜色（通常是灰色、棕色等）
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        return None
    
    # 检测地面颜色范围（灰色、棕色等）
    # 这些值可能需要根据实际图像调整
    lower_ground = np.array([0, 0, 50])
    upper_ground = np.array([180, 50, 200])
    
    mask = cv2.inRange(hsv, lower_ground, upper_ground)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算近似多边形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    
    return None


def correct_aerial_perspective(
    image: np.ndarray,
    method: str = 'auto',
    corners: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, dict]:
    """
    校正航拍图像的透视变形
    
    Args:
        image: 输入航拍图像
        method: 检测方法
            - 'auto': 自动选择（先尝试建筑物，再尝试地面）
            - 'building': 检测建筑物区域
            - 'ground': 检测地面区域
            - 'manual': 使用手动提供的角点
        corners: 手动指定的四个角点（method='manual'时使用）
    
    Returns:
        corrected: 校正后的图像
        info: 包含校正信息的字典
    """
    info = {
        'method_used': method,
        'success': False
    }
    
    if method == 'manual':
        if corners is None or len(corners) != 4:
            raise ValueError("manual模式需要提供4个角点")
        detected_corners = corners
        info['detection_method'] = 'manual'
    elif method == 'building':
        detected_corners = detect_building_region(image)
        info['detection_method'] = 'building'
    elif method == 'ground':
        detected_corners = detect_ground_plane(image)
        info['detection_method'] = 'ground'
    else:  # auto
        # 先尝试建筑物
        detected_corners = detect_building_region(image)
        if detected_corners is not None:
            info['detection_method'] = 'building'
        else:
            # 再尝试地面
            detected_corners = detect_ground_plane(image)
            if detected_corners is not None:
                info['detection_method'] = 'ground'
            else:
                info['detection_method'] = 'failed'
    
    if detected_corners is None:
        info['success'] = False
        info['message'] = '无法自动检测到合适的区域，请尝试手动指定角点'
        return image.copy(), info
    
    # 应用透视校正
    try:
        corrected, transform_matrix = correct_perspective(
            image,
            corners=detected_corners,
            auto_detect=False
        )
        info['success'] = True
        info['corners'] = detected_corners.tolist()
        info['transform_matrix'] = transform_matrix.tolist()
        return corrected, info
    except Exception as e:
        info['success'] = False
        info['error'] = str(e)
        return image.copy(), info


def manual_select_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    交互式手动选择四个角点
    
    注意：这需要GUI环境，在无GUI环境中无法使用
    
    Args:
        image: 输入图像
    
    Returns:
        corners: 四个角点坐标，如果取消返回None
    """
    corners = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select 4 corners', image)
            if len(corners) == 4:
                print("已选择4个角点，按任意键继续...")
    
    cv2.namedWindow('Select 4 corners')
    cv2.setMouseCallback('Select 4 corners', mouse_callback)
    
    print("请在图像上点击4个角点（按顺序：左上、右上、右下、左下）")
    cv2.imshow('Select 4 corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(corners) == 4:
        return np.array(corners, dtype=np.float32)
    return None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python aerial_perspective_correction.py <图像路径> [方法]")
        print("方法: auto, building, ground, manual")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        sys.exit(1)
    
    print(f"使用方法: {method}")
    corrected, info = correct_aerial_perspective(image, method=method)
    
    if info['success']:
        print(f"✅ 成功！检测方法: {info['detection_method']}")
        output_path = f"corrected_aerial_{method}.jpg"
        cv2.imwrite(output_path, corrected)
        print(f"结果已保存到: {output_path}")
    else:
        print(f"❌ 失败: {info.get('message', info.get('error', '未知错误'))}")
        print("建议: 尝试手动指定角点或使用专业软件")
