"""
自适应倾斜校正示例实现
演示几种不同的自适应倾斜检测和校正方法
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import sys

# 尝试导入现有方案
try:
    from solution2_edge_detection import detect_skew_angle, correct_skew
    HAS_SOLUTION2 = True
except ImportError:
    HAS_SOLUTION2 = False
    print("警告: 无法导入 solution2_edge_detection，将使用内置实现")


def detect_skew_by_projection(image: np.ndarray, 
                               angle_range: Tuple[float, float] = (-45, 45),
                               angle_step: float = 0.5) -> float:
    """
    方案1: 投影轮廓法检测倾斜角度
    
    原理: 通过旋转图像并计算水平投影的方差，找到方差最大的角度
    
    Args:
        image: 输入图像（灰度图或彩色图）
        angle_range: 角度搜索范围 (min_angle, max_angle)
        angle_step: 角度搜索步长（度）
    
    Returns:
        angle: 检测到的倾斜角度（度）
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化（可选，对文字图像效果好）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 缩小图像以加快计算（可选）
    h, w = binary.shape
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(binary, (int(w * scale), int(h * scale)))
    else:
        small = binary
    
    best_angle = 0.0
    max_variance = 0.0
    
    # 在角度范围内搜索
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    
    for angle in angles:
        # 旋转图像
        center = (small.shape[1] // 2, small.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        
        # 计算水平投影
        horizontal_projection = np.sum(rotated, axis=1)
        
        # 计算方差（方差越大，说明文字行越清晰）
        variance = np.var(horizontal_projection)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    return best_angle


def detect_skew_by_hough(image: np.ndarray) -> float:
    """
    方案2: 霍夫直线检测法（使用现有实现或内置实现）
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的倾斜角度（度）
    """
    if HAS_SOLUTION2:
        return detect_skew_angle(image)
    else:
        # 内置简单实现
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)
        
        if len(angles) == 0:
            return 0.0
        
        return np.median(angles)


def detect_skew_by_bounding_rect(image: np.ndarray) -> Optional[float]:
    """
    方案3: 最小外接矩形法
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的倾斜角度（度），如果检测失败返回None
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # 调整角度范围到 -45 到 45
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    return angle


def adaptive_skew_correction(image: np.ndarray,
                             method: str = 'auto',
                             angle_threshold: float = 0.5) -> Tuple[np.ndarray, dict]:
    """
    自适应倾斜校正主函数
    
    Args:
        image: 输入图像
        method: 检测方法
            - 'auto': 自动选择最佳方法
            - 'projection': 投影轮廓法
            - 'hough': 霍夫直线检测法
            - 'bounding_rect': 最小外接矩形法
            - 'combined': 组合多种方法
        angle_threshold: 角度阈值，小于此值不校正
    
    Returns:
        corrected_image: 校正后的图像
        info: 包含检测信息的字典
    """
    info = {
        'method_used': method,
        'angle_detected': 0.0,
        'corrected': False
    }
    
    detected_angle = 0.0
    
    if method == 'auto':
        # 自动选择：先尝试霍夫直线，再尝试投影轮廓
        try:
            detected_angle = detect_skew_by_hough(image)
            info['detection_method'] = 'hough'
            if abs(detected_angle) < angle_threshold:
                # 如果霍夫检测角度很小，尝试投影轮廓法验证
                proj_angle = detect_skew_by_projection(image)
                if abs(proj_angle) > abs(detected_angle):
                    detected_angle = proj_angle
                    info['detection_method'] = 'projection'
        except:
            detected_angle = detect_skew_by_projection(image)
            info['detection_method'] = 'projection'
    
    elif method == 'projection':
        detected_angle = detect_skew_by_projection(image)
        info['detection_method'] = 'projection'
    
    elif method == 'hough':
        detected_angle = detect_skew_by_hough(image)
        info['detection_method'] = 'hough'
    
    elif method == 'bounding_rect':
        angle = detect_skew_by_bounding_rect(image)
        if angle is not None:
            detected_angle = angle
            info['detection_method'] = 'bounding_rect'
        else:
            info['detection_method'] = 'bounding_rect (failed)'
    
    elif method == 'combined':
        # 组合多种方法，取中位数
        angles = []
        methods_used = []
        
        try:
            hough_angle = detect_skew_by_hough(image)
            angles.append(hough_angle)
            methods_used.append('hough')
        except:
            pass
        
        try:
            proj_angle = detect_skew_by_projection(image)
            angles.append(proj_angle)
            methods_used.append('projection')
        except:
            pass
        
        try:
            rect_angle = detect_skew_by_bounding_rect(image)
            if rect_angle is not None:
                angles.append(rect_angle)
                methods_used.append('bounding_rect')
        except:
            pass
        
        if len(angles) > 0:
            detected_angle = np.median(angles)
            info['detection_method'] = f"combined ({', '.join(methods_used)})"
            info['all_angles'] = angles
        else:
            detected_angle = 0.0
            info['detection_method'] = 'combined (all failed)'
    
    else:
        raise ValueError(f"未知的方法: {method}")
    
    info['angle_detected'] = detected_angle
    
    # 如果角度小于阈值，不进行校正
    if abs(detected_angle) < angle_threshold:
        info['corrected'] = False
        return image.copy(), info
    
    # 应用校正
    if HAS_SOLUTION2:
        corrected, _ = correct_skew(image, angle=detected_angle, auto_detect=False)
    else:
        # 内置校正实现
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        corrected = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
        )
    
    info['corrected'] = True
    return corrected, info


def demo():
    """演示函数"""
    print("=" * 60)
    print("自适应倾斜校正演示")
    print("=" * 60)
    print()
    
    # 创建测试图像（模拟倾斜的文档）
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制一些文字行（模拟文档）
    for i in range(8):
        y = 100 + i * 60
        cv2.line(img, (80, y), (720, y), (0, 0, 0), 3)
        cv2.putText(img, f"Line {i+1}", (100, y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 创建倾斜图像（旋转5度）
    true_angle = 5.0
    center = (400, 300)
    M = cv2.getRotationMatrix2D(center, true_angle, 1.0)
    skewed = cv2.warpAffine(img, M, (800, 600),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    
    print(f"原始倾斜角度: {true_angle}度")
    print()
    
    # 测试不同方法
    methods = ['auto', 'projection', 'hough', 'combined']
    
    for method in methods:
        print(f"测试方法: {method}")
        print("-" * 40)
        
        try:
            corrected, info = adaptive_skew_correction(skewed, method=method)
            print(f"  检测到的角度: {info['angle_detected']:.2f}度")
            print(f"  使用的检测方法: {info['detection_method']}")
            print(f"  是否校正: {info['corrected']}")
            
            if 'all_angles' in info:
                print(f"  所有检测角度: {[f'{a:.2f}' for a in info['all_angles']]}")
            
            error = abs(info['angle_detected'] - true_angle)
            print(f"  误差: {error:.2f}度")
            print()
            
        except Exception as e:
            print(f"  错误: {e}")
            print()
    
    print("=" * 60)
    print("演示完成！")
    print()
    print("使用方法:")
    print("  from adaptive_skew_correction_demo import adaptive_skew_correction")
    print("  import cv2")
    print("  img = cv2.imread('your_image.jpg')")
    print("  corrected, info = adaptive_skew_correction(img, method='auto')")
    print("  print(f'检测角度: {info[\"angle_detected\"]:.2f}度')")
    print("  cv2.imwrite('corrected.jpg', corrected)")


if __name__ == '__main__':
    demo()
