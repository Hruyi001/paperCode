"""
完整的自适应倾斜校正实现
包含所有6种方案的实现
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import sys

# 尝试导入现有方案
try:
    from solution2_edge_detection import detect_skew_angle as hough_detect_skew, correct_skew
    from solution4_perspective_correction import detect_quadrilateral, correct_perspective
    HAS_EXISTING = True
except ImportError:
    HAS_EXISTING = False
    print("警告: 无法导入现有方案，将使用内置实现")


def crop_white_borders(image: np.ndarray, threshold: int = 240, margin: int = 0) -> np.ndarray:
    """
    自动裁剪图像周围的白色边框（改进版，更彻底）
    
    Args:
        image: 输入图像
        threshold: 白色阈值（0-255），大于此值的像素被认为是白色
        margin: 保留的边距（像素），默认为0，完全裁剪到内容边缘
    
    Returns:
        cropped: 裁剪后的图像
    """
    # 转换为灰度图用于检测
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 方法1: 直接阈值检测
    mask = gray < threshold
    
    # 方法2: 如果直接阈值效果不好，尝试更智能的方法
    # 计算每行/每列的非白色像素比例
    h, w = gray.shape
    
    # 从上到下找到第一个有足够非白色像素的行
    row_sums = np.sum(mask, axis=1)
    y_min = 0
    for i in range(h):
        if row_sums[i] > w * 0.01:  # 至少1%的像素是非白色
            y_min = max(0, i - margin)
            break
    
    # 从下到上找到第一个有足够非白色像素的行
    y_max = h - 1
    for i in range(h - 1, -1, -1):
        if row_sums[i] > w * 0.01:
            y_max = min(h - 1, i + margin)
            break
    
    # 从左到右找到第一个有足够非白色像素的列
    col_sums = np.sum(mask, axis=0)
    x_min = 0
    for i in range(w):
        if col_sums[i] > h * 0.01:  # 至少1%的像素是非白色
            x_min = max(0, i - margin)
            break
    
    # 从右到左找到第一个有足够非白色像素的列
    x_max = w - 1
    for i in range(w - 1, -1, -1):
        if col_sums[i] > h * 0.01:
            x_max = min(w - 1, i + margin)
            break
    
    # 检查是否找到有效区域
    if y_max <= y_min or x_max <= x_min:
        # 如果没找到，使用原来的方法作为备选
        coords = np.column_stack(np.where(mask))
        if len(coords) == 0:
            return image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    
    # 裁剪图像
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped


def rotate_image(image: np.ndarray, angle: float, auto_crop: bool = True) -> np.ndarray:
    """
    旋转图像的辅助函数
    
    Args:
        image: 输入图像
        angle: 旋转角度（度）
        auto_crop: 是否自动裁剪白色边框
    
    Returns:
        rotated: 旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    # 自动裁剪白色边框（使用更低的阈值，更彻底地裁剪）
    if auto_crop:
        rotated = crop_white_borders(rotated, threshold=240, margin=0)
    
    return rotated


# ============================================================================
# 方案1: 投影轮廓法（Projection Profile Method）
# ============================================================================

def method1_projection_profile(
    image: np.ndarray,
    angle_range: Tuple[float, float] = (-45, 45),
    angle_step: float = 0.5
) -> Tuple[float, Dict]:
    """
    方案1: 投影轮廓法检测倾斜角度
    
    Args:
        image: 输入图像
        angle_range: 角度搜索范围
        angle_step: 角度搜索步长
    
    Returns:
        angle: 检测到的角度
        info: 检测信息
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 缩小图像以加快计算
    h, w = binary.shape
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(binary, (int(w * scale), int(h * scale)))
    else:
        small = binary
    
    best_angle = 0.0
    max_variance = 0.0
    variances = []
    
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
        
        # 计算方差
        variance = np.var(horizontal_projection)
        variances.append(variance)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    info = {
        'method': 'projection_profile',
        'max_variance': max_variance,
        'angle_range': angle_range,
        'angle_step': angle_step
    }
    
    return best_angle, info


def correct_skew_method1(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案1进行校正"""
    angle, info = method1_projection_profile(image, **kwargs)
    corrected = rotate_image(image, -angle, auto_crop=auto_crop)  # 负角度校正
    info['detected_angle'] = angle
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 方案2: 霍夫直线检测法（Hough Line Detection）
# ============================================================================

def method2_hough_lines(image: np.ndarray) -> Tuple[float, Dict]:
    """
    方案2: 霍夫直线检测法
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的角度
        info: 检测信息
    """
    if HAS_EXISTING:
        angle = hough_detect_skew(image)
    else:
        # 内置实现
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            angle = 0.0
        else:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta) - 90
                if -45 <= angle_deg <= 45:
                    angles.append(angle_deg)
            
            if len(angles) == 0:
                angle = 0.0
            else:
                angle = np.median(angles)
    
    info = {
        'method': 'hough_lines',
        'detected_angle': angle
    }
    
    return angle, info


def correct_skew_method2(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案2进行校正"""
    angle, info = method2_hough_lines(image)
    if HAS_EXISTING:
        corrected, _ = correct_skew(image, angle=angle, auto_detect=False)
        # 如果使用现有函数，也需要裁剪（使用更低的阈值，更彻底地裁剪）
        if auto_crop:
            corrected = crop_white_borders(corrected, threshold=240, margin=0)
    else:
        corrected = rotate_image(image, -angle, auto_crop=auto_crop)
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 方案3: 最小外接矩形法（Minimum Bounding Rectangle）
# ============================================================================

def method3_bounding_rect(image: np.ndarray) -> Tuple[Optional[float], Dict]:
    """
    方案3: 最小外接矩形法
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的角度，如果失败返回None
        info: 检测信息
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
        info = {
            'method': 'bounding_rect',
            'success': False,
            'reason': 'no_contours'
        }
        return None, info
    
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
    
    info = {
        'method': 'bounding_rect',
        'success': True,
        'detected_angle': angle,
        'rect_size': rect[1]
    }
    
    return angle, info


def correct_skew_method3(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案3进行校正"""
    angle, info = method3_bounding_rect(image)
    
    if angle is None:
        # 如果检测失败，返回原图
        info['corrected'] = False
        return image.copy(), info
    
    corrected = rotate_image(image, -angle, auto_crop=auto_crop)
    info['corrected'] = True
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 方案4: 投影变换+旋转组合法（Perspective + Rotation）
# ============================================================================

def method4_perspective_rotation(image: np.ndarray) -> Tuple[Optional[float], Dict]:
    """
    方案4: 先透视校正，再检测倾斜角度
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的角度，如果失败返回None
        info: 检测信息
    """
    try:
        # 先尝试透视校正
        if HAS_EXISTING:
            corners = detect_quadrilateral(image)
            if corners is not None and len(corners) == 4:
                perspective_corrected, _ = correct_perspective(image, corners=corners)
                
                # 在透视校正后的图像上检测倾斜角度
                angle, hough_info = method2_hough_lines(perspective_corrected)
                
                info = {
                    'method': 'perspective_rotation',
                    'success': True,
                    'perspective_corrected': True,
                    'detected_angle': angle,
                    'hough_info': hough_info
                }
                return angle, info
    except:
        pass
    
    # 如果透视校正失败，直接检测倾斜
    angle, hough_info = method2_hough_lines(image)
    
    info = {
        'method': 'perspective_rotation',
        'success': True,
        'perspective_corrected': False,
        'detected_angle': angle,
        'hough_info': hough_info
    }
    
    return angle, info


def correct_skew_method4(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案4进行校正"""
    angle, info = method4_perspective_rotation(image)
    
    if angle is None:
        info['corrected'] = False
        return image.copy(), info
    
    # 如果成功进行了透视校正，先应用透视校正
    if info.get('perspective_corrected', False) and HAS_EXISTING:
        try:
            corners = detect_quadrilateral(image)
            if corners is not None:
                perspective_corrected, _ = correct_perspective(image, corners=corners)
                # 再应用旋转校正
                corrected = rotate_image(perspective_corrected, -angle, auto_crop=auto_crop)
                info['corrected'] = True
                info['corrected_angle'] = -angle
                return corrected, info
        except:
            pass
    
    # 否则只应用旋转校正
    corrected = rotate_image(image, -angle, auto_crop=auto_crop)
    info['corrected'] = True
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 方案5: 频域分析法（Frequency Domain Analysis）
# ============================================================================

def method5_frequency_domain(image: np.ndarray) -> Tuple[float, Dict]:
    """
    方案5: 频域分析法
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的角度
        info: 检测信息
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 缩小图像以加快计算
    h, w = binary.shape
    scale = min(1.0, 512.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(binary, (int(w * scale), int(h * scale)))
    else:
        small = binary
    
    # FFT变换
    f_transform = np.fft.fft2(small)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # 转换为对数尺度以便可视化
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    
    # 检测主方向（简化版：使用边缘检测）
    edges = cv2.Canny((magnitude_spectrum * 255 / magnitude_spectrum.max()).astype(np.uint8), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta) - 90
            if -45 <= angle_deg <= 45:
                angles.append(angle_deg)
        
        if len(angles) > 0:
            angle = np.median(angles)
        else:
            angle = 0.0
    else:
        # 如果FFT方法失败，回退到霍夫直线检测
        angle, _ = method2_hough_lines(image)
    
    info = {
        'method': 'frequency_domain',
        'detected_angle': angle
    }
    
    return angle, info


def correct_skew_method5(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案5进行校正"""
    angle, info = method5_frequency_domain(image)
    corrected = rotate_image(image, -angle, auto_crop=auto_crop)
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 方案6: 组合方法（Combined Methods）
# ============================================================================

def method6_combined(image: np.ndarray) -> Tuple[float, Dict]:
    """
    方案6: 组合多种方法，取中位数或加权平均
    
    Args:
        image: 输入图像
    
    Returns:
        angle: 检测到的角度
        info: 检测信息
    """
    angles = []
    methods_used = []
    method_results = {}
    
    # 方法1: 投影轮廓法
    try:
        angle1, info1 = method1_projection_profile(image)
        angles.append(angle1)
        methods_used.append('projection')
        method_results['projection'] = angle1
    except Exception as e:
        method_results['projection'] = f'failed: {str(e)}'
    
    # 方法2: 霍夫直线检测
    try:
        angle2, info2 = method2_hough_lines(image)
        if abs(angle2) > 0.1:  # 只使用有意义的角度
            angles.append(angle2)
            methods_used.append('hough')
        method_results['hough'] = angle2
    except Exception as e:
        method_results['hough'] = f'failed: {str(e)}'
    
    # 方法3: 最小外接矩形
    try:
        angle3, info3 = method3_bounding_rect(image)
        if angle3 is not None:
            angles.append(angle3)
            methods_used.append('bounding_rect')
        method_results['bounding_rect'] = angle3
    except Exception as e:
        method_results['bounding_rect'] = f'failed: {str(e)}'
    
    # 方法5: 频域分析（跳过方法4，因为它依赖于其他方法）
    try:
        angle5, info5 = method5_frequency_domain(image)
        if abs(angle5) > 0.1:
            angles.append(angle5)
            methods_used.append('frequency')
        method_results['frequency'] = angle5
    except Exception as e:
        method_results['frequency'] = f'failed: {str(e)}'
    
    # 计算最终角度
    if len(angles) > 0:
        # 使用中位数（更鲁棒）
        final_angle = np.median(angles)
        # 也可以使用平均值
        # final_angle = np.mean(angles)
    else:
        final_angle = 0.0
    
    info = {
        'method': 'combined',
        'detected_angle': final_angle,
        'methods_used': methods_used,
        'all_angles': angles,
        'method_results': method_results
    }
    
    return final_angle, info


def correct_skew_method6(image: np.ndarray, auto_crop: bool = True, **kwargs) -> Tuple[np.ndarray, Dict]:
    """使用方案6进行校正"""
    angle, info = method6_combined(image)
    corrected = rotate_image(image, -angle, auto_crop=auto_crop)
    info['corrected_angle'] = -angle
    return corrected, info


# ============================================================================
# 统一接口
# ============================================================================

METHODS = {
    'method1': ('投影轮廓法', correct_skew_method1),
    'method2': ('霍夫直线检测法', correct_skew_method2),
    'method3': ('最小外接矩形法', correct_skew_method3),
    'method4': ('投影变换+旋转组合法', correct_skew_method4),
    'method5': ('频域分析法', correct_skew_method5),
    'method6': ('组合方法', correct_skew_method6),
}


def correct_skew_all_methods(image: np.ndarray) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    使用所有方法进行校正
    
    Args:
        image: 输入图像
    
    Returns:
        results: 字典，键为方法名，值为(校正图像, 信息字典)的元组
    """
    results = {}
    
    for method_key, (method_name, method_func) in METHODS.items():
        try:
            print(f"正在使用 {method_name} ({method_key})...")
            corrected, info = method_func(image)
            results[method_key] = (corrected, info)
            print(f"  ✓ 完成，检测角度: {info.get('detected_angle', 'N/A'):.2f}度")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            results[method_key] = (image.copy(), {
                'method': method_key,
                'error': str(e),
                'success': False
            })
    
    return results
