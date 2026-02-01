"""
倾斜无人机视图转正射视图校正模块
基于特征匹配 → 单应性矩阵 → 角度提取 → 视角校正变换
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import math


class OrthoCorrection:
    """正射校正类"""
    
    def __init__(self, feature_type='SIFT', ratio_threshold=0.75):
        """
        初始化
        
        Args:
            feature_type: 特征提取类型 ('SIFT' 或 'ORB')
            ratio_threshold: 特征匹配的ratio test阈值
        """
        self.feature_type = feature_type
        self.ratio_threshold = ratio_threshold
        self.detector = None
        self.matcher = None
        self._init_feature_detector()
    
    def _init_feature_detector(self):
        """初始化特征检测器"""
        if self.feature_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            # SIFT使用L2距离，需要BF匹配器
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif self.feature_type == 'ORB':
            self.detector = cv2.ORB_create()
            # ORB使用汉明距离
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")
    
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取图像特征点
        
        Args:
            image: 输入图像
            
        Returns:
            keypoints: 关键点列表
            descriptors: 特征描述子
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """
        匹配特征点
        
        Args:
            desc1: 第一张图的特征描述子
            desc2: 第二张图的特征描述子
            
        Returns:
            good_matches: 过滤后的匹配点对
        """
        if desc1 is None or desc2 is None:
            return []
        
        # 执行匹配
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Ratio test过滤
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography(self, kp1: list, kp2: list, matches: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单应性矩阵
        
        Args:
            kp1: 第一张图的关键点
            kp2: 第二张图的关键点
            matches: 匹配点对
            
        Returns:
            H: 单应性矩阵 (3x3)
            mask: 内点掩码
        """
        if len(matches) < 4:
            raise ValueError(f"匹配点数量不足: {len(matches)}, 至少需要4个")
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        return H, mask
    
    def extract_tilt_angles(self, H: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        从单应性矩阵提取倾斜角度
        
        Args:
            H: 单应性矩阵 (3x3)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            pitch: 俯仰角（度）
            roll: 翻滚角（度）
        """
        h, w = image_shape
        
        # 方法：分析单应性矩阵对图像四个角点的变换效果
        # 计算变换前后角点的位置关系来估计倾斜角度
        
        # 图像四个角点（归一化坐标）
        corners = np.array([
            [0, 0, 1],      # 左上
            [w, 0, 1],     # 右上
            [w, h, 1],     # 右下
            [0, h, 1]      # 左下
        ], dtype=np.float32).T
        
        # 应用单应性变换
        transformed_corners = H @ corners
        transformed_corners = transformed_corners / transformed_corners[2, :]  # 齐次坐标归一化
        
        # 计算原始和变换后的中心点
        orig_center = np.array([w/2, h/2])
        trans_center = np.mean(transformed_corners[:2, :], axis=1)
        
        # 计算各角点相对于中心的位置变化
        orig_vectors = corners[:2, :].T - orig_center
        trans_vectors = transformed_corners[:2, :].T - trans_center
        
        # 分析垂直方向的倾斜（俯仰角）
        # 比较上下边缘的压缩/拉伸
        top_y = np.mean([transformed_corners[1, 0], transformed_corners[1, 1]])
        bottom_y = np.mean([transformed_corners[1, 2], transformed_corners[1, 3]])
        orig_top_y = 0
        orig_bottom_y = h
        
        # 计算垂直方向的倾斜角度
        if abs(bottom_y - top_y) > 1e-6:
            # 通过比较上下边缘的压缩比例估计俯仰角
            vertical_compression = (bottom_y - top_y) / h
            # 简化估计：假设是透视投影，通过压缩比例反推角度
            pitch = math.degrees(math.acos(min(1.0, vertical_compression)))
        else:
            pitch = 0.0
        
        # 分析水平方向的倾斜（翻滚角）
        left_x = np.mean([transformed_corners[0, 0], transformed_corners[0, 3]])
        right_x = np.mean([transformed_corners[0, 1], transformed_corners[0, 2]])
        orig_left_x = 0
        orig_right_x = w
        
        if abs(right_x - left_x) > 1e-6:
            horizontal_compression = (right_x - left_x) / w
            roll = math.degrees(math.acos(min(1.0, horizontal_compression)))
        else:
            roll = 0.0
        
        # 更精确的方法：使用单应性矩阵的分解
        # 尝试从H中提取旋转信息
        try:
            # 假设相机内参为单位矩阵（简化）
            K = np.eye(3)
            # 尝试分解单应性矩阵
            # H = K * [R | t] * K^(-1) 对于平面场景
            
            # 提取H的前两列作为旋转矩阵的前两列
            h1 = H[:, 0]
            h2 = H[:, 1]
            
            # 归一化
            norm1 = np.linalg.norm(h1)
            norm2 = np.linalg.norm(h2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                h1 = h1 / norm1
                h2 = h2 / norm2
                
                # 构建旋转矩阵的前两列
                r1 = h1
                r2 = h2
                r3 = np.cross(r1, r2)
                
                # 构建完整的旋转矩阵
                R = np.column_stack([r1, r2, r3])
                
                # 使用SVD确保R是正交矩阵
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt
                
                # 从旋转矩阵提取欧拉角（ZYX顺序）
                # 但这里我们主要关心俯仰角和翻滚角
                sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
                singular = sy < 1e-6
                
                if not singular:
                    pitch_from_R = math.degrees(math.atan2(-R[2, 0], sy))
                    roll_from_R = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    
                    # 使用更精确的旋转矩阵分解结果
                    pitch = pitch_from_R
                    roll = roll_from_R
        except:
            # 如果分解失败，使用之前估计的角度
            pass
        
        return pitch, roll
    
    def build_perspective_correction_matrix(self, pitch: float, roll: float, 
                                           image_shape: Tuple[int, int],
                                           focal_length: Optional[float] = None) -> np.ndarray:
        """
        基于倾斜角度构建视角校正变换矩阵（改进版）
        直接校正到垂直俯视角度（90度）
        
        Args:
            pitch: 俯仰角（度），从水平方向向下倾斜的角度
            roll: 翻滚角（度），左右倾斜的角度
            image_shape: 图像尺寸 (height, width)
            focal_length: 焦距（像素），如果为None则自动估计
            
        Returns:
            H_correct: 视角校正变换矩阵 (3x3)
        """
        h, w = image_shape
        
        # 如果没有提供焦距，使用图像对角线长度作为估计
        if focal_length is None:
            focal_length = max(w, h) * 1.2  # 使用更合理的焦距估计
        
        # 转换为弧度
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # 构建相机内参矩阵
        K = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 构建旋转矩阵：从当前倾斜视角旋转到垂直俯视（90度）
        # 首先绕X轴旋转（校正翻滚角）
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(-roll_rad), -math.sin(-roll_rad)],
            [0, math.sin(-roll_rad), math.cos(-roll_rad)]
        ], dtype=np.float64)
        
        # 然后绕Y轴旋转（校正俯仰角，旋转到90度垂直向下）
        # 当前俯仰角是pitch，需要旋转到90度（垂直向下）
        # 所以需要旋转的角度是 (90 - pitch) 度，但这里pitch是从水平向下的角度
        # 如果pitch是向下倾斜的角度，那么需要旋转 -pitch 度来校正
        Ry = np.array([
            [math.cos(-pitch_rad), 0, math.sin(-pitch_rad)],
            [0, 1, 0],
            [-math.sin(-pitch_rad), 0, math.cos(-pitch_rad)]
        ], dtype=np.float64)
        
        # 组合旋转：先校正翻滚，再校正俯仰
        R_correct = Ry @ Rx
        
        # 构建透视变换矩阵
        # H = K * R_correct * K^(-1)
        K_inv = np.linalg.inv(K)
        H_correct = K @ R_correct @ K_inv
        
        # 归一化
        H_correct = H_correct / H_correct[2, 2]
        
        return H_correct
    
    def build_simple_perspective_matrix(self, pitch: float, roll: float,
                                       image_shape: Tuple[int, int]) -> np.ndarray:
        """
        构建简单的透视校正矩阵（基于单应性变换）
        使用更直接的方法，适合小到中等角度倾斜
        
        Args:
            pitch: 俯仰角（度）
            roll: 翻滚角（度）
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            H: 单应性变换矩阵 (3x3)
        """
        h, w = image_shape
        
        # 转换为弧度
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # 计算透视变换参数
        # 基于倾斜角度计算单应性矩阵的参数
        # 这是一个简化的方法，假设地面是平面
        
        # 计算倾斜导致的透视畸变
        # 俯仰角影响垂直方向的透视
        tan_pitch = math.tan(pitch_rad)
        tan_roll = math.tan(roll_rad)
        
        # 构建单应性矩阵
        # 这是一个简化的透视变换矩阵
        H = np.eye(3, dtype=np.float64)
        
        # 根据角度调整矩阵元素
        # 俯仰角主要影响垂直方向的缩放和透视
        if abs(pitch_rad) > 0.01:  # 如果角度足够大
            # 垂直方向的透视校正
            scale_y = 1.0 / math.cos(pitch_rad)
            H[1, 1] = scale_y
            H[1, 2] = h * (1 - scale_y) / 2
        
        # 翻滚角主要影响水平方向的透视
        if abs(roll_rad) > 0.01:
            # 水平方向的透视校正
            scale_x = 1.0 / math.cos(roll_rad)
            H[0, 0] = scale_x
            H[0, 2] = w * (1 - scale_x) / 2
        
        # 添加透视效果（基于角度）
        # 这需要更复杂的计算，这里使用简化方法
        if abs(pitch_rad) > 0.01:
            # 添加垂直方向的透视畸变校正
            perspective_factor = math.sin(pitch_rad) * 0.001
            H[2, 1] = perspective_factor
        
        if abs(roll_rad) > 0.01:
            # 添加水平方向的透视畸变校正
            perspective_factor = math.sin(roll_rad) * 0.001
            H[2, 0] = perspective_factor
        
        return H
    
    def correct_to_vertical_view(self, tilted_image: np.ndarray, pitch: float, roll: float,
                                height_estimate: Optional[float] = None) -> np.ndarray:
        """
        直接校正到垂直俯视角度（90度垂直向下）
        使用更准确的透视投影方法
        
        Args:
            tilted_image: 倾斜的无人机图像
            pitch: 俯仰角（度），从水平方向向下倾斜的角度（0度=水平，90度=垂直向下）
            roll: 翻滚角（度），左右倾斜的角度
            height_estimate: 估计的拍摄高度（米），用于计算更准确的透视变换
            
        Returns:
            corrected_image: 校正后的垂直俯视图像
        """
        print(f"校正到垂直俯视角度...")
        print(f"  当前俯仰角: {pitch:.2f}° (需要校正到 90°)")
        print(f"  翻滚角: {roll:.2f}°")
        
        h, w = tilted_image.shape[:2]
        
        # 转换为弧度
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # 计算需要旋转的角度来达到垂直俯视（90度）
        # 如果当前是pitch度向下，需要再旋转 (90 - pitch) 度
        target_pitch = 90.0  # 目标：垂直向下
        pitch_correction = math.radians(target_pitch - pitch)
        roll_correction = -roll_rad  # 校正翻滚角
        
        # 使用更准确的焦距估计
        # 对于无人机图像，焦距通常与图像尺寸相关
        focal_length = max(w, h) * 1.5
        
        # 构建相机内参矩阵
        K = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 构建旋转矩阵：校正到垂直俯视
        # 先校正翻滚角（绕X轴）
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_correction), -math.sin(roll_correction)],
            [0, math.sin(roll_correction), math.cos(roll_correction)]
        ], dtype=np.float64)
        
        # 再校正俯仰角（绕Y轴），旋转到垂直向下
        Ry = np.array([
            [math.cos(pitch_correction), 0, math.sin(pitch_correction)],
            [0, 1, 0],
            [-math.sin(pitch_correction), 0, math.cos(pitch_correction)]
        ], dtype=np.float64)
        
        # 组合旋转
        R = Ry @ Rx
        
        # 构建透视变换矩阵
        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv
        
        # 归一化
        H = H / H[2, 2]
        
        # 应用变换
        corrected_image = self.apply_correction(tilted_image, H)
        print("  完成!")
        
        return corrected_image
    
    def apply_correction(self, image: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        应用透视变换
        
        Args:
            image: 输入图像
            H: 变换矩阵
            
        Returns:
            corrected: 校正后的图像
        """
        h, w = image.shape[:2]
        
        # 计算变换后的图像边界
        corners = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        
        # 应用变换到角点
        corners_homogeneous = np.column_stack([corners, np.ones(4)])
        transformed_corners = (H @ corners_homogeneous.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
        
        # 计算输出图像尺寸
        x_min = int(np.floor(transformed_corners[:, 0].min()))
        x_max = int(np.ceil(transformed_corners[:, 0].max()))
        y_min = int(np.floor(transformed_corners[:, 1].min()))
        y_max = int(np.ceil(transformed_corners[:, 1].max()))
        
        # 添加平移以保持图像在正象限
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        H_final = translation @ H
        
        # 计算输出尺寸
        out_w = x_max - x_min
        out_h = y_max - y_min
        
        # 应用变换
        corrected = cv2.warpPerspective(
            image, H_final, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return corrected
    
    def process(self, tilted_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        完整的处理流程
        
        Args:
            tilted_image: 倾斜的无人机图像
            reference_image: 参考正射图像
            
        Returns:
            corrected_image: 校正后的正射图像
            pitch: 俯仰角（度）
            roll: 翻滚角（度）
        """
        # 步骤1: 特征提取
        print("步骤1: 提取特征点...")
        kp1, desc1 = self.extract_features(tilted_image)
        kp2, desc2 = self.extract_features(reference_image)
        print(f"  倾斜图特征点: {len(kp1)}")
        print(f"  参考图特征点: {len(kp2)}")
        
        if desc1 is None or desc2 is None:
            raise ValueError("无法提取特征点，请检查图像")
        
        # 步骤2: 特征匹配
        print("步骤2: 匹配特征点...")
        matches = self.match_features(desc1, desc2)
        print(f"  初始匹配点: {len(matches)}")
        
        if len(matches) < 4:
            raise ValueError(f"匹配点数量不足: {len(matches)}, 至少需要4个")
        
        # 步骤3: 计算单应性矩阵
        print("步骤3: 计算单应性矩阵...")
        H, mask = self.compute_homography(kp1, kp2, matches)
        inliers = np.sum(mask)
        print(f"  内点数量: {inliers}/{len(matches)}")
        
        # 步骤4: 提取倾斜角度
        print("步骤4: 提取倾斜角度...")
        pitch, roll = self.extract_tilt_angles(H, tilted_image.shape[:2])
        print(f"  俯仰角 (pitch): {pitch:.2f}°")
        print(f"  翻滚角 (roll): {roll:.2f}°")
        
        # 步骤5: 构建视角校正变换矩阵
        print("步骤5: 构建视角校正变换矩阵...")
        H_correct = self.build_perspective_correction_matrix(
            pitch, roll, tilted_image.shape[:2]
        )
        
        # 步骤6: 应用变换
        print("步骤6: 应用透视变换...")
        corrected_image = self.apply_correction(tilted_image, H_correct)
        print("  完成!")
        
        return corrected_image, pitch, roll
    
    def correct_from_angles(self, tilted_image: np.ndarray, pitch: float, roll: float,
                           focal_length: Optional[float] = None, 
                           use_simple: bool = False) -> np.ndarray:
        """
        基于给定的倾斜角度直接校正图像（不需要参考图）
        
        Args:
            tilted_image: 倾斜的无人机图像
            pitch: 俯仰角（度），从水平方向向下倾斜的角度
            roll: 翻滚角（度），左右倾斜的角度
            focal_length: 焦距（像素），如果为None则自动估计
            use_simple: 是否使用简化的透视变换方法（适合小角度）
            
        Returns:
            corrected_image: 校正后的正射图像
        """
        print(f"基于角度进行校正...")
        print(f"  俯仰角 (pitch): {pitch:.2f}°")
        print(f"  翻滚角 (roll): {roll:.2f}°")
        
        if use_simple:
            # 使用简化的透视变换方法
            print("  使用简化透视变换方法...")
            H_correct = self.build_simple_perspective_matrix(
                pitch, roll, tilted_image.shape[:2]
            )
        else:
            # 使用完整的透视变换方法
            print("  使用完整透视变换方法...")
            H_correct = self.build_perspective_correction_matrix(
                pitch, roll, tilted_image.shape[:2], focal_length
            )
        
        # 应用变换
        corrected_image = self.apply_correction(tilted_image, H_correct)
        print("  完成!")
        
        return corrected_image


def create_comparison_image(tilted_img: np.ndarray, reference_img: np.ndarray, 
                           corrected_img: np.ndarray, output_path: str,
                           pitch: float = 0.0, roll: float = 0.0) -> Optional[str]:
    """
    创建对比图，展示原始倾斜图、参考正射图和校正后的图像
    
    Args:
        tilted_img: 原始倾斜图像
        reference_img: 参考正射图像
        corrected_img: 校正后的图像
        output_path: 输出文件路径
        pitch: 俯仰角（度）
        roll: 翻滚角（度）
        
    Returns:
        对比图的保存路径
    """
    import os
    
    # 确定对比图的保存路径
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_name)[0]
    output_ext = os.path.splitext(output_name)[1]
    comparison_path = os.path.join(output_dir, f"{output_name_no_ext}_comparison{output_ext}")
    
    # 设置统一的显示高度（像素）
    display_height = 600
    
    # 调整图像大小，保持宽高比
    def resize_with_aspect_ratio(img, target_height):
        h, w = img.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    tilted_resized = resize_with_aspect_ratio(tilted_img, display_height)
    reference_resized = resize_with_aspect_ratio(reference_img, display_height)
    corrected_resized = resize_with_aspect_ratio(corrected_img, display_height)
    
    # 确保所有图像高度一致
    h = display_height
    tilted_w = tilted_resized.shape[1]
    reference_w = reference_resized.shape[1]
    corrected_w = corrected_resized.shape[1]
    
    # 创建对比图（三张图并排，额外空间用于角度信息）
    total_width = tilted_w + reference_w + corrected_w
    comparison = np.zeros((h + 100, total_width, 3), dtype=np.uint8)  # 额外100像素用于文字和角度信息
    
    # 放置图像
    comparison[50:h+50, 0:tilted_w] = tilted_resized
    comparison[50:h+50, tilted_w:tilted_w+reference_w] = reference_resized
    comparison[50:h+50, tilted_w+reference_w:tilted_w+reference_w+corrected_w] = corrected_resized
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_scale_small = 0.6
    thickness = 2
    thickness_small = 1
    color = (255, 255, 255)
    color_angle = (0, 255, 255)  # 黄色用于角度信息
    
    # 计算文字位置（居中）
    def get_text_position(text, x_start, width):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x_start + (width - text_size[0]) // 2
        return text_x, 30
    
    # 添加标题标签
    text1_x, _ = get_text_position("Original Tilted", 0, tilted_w)
    cv2.putText(comparison, "Original Tilted", (text1_x, 30), font, font_scale, color, thickness)
    
    text2_x, _ = get_text_position("Reference Ortho", tilted_w, reference_w)
    cv2.putText(comparison, "Reference Ortho", (text2_x, 30), font, font_scale, color, thickness)
    
    text3_x, _ = get_text_position("Corrected Result", tilted_w + reference_w, corrected_w)
    cv2.putText(comparison, "Corrected Result", (text3_x, 30), font, font_scale, color, thickness)
    
    # 添加角度信息（在校正结果图下方）
    angle_text_y = h + 75
    angle_info = f"Pitch: {pitch:.1f}°  Roll: {roll:.1f}°"
    angle_text_size = cv2.getTextSize(angle_info, font, font_scale_small, thickness_small)[0]
    angle_text_x = tilted_w + reference_w + (corrected_w - angle_text_size[0]) // 2
    cv2.putText(comparison, angle_info, (angle_text_x, angle_text_y), 
                font, font_scale_small, color_angle, thickness_small)
    
    # 添加总角度信息
    total_angle = math.sqrt(pitch**2 + roll**2)
    total_angle_text = f"Total: {total_angle:.1f}°"
    total_text_size = cv2.getTextSize(total_angle_text, font, font_scale_small, thickness_small)[0]
    total_text_x = tilted_w + reference_w + (corrected_w - total_text_size[0]) // 2
    cv2.putText(comparison, total_angle_text, (total_text_x, angle_text_y + 20), 
                font, font_scale_small, color_angle, thickness_small)
    
    # 添加分隔线
    cv2.line(comparison, (tilted_w, 50), (tilted_w, h+50), (255, 255, 255), 2)
    cv2.line(comparison, (tilted_w + reference_w, 50), (tilted_w + reference_w, h+50), (255, 255, 255), 2)
    
    # 保存对比图
    success = cv2.imwrite(comparison_path, comparison)
    if success:
        return comparison_path
    else:
        print(f"警告: 无法保存对比图到 {comparison_path}")
        return None


def correct_from_angles_main():
    """基于角度校正的主函数"""
    import sys
    import os
    
    if len(sys.argv) < 4:
        print("用法: python ortho_correction.py --angles <倾斜图路径> <pitch角度> <roll角度> [输出路径] [焦距]")
        print("示例: python ortho_correction.py --angles tilted.jpg 25.5 12.3 output.jpg")
        print("      python ortho_correction.py --angles tilted.jpg 25.5 12.3 output.jpg 2000")
        return
    
    tilted_path = sys.argv[2]
    pitch = float(sys.argv[3])
    roll = float(sys.argv[4])
    output_path = sys.argv[5] if len(sys.argv) > 5 else "corrected_output.jpg"
    focal_length = float(sys.argv[6]) if len(sys.argv) > 6 else None
    
    # 处理输出路径
    if os.path.isdir(output_path) or (not os.path.exists(output_path) and not output_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))):
        tilted_basename = os.path.basename(tilted_path)
        tilted_name, tilted_ext = os.path.splitext(tilted_basename)
        if tilted_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            output_ext = tilted_ext
        else:
            output_ext = '.jpg'
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"{tilted_name}_corrected{output_ext}")
        else:
            output_path = output_path + output_ext
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 读取图像
    print(f"读取倾斜图: {tilted_path}")
    tilted_image = cv2.imread(tilted_path)
    if tilted_image is None:
        raise ValueError(f"无法读取图像: {tilted_path}")
    
    # 创建校正器
    corrector = OrthoCorrection()
    
    # 执行校正（使用改进的垂直俯视校正方法）
    corrected_image = corrector.correct_to_vertical_view(tilted_image, pitch, roll)
    
    # 输出角度信息
    total_angle = math.sqrt(pitch**2 + roll**2)
    print("\n" + "="*50)
    print("校正角度信息:")
    print("="*50)
    print(f"Pitch (俯仰角): {pitch:.2f}°")
    print(f"Roll  (翻滚角): {roll:.2f}°")
    print(f"总倾斜角度: {total_angle:.2f}°")
    print("="*50 + "\n")
    
    # 保存结果
    print(f"保存结果: {output_path}")
    success = cv2.imwrite(output_path, corrected_image)
    if not success:
        raise ValueError(f"无法保存图像到: {output_path}")
    print("处理完成!")
    
    # 生成对比图（只有原始图和校正后的图）
    print("生成对比图...")
    comparison_path = create_simple_comparison_image(
        tilted_image, corrected_image, output_path, pitch, roll
    )
    if comparison_path:
        print(f"对比图已保存到: {comparison_path}")


def create_simple_comparison_image(tilted_img: np.ndarray, corrected_img: np.ndarray,
                                  output_path: str, pitch: float = 0.0, roll: float = 0.0) -> Optional[str]:
    """
    创建简单的对比图（只有原始图和校正后的图，没有参考图）
    
    Args:
        tilted_img: 原始倾斜图像
        corrected_img: 校正后的图像
        output_path: 输出文件路径
        pitch: 俯仰角（度）
        roll: 翻滚角（度）
        
    Returns:
        对比图的保存路径
    """
    import os
    
    # 确定对比图的保存路径
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_name)[0]
    output_ext = os.path.splitext(output_name)[1]
    comparison_path = os.path.join(output_dir, f"{output_name_no_ext}_comparison{output_ext}")
    
    # 设置统一的显示高度
    display_height = 600
    
    # 调整图像大小，保持宽高比
    def resize_with_aspect_ratio(img, target_height):
        h, w = img.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    tilted_resized = resize_with_aspect_ratio(tilted_img, display_height)
    corrected_resized = resize_with_aspect_ratio(corrected_img, display_height)
    
    # 确保所有图像高度一致
    h = display_height
    tilted_w = tilted_resized.shape[1]
    corrected_w = corrected_resized.shape[1]
    
    # 创建对比图（两张图并排）
    total_width = tilted_w + corrected_w
    comparison = np.zeros((h + 100, total_width, 3), dtype=np.uint8)
    
    # 放置图像
    comparison[50:h+50, 0:tilted_w] = tilted_resized
    comparison[50:h+50, tilted_w:tilted_w+corrected_w] = corrected_resized
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_scale_small = 0.6
    thickness = 2
    thickness_small = 1
    color = (255, 255, 255)
    color_angle = (0, 255, 255)
    
    # 计算文字位置（居中）
    def get_text_position(text, x_start, width):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x_start + (width - text_size[0]) // 2
        return text_x, 30
    
    # 添加标题标签
    text1_x, _ = get_text_position("Original Tilted", 0, tilted_w)
    cv2.putText(comparison, "Original Tilted", (text1_x, 30), font, font_scale, color, thickness)
    
    text2_x, _ = get_text_position("Corrected Result", tilted_w, corrected_w)
    cv2.putText(comparison, "Corrected Result", (text2_x, 30), font, font_scale, color, thickness)
    
    # 添加角度信息
    angle_text_y = h + 75
    angle_info = f"Pitch: {pitch:.1f}°  Roll: {roll:.1f}°"
    angle_text_size = cv2.getTextSize(angle_info, font, font_scale_small, thickness_small)[0]
    angle_text_x = tilted_w + (corrected_w - angle_text_size[0]) // 2
    cv2.putText(comparison, angle_info, (angle_text_x, angle_text_y), 
                font, font_scale_small, color_angle, thickness_small)
    
    # 添加总角度信息
    total_angle = math.sqrt(pitch**2 + roll**2)
    total_angle_text = f"Total: {total_angle:.1f}°"
    total_text_size = cv2.getTextSize(total_angle_text, font, font_scale_small, thickness_small)[0]
    total_text_x = tilted_w + (corrected_w - total_text_size[0]) // 2
    cv2.putText(comparison, total_angle_text, (total_text_x, angle_text_y + 20), 
                font, font_scale_small, color_angle, thickness_small)
    
    # 添加分隔线
    cv2.line(comparison, (tilted_w, 50), (tilted_w, h+50), (255, 255, 255), 2)
    
    # 保存对比图
    success = cv2.imwrite(comparison_path, comparison)
    if success:
        return comparison_path
    else:
        print(f"警告: 无法保存对比图到 {comparison_path}")
        return None


def main():
    """主函数示例"""
    import sys
    import os
    
    # 检查是否是角度模式
    if len(sys.argv) > 1 and sys.argv[1] == '--angles':
        correct_from_angles_main()
        return
    
    if len(sys.argv) < 3:
        print("用法1（基于参考图）: python ortho_correction.py <倾斜图路径> <参考正射图路径> [输出路径]")
        print("用法2（基于角度）: python ortho_correction.py --angles <倾斜图路径> <pitch角度> <roll角度> [输出路径] [焦距]")
        print("示例1: python ortho_correction.py tilted.jpg reference.jpg output.jpg")
        print("示例2: python ortho_correction.py --angles tilted.jpg 25.5 12.3 output.jpg")
        return
    
    tilted_path = sys.argv[1]
    reference_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "corrected_output.jpg"
    
    # 处理输出路径：如果是目录，自动生成文件名
    if os.path.isdir(output_path) or (not os.path.exists(output_path) and not output_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))):
        # 输出路径是目录或没有扩展名，从输入文件名生成输出文件名
        tilted_basename = os.path.basename(tilted_path)
        tilted_name, tilted_ext = os.path.splitext(tilted_basename)
        
        # 确定输出扩展名（优先使用输入图像的扩展名，否则使用.jpg）
        if tilted_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            output_ext = tilted_ext
        else:
            output_ext = '.jpg'
        
        # 生成输出文件名
        if os.path.isdir(output_path):
            # 如果是目录，在目录下创建文件
            output_path = os.path.join(output_path, f"{tilted_name}_corrected{output_ext}")
        else:
            # 如果没有扩展名，添加扩展名
            output_path = output_path + output_ext
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 读取图像
    print(f"读取倾斜图: {tilted_path}")
    tilted_image = cv2.imread(tilted_path)
    if tilted_image is None:
        raise ValueError(f"无法读取图像: {tilted_path}")
    
    print(f"读取参考图: {reference_path}")
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        raise ValueError(f"无法读取图像: {reference_path}")
    
    # 创建校正器
    corrector = OrthoCorrection(feature_type='SIFT')
    
    # 执行校正
    corrected_image, pitch, roll = corrector.process(tilted_image, reference_image)
    
    # 输出校正角度信息
    print("\n" + "="*50)
    print("校正角度信息:")
    print("="*50)
    print(f"Pitch (俯仰角): {pitch:.2f}°")
    print(f"Roll  (翻滚角): {roll:.2f}°")
    print(f"总倾斜角度: {math.sqrt(pitch**2 + roll**2):.2f}°")
    print("="*50 + "\n")
    
    # 保存结果
    print(f"保存结果: {output_path}")
    success = cv2.imwrite(output_path, corrected_image)
    if not success:
        raise ValueError(f"无法保存图像到: {output_path}")
    print("处理完成!")
    
    # 生成对比图
    print("生成对比图...")
    comparison_path = create_comparison_image(
        tilted_image, reference_image, corrected_image, output_path, pitch, roll
    )
    if comparison_path:
        print(f"对比图已保存到: {comparison_path}")


if __name__ == "__main__":
    main()
