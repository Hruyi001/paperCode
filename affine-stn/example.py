"""
使用示例脚本
演示如何使用 OrthoCorrection 进行视角校正
"""

import cv2
import numpy as np
from ortho_correction import OrthoCorrection


def example_usage():
    """示例用法"""
    
    # 示例1: 基本使用
    print("=" * 50)
    print("示例1: 基本使用")
    print("=" * 50)
    
    # 读取图像（请替换为你的图像路径）
    tilted_path = "tilted_image.jpg"  # 倾斜的无人机图像
    reference_path = "reference_ortho.jpg"  # 参考正射图像
    
    try:
        tilted_image = cv2.imread(tilted_path)
        reference_image = cv2.imread(reference_path)
        
        if tilted_image is None or reference_image is None:
            print(f"警告: 无法读取图像文件")
            print("请确保图像文件存在，或修改路径")
            return
        
        # 创建校正器（使用SIFT特征）
        corrector = OrthoCorrection(feature_type='SIFT', ratio_threshold=0.75)
        
        # 执行校正
        corrected_image = corrector.process(tilted_image, reference_image)
        
        # 保存结果
        output_path = "corrected_output.jpg"
        cv2.imwrite(output_path, corrected_image)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 如果图像文件不存在，请先准备测试图像")


def example_with_orb():
    """使用ORB特征的示例（速度更快）"""
    
    print("\n" + "=" * 50)
    print("示例2: 使用ORB特征（速度更快）")
    print("=" * 50)
    
    tilted_path = "tilted_image.jpg"
    reference_path = "reference_ortho.jpg"
    
    try:
        tilted_image = cv2.imread(tilted_path)
        reference_image = cv2.imread(reference_path)
        
        if tilted_image is None or reference_image is None:
            print("警告: 无法读取图像文件")
            return
        
        # 使用ORB特征（速度更快，但精度可能略低）
        corrector = OrthoCorrection(feature_type='ORB', ratio_threshold=0.7)
        
        corrected_image = corrector.process(tilted_image, reference_image)
        
        output_path = "corrected_output_orb.jpg"
        cv2.imwrite(output_path, corrected_image)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")


def example_step_by_step():
    """分步骤处理示例（更灵活的控制）"""
    
    print("\n" + "=" * 50)
    print("示例3: 分步骤处理")
    print("=" * 50)
    
    tilted_path = "tilted_image.jpg"
    reference_path = "reference_ortho.jpg"
    
    try:
        tilted_image = cv2.imread(tilted_path)
        reference_image = cv2.imread(reference_path)
        
        if tilted_image is None or reference_image is None:
            print("警告: 无法读取图像文件")
            return
        
        corrector = OrthoCorrection(feature_type='SIFT')
        
        # 步骤1: 特征提取
        print("步骤1: 提取特征点...")
        kp1, desc1 = corrector.extract_features(tilted_image)
        kp2, desc2 = corrector.extract_features(reference_image)
        print(f"  倾斜图: {len(kp1)} 个特征点")
        print(f"  参考图: {len(kp2)} 个特征点")
        
        # 步骤2: 特征匹配
        print("\n步骤2: 匹配特征点...")
        matches = corrector.match_features(desc1, desc2)
        print(f"  匹配点: {len(matches)} 个")
        
        # 步骤3: 计算单应性矩阵
        print("\n步骤3: 计算单应性矩阵...")
        H, mask = corrector.compute_homography(kp1, kp2, matches)
        inliers = np.sum(mask)
        print(f"  内点: {inliers}/{len(matches)}")
        
        # 步骤4: 提取角度
        print("\n步骤4: 提取倾斜角度...")
        pitch, roll = corrector.extract_tilt_angles(H, tilted_image.shape[:2])
        print(f"  俯仰角: {pitch:.2f}°")
        print(f"  翻滚角: {roll:.2f}°")
        
        # 步骤5: 构建校正矩阵
        print("\n步骤5: 构建视角校正变换...")
        H_correct = corrector.build_perspective_correction_matrix(
            pitch, roll, tilted_image.shape[:2]
        )
        
        # 步骤6: 应用变换
        print("\n步骤6: 应用变换...")
        corrected_image = corrector.apply_correction(tilted_image, H_correct)
        
        # 保存结果
        output_path = "corrected_output_stepwise.jpg"
        cv2.imwrite(output_path, corrected_image)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("倾斜无人机视图转正射视图校正 - 使用示例")
    print("=" * 50)
    
    # 运行示例
    example_usage()
    # example_with_orb()
    # example_step_by_step()
    
    print("\n" + "=" * 50)
    print("提示: 请将示例中的图像路径替换为你的实际图像路径")
    print("=" * 50)
