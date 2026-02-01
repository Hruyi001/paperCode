"""
方案1: 基于特征点匹配的自动对齐
使用SIFT/ORB/AKAZE特征检测器找到特征点，通过匹配计算仿射变换矩阵
适用于有参考图像的情况

使用方法:
    python solution1_feature_matching.py source.jpg reference.jpg
    或
    from solution1_feature_matching import align_image_with_features
    aligned, transform = align_image_with_features(source_img, reference_img)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import argparse
import os
from datetime import datetime


def align_image_with_features(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    detector_type: str = 'SIFT',
    min_match_count: int = 4,
    show_matches: bool = False,
    ratio_threshold: float = 0.75
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[dict]]:
    """
    使用特征点匹配对齐图像
    
    Args:
        source_img: 需要对齐的源图像
        reference_img: 参考图像
        detector_type: 特征检测器类型 ('SIFT', 'ORB', 或 'AKAZE')
        min_match_count: 最少匹配点数量
        show_matches: 是否返回匹配信息用于可视化
        ratio_threshold: Lowe's ratio test的阈值（仅用于SIFT）
    
    Returns:
        aligned_img: 对齐后的图像
        transform_matrix: 仿射变换矩阵 (3x3)，如果失败返回None
        match_info: 匹配信息字典（如果show_matches=True），包含:
            - keypoints1: 源图像特征点
            - keypoints2: 参考图像特征点
            - matches: 匹配点对
            - inliers: 内点掩码
    """
    # 转换为灰度图
    if len(source_img.shape) == 3:
        gray1 = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = source_img.copy()
    
    if len(reference_img.shape) == 3:
        gray2 = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = reference_img.copy()
    
    # 选择特征检测器
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=5000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif detector_type == 'AKAZE':
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError("detector_type must be 'SIFT', 'ORB', or 'AKAZE'")
    
    # 检测特征点和描述符
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        print(f"警告: 无法检测到特征点 (源图像: {len(kp1) if kp1 else 0}, 参考图像: {len(kp2) if kp2 else 0})")
        return source_img, None, None if show_matches else (source_img, None)
    
    if len(des1) < min_match_count or len(des2) < min_match_count:
        print(f"警告: 特征点不足 (源图像: {len(des1)}, 参考图像: {len(des2)}, 需要: {min_match_count})")
        return source_img, None, None if show_matches else (source_img, None)
    
    # 匹配特征点
    if detector_type == 'SIFT':
        matches = matcher.knnMatch(des1, des2, k=2)
        # 使用Lowe's ratio test筛选好的匹配
        good_matches = []
        all_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                all_matches.append(m)
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
    else:  # ORB 或 AKAZE
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        # 取前N个最佳匹配，但至少需要min_match_count个
        num_matches = max(min_match_count * 2, min(100, len(matches)))
        good_matches = matches[:num_matches]
        all_matches = matches
    
    if len(good_matches) < min_match_count:
        print(f"警告: 匹配点不足 ({len(good_matches)}/{min_match_count})")
        return source_img, None, None if show_matches else (source_img, None)
    
    print(f"找到 {len(good_matches)} 个良好匹配点")
    
    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC计算仿射变换矩阵
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, 
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.99
    )
    
    if transform_matrix is None:
        print("错误: 无法估计仿射变换矩阵")
        return source_img, None, None if show_matches else (source_img, None)
    
    # 统计内点数量
    inlier_count = np.sum(inliers) if inliers is not None else len(good_matches)
    print(f"RANSAC内点数量: {inlier_count}/{len(good_matches)}")
    
    # 应用仿射变换
    h, w = reference_img.shape[:2]
    aligned_img = cv2.warpAffine(
        source_img, transform_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # 转换为3x3矩阵（齐次坐标）
    transform_3x3 = np.vstack([transform_matrix, [0, 0, 1]])
    
    # 准备返回信息
    if show_matches:
        match_info = {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'matches': good_matches,
            'inliers': inliers,
            'match_count': len(good_matches),
            'inlier_count': inlier_count
        }
        return aligned_img, transform_3x3, match_info
    else:
        return aligned_img, transform_3x3


def visualize_matches(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    match_info: dict,
    max_matches: int = 50
) -> np.ndarray:
    """
    可视化特征点匹配结果
    
    Args:
        source_img: 源图像
        reference_img: 参考图像
        match_info: 匹配信息字典
        max_matches: 最多显示的匹配点数量
    
    Returns:
        vis_img: 可视化图像
    """
    kp1 = match_info['keypoints1']
    kp2 = match_info['keypoints2']
    matches = match_info['matches']
    inliers = match_info.get('inliers', None)
    
    # 只显示内点匹配
    if inliers is not None:
        inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
        matches_to_show = inlier_matches[:max_matches]
    else:
        matches_to_show = matches[:max_matches]
    
    # 绘制匹配结果
    vis_img = cv2.drawMatches(
        source_img, kp1,
        reference_img, kp2,
        matches_to_show, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return vis_img


def main():
    """主函数：处理命令行参数"""
    parser = argparse.ArgumentParser(
        description='基于特征点匹配的图像对齐工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用ORB检测器对齐图像
  python solution1_feature_matching.py source.jpg reference.jpg --detector ORB
  
  # 使用SIFT检测器并显示匹配点
  python solution1_feature_matching.py source.jpg reference.jpg --detector SIFT --show-matches
  
  # 保存结果到文件
  python solution1_feature_matching.py source.jpg reference.jpg -o aligned.jpg
        """
    )
    
    parser.add_argument('source', help='源图像路径（需要对齐的图像）')
    parser.add_argument('reference', help='参考图像路径（对齐的目标）')
    parser.add_argument('-d', '--detector', 
                       choices=['SIFT', 'ORB', 'AKAZE'],
                       default='SIFT',
                       help='特征检测器类型 (默认: SIFT)')
    parser.add_argument('-o', '--output',
                       help='输出图像路径（可选）')
    parser.add_argument('--show-matches', action='store_true',
                       help='显示特征点匹配结果')
    parser.add_argument('--min-matches', type=int, default=4,
                       help='最少匹配点数量 (默认: 4)')
    parser.add_argument('--save-comparison', action='store_true',
                       help='保存对比图到文件（即使无GUI也保存）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示图像窗口（适用于无GUI环境）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.source):
        print(f"错误: 源图像文件不存在: {args.source}")
        return
    
    if not os.path.exists(args.reference):
        print(f"错误: 参考图像文件不存在: {args.reference}")
        return
    
    # 读取图像
    print(f"读取源图像: {args.source}")
    source = cv2.imread(args.source)
    if source is None:
        print(f"错误: 无法读取源图像: {args.source}")
        return
    
    print(f"读取参考图像: {args.reference}")
    reference = cv2.imread(args.reference)
    if reference is None:
        print(f"错误: 无法读取参考图像: {args.reference}")
        return
    
    print(f"\n使用 {args.detector} 特征检测器进行对齐...")
    print("-" * 60)
    
    # 对齐图像
    result = align_image_with_features(
        source, reference,
        detector_type=args.detector,
        min_match_count=args.min_matches,
        show_matches=args.show_matches
    )
    
    if args.show_matches:
        aligned, transform, match_info = result
        if match_info is None:
            print("对齐失败，无法显示匹配点")
            return
    else:
        aligned, transform = result
        match_info = None
    
    if transform is None:
        print("\n❌ 对齐失败！")
        print("可能的原因:")
        print("  - 两张图像差异太大")
        print("  - 特征点不足")
        print("  - 尝试使用不同的检测器 (--detector ORB 或 --detector AKAZE)")
        return
    
    print("\n✅ 对齐成功！")
    print("\n变换矩阵:")
    print(transform)
    
    # 保存结果
    if args.output:
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 检查输出路径是否有效
        if os.path.isdir(args.output):
            # 如果是目录，生成文件名
            source_basename = os.path.splitext(os.path.basename(args.source))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = os.path.join(args.output, f"aligned_{source_basename}_{timestamp}.jpg")
        
        # 确保有文件扩展名
        if not os.path.splitext(args.output)[1]:
            args.output = args.output + ".jpg"
        
        # 保存图像
        success = cv2.imwrite(args.output, aligned)
        if success:
            print(f"\n结果已保存到: {args.output}")
        else:
            print(f"\n警告: 保存图像失败，路径: {args.output}")
            print("可能的原因:")
            print("  - 路径无效或没有写权限")
            print("  - 文件扩展名不支持")
            print("  - 磁盘空间不足")
    
    # 创建对比图像（无论是否有GUI都创建，用于保存）
    h1, w1 = source.shape[:2]
    h2, w2 = reference.shape[:2]
    h3, w3 = aligned.shape[:2]
    
    max_h = max(h1, h2, h3)
    total_w = w1 + w2 + w3
    
    comparison = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    comparison[:h1, :w1] = source
    comparison[:h2, w1:w1+w2] = reference
    comparison[:h3, w1+w2:w1+w2+w3] = aligned
    
    # 添加标签
    font_scale = min(w1, w2, w3) / 400.0
    thickness = max(1, int(font_scale * 2))
    cv2.putText(comparison, 'Source', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    cv2.putText(comparison, 'Reference', (w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv2.putText(comparison, 'Aligned', (w1 + w2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    
    # 保存对比图（如果请求）
    if args.save_comparison or args.output:
        if args.output:
            # 在输出文件同目录保存对比图
            output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
            output_basename = os.path.splitext(os.path.basename(args.output))[0]
            comparison_path = os.path.join(output_dir, f"comparison_{output_basename}.jpg")
        else:
            # 保存到当前目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = f"comparison_{timestamp}.jpg"
        
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图已保存到: {comparison_path}")
    
    # 检查是否有GUI支持
    has_gui = False
    if not args.no_display:
        try:
            # 尝试创建一个测试窗口来检测GUI是否可用
            test_img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.destroyAllWindows()
            has_gui = True
        except:
            has_gui = False
    
    # 显示结果（仅在GUI可用且未禁用显示时）
    if has_gui and not args.no_display:
        print("\n显示结果窗口（按任意键关闭）...")
        
        if args.show_matches and match_info:
            matches_img = visualize_matches(source, reference, match_info)
            try:
                cv2.imshow('Feature Matches', matches_img)
            except:
                print("警告: 无法显示匹配点图像（GUI不可用）")
        
        try:
            cv2.imshow('Comparison: Source | Reference | Aligned', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("警告: 无法显示对比图像（GUI不可用）")
    else:
        if args.no_display:
            print("\n注意: 已禁用图像显示（--no-display）")
        else:
            print("\n注意: 当前环境无GUI支持，跳过图像显示")
        print("对齐结果已保存到文件，可以使用图像查看器查看")


def demo():
    """演示函数：使用示例图像"""
    print("=" * 60)
    print("方案1演示: 基于特征点匹配的自动对齐")
    print("=" * 60)
    
    # 创建示例图像
    print("\n创建测试图像...")
    source = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(source, (50, 50), (350, 350), (0, 0, 0), 3)
    cv2.circle(source, (200, 200), 50, (255, 0, 0), -1)
    cv2.putText(source, 'SOURCE', (150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 创建旋转、缩放、平移后的参考图像
    center = (200, 200)
    M = cv2.getRotationMatrix2D(center, 15, 1.1)
    M[0, 2] += 20
    M[1, 2] += 10
    reference = cv2.warpAffine(source, M, (400, 400))
    
    print("测试不同特征检测器...")
    detectors = ['SIFT', 'ORB', 'AKAZE']
    
    for detector in detectors:
        print(f"\n{'='*60}")
        print(f"使用 {detector} 检测器")
        print(f"{'='*60}")
        
        try:
            result = align_image_with_features(
                source, reference, 
                detector_type=detector,
                show_matches=True
            )
            
            if len(result) == 3:
                aligned, transform, match_info = result
            else:
                aligned, transform = result
                match_info = None
            
            if transform is not None:
                print(f"✅ {detector} 对齐成功")
                if match_info:
                    print(f"   匹配点: {match_info['match_count']}")
                    print(f"   内点: {match_info['inlier_count']}")
                
                # 显示结果
                if match_info:
                    matches_img = visualize_matches(source, reference, match_info, max_matches=20)
                    cv2.imshow(f'{detector} Matches', matches_img)
                
                comparison = np.hstack([source, reference, aligned])
                cv2.imshow(f'{detector} Result', comparison)
                cv2.waitKey(1000)  # 等待1秒
            else:
                print(f"❌ {detector} 对齐失败")
        except Exception as e:
            print(f"❌ {detector} 出错: {e}")
    
    print("\n" + "="*60)
    print("演示完成！按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 演示模式
        demo()
