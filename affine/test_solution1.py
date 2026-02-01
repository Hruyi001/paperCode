"""
测试方案1的简单示例
"""

import cv2
import numpy as np
from solution1_feature_matching import align_image_with_features, visualize_matches

def test_basic():
    """基本测试"""
    print("=" * 60)
    print("测试1: 基本功能测试")
    print("=" * 60)
    
    # 创建测试图像
    source = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(source, (50, 50), (350, 350), (0, 0, 0), 3)
    cv2.circle(source, (200, 200), 50, (255, 0, 0), -1)
    cv2.putText(source, 'SOURCE', (150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 创建变换后的参考图像
    center = (200, 200)
    M = cv2.getRotationMatrix2D(center, 15, 1.1)
    M[0, 2] += 20
    M[1, 2] += 10
    reference = cv2.warpAffine(source, M, (400, 400))
    
    # 测试对齐
    print("\n使用ORB检测器...")
    result = align_image_with_features(
        source, reference,
        detector_type='ORB',
        show_matches=True
    )
    
    if len(result) == 3:
        aligned, transform, match_info = result
    else:
        aligned, transform = result
        match_info = None
    
    if transform is not None:
        print("✅ 对齐成功！")
        print(f"\n变换矩阵:\n{transform}")
        
        if match_info:
            print(f"\n匹配统计:")
            print(f"  - 匹配点数量: {match_info['match_count']}")
            print(f"  - 内点数量: {match_info['inlier_count']}")
            
            # 可视化匹配
            matches_img = visualize_matches(source, reference, match_info, max_matches=30)
            cv2.imshow('Feature Matches', matches_img)
        
        # 显示对比
        comparison = np.hstack([source, reference, aligned])
        cv2.putText(comparison, 'Source', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, 'Reference', (410, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Aligned', (810, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Comparison', comparison)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ 对齐失败")


def test_different_detectors():
    """测试不同检测器"""
    print("\n" + "=" * 60)
    print("测试2: 不同特征检测器对比")
    print("=" * 60)
    
    # 创建测试图像
    source = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(source, (50, 50), (350, 350), (0, 0, 0), 3)
    cv2.circle(source, (200, 200), 50, (255, 0, 0), -1)
    
    center = (200, 200)
    M = cv2.getRotationMatrix2D(center, 20, 1.0)
    reference = cv2.warpAffine(source, M, (400, 400))
    
    detectors = ['SIFT', 'ORB', 'AKAZE']
    results = {}
    
    for detector in detectors:
        print(f"\n测试 {detector}...")
        try:
            result = align_image_with_features(
                source, reference,
                detector_type=detector,
                show_matches=False
            )
            aligned, transform = result
            
            if transform is not None:
                print(f"  ✅ {detector} 成功")
                results[detector] = aligned
            else:
                print(f"  ❌ {detector} 失败")
        except Exception as e:
            print(f"  ❌ {detector} 出错: {e}")
    
    # 显示所有结果
    if results:
        print("\n显示所有结果...")
        images = [source, reference]
        labels = ['Source', 'Reference']
        
        for detector, img in results.items():
            images.append(img)
            labels.append(f'{detector} Aligned')
        
        # 创建网格显示
        rows = 2
        cols = (len(images) + 1) // 2
        h, w = images[0].shape[:2]
        grid = np.ones((h * rows, w * cols, 3), dtype=np.uint8) * 255
        
        for i, (img, label) in enumerate(zip(images, labels)):
            row = i // cols
            col = i % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
            cv2.putText(grid, label, (col*w+10, row*h+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('All Results', grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("方案1测试程序")
    print("="*60)
    
    # 运行测试
    test_basic()
    test_different_detectors()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
