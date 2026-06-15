#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试批量处理脚本的图像匹配逻辑
"""

import os
import sys

def find_matching_satellite_image(drone_img_path, satellite_dir):
    """测试图像匹配函数"""
    drone_dir = os.path.dirname(drone_img_path)
    location_id = os.path.basename(drone_dir)
    
    satellite_path = os.path.join(satellite_dir, location_id, f"{location_id}.jpg")
    
    if os.path.exists(satellite_path):
        return satellite_path
    
    return None

# 测试用例
test_cases = [
    ("/datasets/University-Release/test/query_drone/0001/image-06.jpeg",
     "/datasets/University-Release/test/gallery_satellite",
     "/datasets/University-Release/test/gallery_satellite/0001/0001.jpg"),
    ("/datasets/University-Release/test/query_drone/1483/image-51.jpeg",
     "/datasets/University-Release/test/gallery_satellite",
     "/datasets/University-Release/test/gallery_satellite/1483/1483.jpg"),
]

print("测试图像匹配逻辑:")
print("=" * 60)

all_passed = True
for drone_path, satellite_dir, expected_path in test_cases:
    result = find_matching_satellite_image(drone_path, satellite_dir)
    passed = (result == expected_path and os.path.exists(result))
    status = "✓" if passed else "✗"
    
    print(f"{status} Drone: {os.path.basename(drone_path)}")
    print(f"  期望: {expected_path}")
    print(f"  实际: {result}")
    print(f"  存在: {os.path.exists(result) if result else False}")
    
    if not passed:
        all_passed = False
    print()

if all_passed:
    print("=" * 60)
    print("所有测试通过！匹配逻辑正确。")
    sys.exit(0)
else:
    print("=" * 60)
    print("部分测试失败，请检查匹配逻辑。")
    sys.exit(1)
