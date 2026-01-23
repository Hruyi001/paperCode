#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试theta参数的值，看看为什么对齐图没有变化
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
sys.path.append('/opt/Safe-Net-main-v1.2')

# 模拟一个theta值
theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
print("单位矩阵theta:")
print(theta)

# 检查F.affine_grid的行为
# F.affine_grid期望的theta是相对于归一化坐标的
# 它会根据size自动处理归一化

# 测试1: 16x16特征图（模型中的实际使用）
feat_size = (1, 3, 16, 16)
grid_feat = F.affine_grid(theta.unsqueeze(0), feat_size, align_corners=True)
print(f"\n特征图16x16的grid范围:")
print(f"min: {grid_feat.min().item():.4f}, max: {grid_feat.max().item():.4f}")

# 测试2: 256x256图像
img_size = (1, 3, 256, 256)
grid_img = F.affine_grid(theta.unsqueeze(0), img_size, align_corners=True)
print(f"\n图像256x256的grid范围:")
print(f"min: {grid_img.min().item():.4f}, max: {grid_img.max().item():.4f}")

# 测试3: 一个小的旋转
theta_rot = torch.tensor([[0.9, -0.1, 0.0], [0.1, 0.9, 0.0]], dtype=torch.float32)
print(f"\n旋转theta:")
print(theta_rot)
grid_rot = F.affine_grid(theta_rot.unsqueeze(0), img_size, align_corners=True)
print(f"旋转后的grid范围:")
print(f"min: {grid_rot.min().item():.4f}, max: {grid_rot.max().item():.4f}")

# 关键发现：F.affine_grid会根据size自动归一化
# 所以theta矩阵的值应该是相对于归一化坐标的，不需要手动缩放
