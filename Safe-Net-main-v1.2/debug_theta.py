#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：检查theta参数的实际值，看看为什么对齐图没有变化
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/opt/Safe-Net-main-v1.2')

# 测试1: 单位矩阵（无变换）
theta_identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
print("=" * 60)
print("测试1: 单位矩阵（应该无变化）")
print("=" * 60)
print(f"Theta:\n{theta_identity}")

# 测试2: 小的旋转
theta_rot = torch.tensor([[0.95, -0.05, 0.0], [0.05, 0.95, 0.0]], dtype=torch.float32)
print("\n" + "=" * 60)
print("测试2: 小的旋转（应该有轻微变化）")
print("=" * 60)
print(f"Theta:\n{theta_rot}")
angle = np.arctan2(theta_rot[1, 0].item(), theta_rot[0, 0].item()) * 180 / np.pi
scale = np.sqrt(theta_rot[0, 0].item()**2 + theta_rot[1, 0].item()**2)
print(f"旋转角度: {angle:.2f}°")
print(f"缩放: {scale:.4f}")

# 测试3: 小的平移
theta_trans = torch.tensor([[1.0, 0.0, 0.1], [0.0, 1.0, 0.1]], dtype=torch.float32)
print("\n" + "=" * 60)
print("测试3: 小的平移（应该有轻微变化）")
print("=" * 60)
print(f"Theta:\n{theta_trans}")
print(f"平移: ({theta_trans[0, 2].item():.4f}, {theta_trans[1, 2].item():.4f})")

# 测试4: 检查F.affine_grid的行为
print("\n" + "=" * 60)
print("测试4: F.affine_grid的行为")
print("=" * 60)

# 16x16特征图（模型中的实际使用）
feat_size = (1, 3, 16, 16)
grid_feat = F.affine_grid(theta_identity.unsqueeze(0), feat_size, align_corners=True)
print(f"16x16特征图的grid范围: [{grid_feat.min().item():.4f}, {grid_feat.max().item():.4f}]")
print(f"16x16特征图的grid中心: {grid_feat[0, 8, 8, :].tolist()}")

# 256x256图像
img_size = (1, 3, 256, 256)
grid_img = F.affine_grid(theta_identity.unsqueeze(0), img_size, align_corners=True)
print(f"256x256图像的grid范围: [{grid_img.min().item():.4f}, {grid_img.max().item():.4f}]")
print(f"256x256图像的grid中心: {grid_img[0, 128, 128, :].tolist()}")

# 关键发现
print("\n" + "=" * 60)
print("关键发现:")
print("=" * 60)
print("1. F.affine_grid会自动归一化坐标到[-1, 1]范围")
print("2. 对于单位矩阵，grid的值应该是[-1, 1]的线性映射")
print("3. 如果theta接近单位矩阵，变换效果会很小")
print("4. 如果模型输出的theta接近单位矩阵，说明图像已经对齐或变换很小")
