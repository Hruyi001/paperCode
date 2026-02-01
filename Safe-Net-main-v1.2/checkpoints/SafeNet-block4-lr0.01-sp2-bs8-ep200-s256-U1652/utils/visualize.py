"""
可视化工具函数
用于生成Aligned图、Partition图和Heatmap图
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import math


def align_image(original_img, theta, img_size=256, debug=False):
    """
    根据FAM模块输出的theta参数，对原始图像应用affine变换生成对齐图
    
    Args:
        original_img: PIL Image或numpy array，原始图像
        theta: torch.Tensor，形状为[2, 3]的affine变换矩阵
        img_size: int，目标图像尺寸
        debug: bool，是否打印调试信息
    
    Returns:
        aligned_img: numpy array，对齐后的图像
    
    注意：如果theta接近单位矩阵，变换效果会很小，图像看起来几乎不变。
    这是正常的，说明图像已经对齐或不需要大的变换。
    """
    # 转换为numpy数组
    if isinstance(original_img, Image.Image):
        img_np = np.array(original_img)
    else:
        img_np = original_img.copy()
    
    h_orig, w_orig = img_np.shape[:2]
    
    # 确保theta是torch tensor
    if isinstance(theta, np.ndarray):
        theta = torch.from_numpy(theta).float()
    elif not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    
    # 如果theta是[batch, 2, 3]格式，取第一个batch
    if len(theta.shape) == 3:
        theta = theta[0]
    
    # 将theta转换为2x3矩阵（如果已经是2x3则直接使用）
    if theta.shape != (2, 3):
        theta = theta.view(2, 3)
    
    # 关键理解：F.affine_grid期望的theta矩阵是相对于归一化坐标空间（-1到1）的
    # 它会根据输入tensor的size自动处理归一化
    # 
    # 在模型中：F.affine_grid(A_theta, patch_features.size()) 对16x16特征图
    # 在可视化中：F.affine_grid(theta, img_tensor.size()) 对256x256图像
    # 
    # 由于F.affine_grid会自动归一化，所以可以直接使用theta
    # 但需要确保图像尺寸与模型训练时的尺寸一致（通常是256x256）
    
    # 如果原图尺寸不是img_size，先resize到img_size
    if h_orig != img_size or w_orig != img_size:
        img_pil = Image.fromarray(img_np).resize((img_size, img_size), Image.LANCZOS)
        img_np = np.array(img_pil)
        h_orig, w_orig = img_np.shape[:2]
    
    # 转换为PIL Image
    img_pil = Image.fromarray(img_np)
    
    # 转换为tensor
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)  # [1, 3, H, W]
    
    # 直接使用theta创建affine grid
    # F.affine_grid会根据img_tensor.size()自动处理归一化
    # theta矩阵中的平移参数已经是相对于归一化坐标的
    
    # 调试：打印theta值和变换参数
    if debug:
        print(f"\nTheta矩阵:\n{theta}")
        angle = math.atan2(theta[1, 0].item(), theta[0, 0].item()) * 180 / math.pi
        scale = math.sqrt(theta[0, 0].item()**2 + theta[1, 0].item()**2)
        print(f"旋转角度: {angle:.2f}°")
        print(f"缩放: {scale:.4f}")
        print(f"平移: ({theta[0, 2].item():.4f}, {theta[1, 2].item():.4f})")
        
        # 检查是否接近单位矩阵
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=theta.dtype)
        diff = torch.abs(theta - identity).sum().item()
        print(f"与单位矩阵的差异: {diff:.6f}")
        if diff < 0.01:
            print("⚠ 警告: theta接近单位矩阵，变换效果会很小，图像看起来几乎不变")
            print("   这是正常的，说明图像已经对齐或不需要大的变换")
    
    grid = F.affine_grid(theta.unsqueeze(0), img_tensor.size(), align_corners=True)
    
    # 应用grid sample
    aligned_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # 转换回numpy
    # ToTensor()将PIL Image (RGB)转换为tensor，顺序保持RGB
    # grid_sample输出后，经过permute(1,2,0)，通道顺序仍然是RGB
    aligned_img = aligned_tensor[0].permute(1, 2, 0).cpu().numpy()
    aligned_img = (aligned_img * 255).astype(np.uint8)
    
    # 确保输出是RGB格式（PIL Image和matplotlib使用RGB）
    # 不需要颜色空间转换，因为tensor已经是RGB顺序
    return aligned_img


def draw_partition(aligned_img, boundaries, img_size=256):
    """
    在Aligned图上绘制FPM模块输出的分区边界
    
    Args:
        aligned_img: numpy array，对齐后的图像
        boundaries: list，分区边界列表，如[b1, b2, b3]（基于16x16特征图）
        img_size: int，图像尺寸
    
    Returns:
        partition_img: numpy array，带分区边界的图像
    """
    # 创建图像副本
    partition_img = aligned_img.copy()
    
    if isinstance(partition_img, Image.Image):
        partition_img = np.array(partition_img)
    
    h_orig, w_orig = partition_img.shape[:2]
    
    # 如果boundaries是list of lists（batch），取第一个
    if isinstance(boundaries, list) and len(boundaries) > 0:
        if isinstance(boundaries[0], list):
            boundaries = boundaries[0]
        elif isinstance(boundaries[0], (int, float, torch.Tensor)):
            # 已经是单个batch的boundaries
            boundaries = [int(b.item()) if isinstance(b, torch.Tensor) else int(b) for b in boundaries]
    
    # 转换为列表
    if isinstance(boundaries, torch.Tensor):
        boundaries = boundaries.tolist()
    
    # 特征图尺寸（16x16）
    feat_size = 16
    scale_factor = min(h_orig, w_orig) / feat_size
    
    # 图像中心
    center_h, center_w = h_orig // 2, w_orig // 2
    
    # 绘制所有边界框（从内到外）
    for i, boundary in enumerate(boundaries):
        # 边界值（基于特征图空间）
        boundary_pixels = int(boundary * scale_factor)
        
        # 计算矩形框的左上角和右下角
        x1 = center_w - boundary_pixels
        y1 = center_h - boundary_pixels
        x2 = center_w + boundary_pixels
        y2 = center_h + boundary_pixels
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w_orig))
        y1 = max(0, min(y1, h_orig))
        x2 = max(0, min(x2, w_orig))
        y2 = max(0, min(y2, h_orig))
        
        # 绘制矩形框（使用浅蓝色虚线）
        # OpenCV不支持虚线，所以我们用点线模拟
        color = (173, 216, 230)  # 浅蓝色 (BGR格式，因为cv2使用BGR)
        
        # 绘制矩形（实线）
        cv2.rectangle(partition_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 为了更好的可视化，我们也可以绘制虚线效果
        # 使用matplotlib来绘制虚线矩形
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(w_orig/100, h_orig/100), dpi=100)
            ax.imshow(cv2.cvtColor(partition_img, cv2.COLOR_BGR2RGB) if len(partition_img.shape) == 3 else partition_img)
            
            # 绘制所有边界框（虚线）
            for j, b in enumerate(boundaries):
                bp = int(b * scale_factor)
                rect = patches.Rectangle(
                    (center_w - bp, center_h - bp),
                    2 * bp, 2 * bp,
                    linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
            
            ax.axis('off')
            ax.set_xlim(0, w_orig)
            ax.set_ylim(h_orig, 0)
            
            # 转换为numpy数组
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            partition_img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return partition_img
        except:
            # 如果matplotlib不可用，返回OpenCV绘制的版本
            pass
    
    # 如果matplotlib绘制失败，返回OpenCV版本
    if len(partition_img.shape) == 3:
        partition_img = cv2.cvtColor(partition_img, cv2.COLOR_BGR2RGB)
    
    return partition_img


def generate_heatmap(f_p_aligned, original_img_size=(256, 256)):
    """
    根据FAM输出的对齐后特征图生成热力图
    
    Args:
        f_p_aligned: torch.Tensor，形状为[C, H, W]或[B, C, H, W]的对齐后特征图
        original_img_size: tuple，原始图像尺寸 (height, width)
    
    Returns:
        heatmap: numpy array，热力图（RGB格式）
    """
    # 确保是torch tensor
    if isinstance(f_p_aligned, np.ndarray):
        f_p_aligned = torch.from_numpy(f_p_aligned).float()
    
    # 如果是4D tensor [B, C, H, W]，取第一个batch
    if len(f_p_aligned.shape) == 4:
        f_p_aligned = f_p_aligned[0]
    
    # f_p_aligned形状应该是 [C, H, W]
    # 沿通道维度求和，得到 [H, W] 的单通道特征图
    heatmap = f_p_aligned.sum(dim=0)  # [H, W]
    
    # 转换为numpy
    heatmap_np = heatmap.detach().cpu().numpy()
    
    # 归一化到0-255
    heatmap_min = heatmap_np.min()
    heatmap_max = heatmap_np.max()
    if heatmap_max > heatmap_min:
        heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min) * 255
    else:
        heatmap_np = np.zeros_like(heatmap_np)
    
    # 上采样到原始图像尺寸
    h_orig, w_orig = original_img_size
    heatmap_resized = cv2.resize(heatmap_np.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # 应用JET配色方案（红/橙表示高响应，蓝表示低响应）
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # 转换为RGB格式
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb
