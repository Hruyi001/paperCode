"""
特征可视化热力图生成脚本

推荐使用位置：
1. pfeat_align (output[0]) - 最能代表DSA方法效果 ⭐⭐⭐⭐⭐
2. part_features (output[-1]) - 基础特征可视化 ⭐⭐⭐⭐
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def generate_heatmap_pfeat_align(model, image_path, device, img_size=384, training_mode=False):
    """
    使用 pfeat_align (output[0]) 生成热力图 - 推荐方法 ⭐⭐⭐⭐⭐
    最能代表DSA方法的核心效果
    
    Args:
        model: 模型
        image_path: 图像路径
        device: 设备
        img_size: 图像尺寸
        training_mode: 是否使用训练模式（训练模式下才能获取pfeat_align）
    """
    if training_mode:
        model.train()  # 训练模式才能获取pfeat_align
    else:
        model.eval()
        print("警告: 评估模式下无法获取pfeat_align，将使用part_features")
        return generate_heatmap_part_features(model, image_path, device, img_size)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    original_img = np.array(img)
    
    with torch.no_grad():
        # 获取模型输出
        output = model(img_tensor)
        
        # 训练模式下: output = [pfeat_align, cls, features, gap_feature, part_features]
        # 获取DSA对齐后的特征 (output[0])
        pfeat_align = output[0]  # (B, 512, H*W)
        
        # 计算特征图的空间尺寸
        # ConvNeXt下采样32倍，所以 H = W = img_size // 32
        H = W = img_size // 32  # 对于384×384输入，H=W=12
        
        # Reshape为空间特征图
        pfeat_align = pfeat_align.view(1, 512, H, W)  # (B, 512, H, W)
        
        # 生成热力图：对通道维度求平均
        heatmap = pfeat_align[0].mean(dim=0).detach().cpu().numpy()  # (H, W)
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 上采样到原图大小
        heatmap_resized = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # 应用colormap
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 叠加到原图
        alpha = 0.5
        blended = cv2.addWeighted(original_img, alpha, heatmap_colored, 1 - alpha, 0)
        
        return heatmap_resized, heatmap_colored, blended, pfeat_align


def generate_heatmap_part_features(model, image_path, device, img_size=384):
    """
    使用 part_features 生成热力图 - 基础特征可视化 ⭐⭐⭐⭐
    展示backbone的原始特征提取能力
    
    注意: 评估模式下 output = [gap_feature, part_features]
          训练模式下 output = [pfeat_align, cls, features, gap_feature, part_features]
    """
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    original_img = np.array(img)
    
    with torch.no_grad():
        # 获取模型输出
        output = model(img_tensor)
        
        # 评估模式下: output = [gap_feature, part_features]
        # 训练模式下: output = [pfeat_align, cls, features, gap_feature, part_features]
        if len(output) == 2:
            # 评估模式
            part_features = output[1]  # (B, 768, H, W)
        else:
            # 训练模式
            part_features = output[-1]  # (B, 768, H, W)
        
        # 生成热力图：对通道维度求平均
        heatmap = part_features[0].mean(dim=0).detach().cpu().numpy()  # (H, W)
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 上采样到原图大小
        H, W = heatmap.shape
        heatmap_resized = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # 应用colormap
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 叠加到原图
        alpha = 0.5
        blended = cv2.addWeighted(original_img, alpha, heatmap_colored, 1 - alpha, 0)
        
        return heatmap_resized, heatmap_colored, blended, part_features


def generate_comparison_heatmap(model, image_path, device, img_size=384, save_path=None, training_mode=False):
    """
    生成对比热力图：同时展示 part_features 和 pfeat_align
    突出DSA模块的改进效果
    
    Args:
        training_mode: 是否使用训练模式（训练模式下才能获取pfeat_align）
    """
    if training_mode:
        model.train()
    else:
        model.eval()
        print("警告: 评估模式下无法获取pfeat_align，将只显示part_features")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    original_img = np.array(img)
    
    with torch.no_grad():
        output = model(img_tensor)
        
        # 1. 获取 part_features (原始特征)
        if len(output) == 2:
            # 评估模式
            part_features = output[1]  # (B, 768, H, W)
        else:
            # 训练模式
            part_features = output[-1]  # (B, 768, H, W)
        
        heatmap_part = part_features[0].mean(dim=0).detach().cpu().numpy()
        heatmap_part = (heatmap_part - heatmap_part.min()) / (heatmap_part.max() - heatmap_part.min() + 1e-8)
        heatmap_part = cv2.resize(heatmap_part, (img_size, img_size))
        heatmap_part_colored = cv2.applyColorMap((heatmap_part * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended_part = cv2.addWeighted(original_img, 0.5, heatmap_part_colored, 0.5, 0)
        
        # 2. 获取 pfeat_align (DSA处理后的特征) - 仅在训练模式下可用
        if training_mode and len(output) > 2:
            pfeat_align = output[0]  # (B, 512, H*W)
            H = W = img_size // 32
            pfeat_align = pfeat_align.view(1, 512, H, W)
            heatmap_dsa = pfeat_align[0].mean(dim=0).detach().cpu().numpy()
            heatmap_dsa = (heatmap_dsa - heatmap_dsa.min()) / (heatmap_dsa.max() - heatmap_dsa.min() + 1e-8)
            heatmap_dsa = cv2.resize(heatmap_dsa, (img_size, img_size))
            heatmap_dsa_colored = cv2.applyColorMap((heatmap_dsa * 255).astype(np.uint8), cv2.COLORMAP_JET)
            blended_dsa = cv2.addWeighted(original_img, 0.5, heatmap_dsa_colored, 0.5, 0)
            
            # 创建对比图（包含DSA）
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # 第一行：原始特征
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Original Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(blended_part)
            axes[0, 1].set_title('Backbone Features (part_features)', fontsize=12)
            axes[0, 1].axis('off')
            
            # 第二行：DSA处理后的特征
            axes[1, 0].imshow(heatmap_dsa_colored)
            axes[1, 0].set_title('DSA Heatmap (pfeat_align)', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(blended_dsa)
            axes[1, 1].set_title('DSA Features Overlay (pfeat_align)', fontsize=12)
            axes[1, 1].axis('off')
        else:
            # 只显示part_features
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(blended_part)
            axes[1].set_title('Backbone Features (part_features)', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比热力图已保存到: {save_path}")
        
        plt.show()
        
        if training_mode and len(output) > 2:
            return blended_part, blended_dsa
        else:
            return blended_part, None


def visualize_attention_weights(model, image_path, device, img_size=384):
    """
    可视化DSA模块中的注意力权重
    展示模型关注的重点区域
    """
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    original_img = np.array(img)
    
    # 注意：这需要在模型forward中hook来获取中间注意力权重
    # 这里提供一个框架，实际使用时需要修改模型代码添加hook
    print("注意：可视化注意力权重需要在模型forward中添加hook来获取中间变量W")
    print("可以在 build_convnext.forward() 中保存 W 的值")


if __name__ == "__main__":
    """
    使用示例
    """
    import sys
    from sample4geo.hand_convnext.model import make_model
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 384
    
    # 创建模型（需要根据实际配置调整）
    class Config:
        nclasses = 701
        block = 2
        triplet_loss = True
        resnet = False
    
    config = Config()
    model = make_model(config)
    
    # 加载权重（如果有）
    # checkpoint = torch.load('path/to/checkpoint.pth')
    # model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    
    # 生成热力图
    image_path = "path/to/your/image.jpg"  # 替换为实际图像路径
    
    print("=" * 50)
    print("方法1: 使用 pfeat_align (推荐 - 最能代表DSA效果)")
    print("注意: 需要设置 training_mode=True 才能获取pfeat_align")
    print("=" * 50)
    heatmap, colored, blended, features = generate_heatmap_pfeat_align(
        model, image_path, device, img_size, training_mode=True
    )
    cv2.imwrite("heatmap_pfeat_align.jpg", blended)
    
    print("\n" + "=" * 50)
    print("方法2: 使用 part_features (基础特征)")
    print("=" * 50)
    heatmap, colored, blended, features = generate_heatmap_part_features(
        model, image_path, device, img_size
    )
    cv2.imwrite("heatmap_part_features.jpg", blended)
    
    print("\n" + "=" * 50)
    print("方法3: 生成对比热力图")
    print("注意: 需要设置 training_mode=True 才能获取pfeat_align进行对比")
    print("=" * 50)
    generate_comparison_heatmap(
        model, image_path, device, img_size, 
        save_path="comparison_heatmap.png",
        training_mode=True
    )
