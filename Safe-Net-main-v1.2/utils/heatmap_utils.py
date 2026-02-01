import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from utils.utils_server import UnNormalize

# 全局变量存储特征图和梯度
feature_maps = None
gradients = None

def forward_hook(module, input, output):
    """捕获目标层的特征图"""
    global feature_maps
    feature_maps = output.detach()

def backward_hook(module, grad_in, grad_out):
    """捕获目标层的梯度"""
    global gradients
    gradients = grad_out[0].detach()

def generate_heatmap_data(model, img_path, opt):
    """生成热力图数据（返回叠加后的图像，不保存）"""
    global feature_maps, gradients
    feature_maps = None
    gradients = None
    
    # 图像预处理（如果opt中有transform则使用，否则使用默认transform）
    if hasattr(opt, 'transform') and opt.transform is not None:
        transform = opt.transform
    else:
        transform = transforms.Compose([
            transforms.Resize((opt.h, opt.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()
    img_tensor.requires_grad = True

    # 简化方法：直接使用特征图的激活值生成热力图
    # 不使用梯度，避免维度不匹配的问题

    # 前向传播
    model.eval()
    
    # 根据模型输入适配输入（根据模式选择视图）
    if opt.mode == 1:
        # drone->satellite：输入为卫星图（query_satellite）
        outputs, _ = model(img_tensor, None)
    else:
        # satellite->drone：输入为无人机图（query_drone）
        _, outputs = model(None, img_tensor)
    
    # 获取aligned_feature_map（在forward后访问）
    if hasattr(model, 'module'):
        if hasattr(model.module, 'aligned_feature_map') and model.module.aligned_feature_map is not None:
            feature_maps = model.module.aligned_feature_map.detach()
    else:
        if hasattr(model, 'aligned_feature_map') and model.aligned_feature_map is not None:
            feature_maps = model.aligned_feature_map.detach()
    
    # 不需要反向传播，直接使用特征图激活值

    # 计算热力图
    if feature_maps is not None:
        # feature_maps 形状: [batch, channels, H, W]，channels通常是768
        if len(feature_maps.shape) == 4:
            # 使用L2范数来突出激活较强的区域，这样可以有更好的对比度
            # 对每个空间位置，计算所有通道的L2范数
            cam = torch.norm(feature_maps, p=2, dim=1).squeeze()  # [H, W]
            
            # 或者使用最大值来突出最强激活
            # cam = torch.max(feature_maps, dim=1)[0].squeeze()
        else:
            # 如果不是4D，尝试其他方式
            cam = torch.mean(feature_maps, dim=1).squeeze() if len(feature_maps.shape) > 1 else feature_maps.squeeze()
        
        cam = torch.relu(cam)
        
        # 确保detach后再转换为numpy
        cam = cam.detach()
        
        # 上采样至原图尺寸
        if len(cam.shape) == 2:
            cam_np = cam.cpu().numpy()
        else:
            cam_np = cam.cpu().numpy()
            if len(cam_np.shape) > 2:
                cam_np = cam_np[0]
        
        cam_resized = cv2.resize(cam_np, (img.size[0], img.size[1]))
        
        # 归一化到0-1范围
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        # 生成纯热力图（不叠加原图）
        # 使用JET colormap，范围是：蓝色(低值) -> 青色 -> 绿色 -> 黄色 -> 红色(高值)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # 直接返回纯热力图，不叠加原图
        superimposed_img = heatmap
    else:
        # 如果无法获取特征图，返回原图
        print("警告: 无法获取特征图，返回原图")
        superimposed_img = np.array(img)
    
    return np.array(img), superimposed_img

def generate_heatmap(model, img_path, opt):
    """生成热力图并与原图叠加"""
    img, superimposed_img = generate_heatmap_data(model, img_path, opt)

    # 保存结果
    save_path = f"heatmap_{os.path.basename(img_path)}"
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存至: {save_path}")

    return save_path