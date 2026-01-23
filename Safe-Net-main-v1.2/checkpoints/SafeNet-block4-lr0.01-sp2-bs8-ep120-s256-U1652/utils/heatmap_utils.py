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

def generate_heatmap(model, img_path, opt):
    """生成热力图并与原图叠加"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()

    # 注册钩子（目标层为FAM输出的aligned_feature_map）
    target_layer = model.aligned_feature_map  # 对应model.py中定义的特征图
    hooks = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_backward_hook(backward_hook)
    ]

    # 前向传播
    model.eval()
    with torch.no_grad():
        # 根据模型输入适配输入（根据模式选择视图）
        if opt.mode == 1:
            # drone->satellite：输入为卫星图（query_satellite）
            outputs, _ = model(img_tensor, None)
        else:
            # satellite->drone：输入为无人机图（query_drone）
            _, outputs = model(None, img_tensor)

    # 反向传播（针对预测类别计算梯度）
    model.zero_grad()
    if isinstance(outputs, list):
        cls_output = outputs[0]  # 多分支取第一个分类输出
    else:
        cls_output = outputs
    class_idx = torch.argmax(cls_output, dim=1)
    cls_output[0, class_idx].backward()

    # 计算Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1).squeeze()
    cam = torch.relu(cam)

    # 上采样至原图尺寸
    cam = cv2.resize(cam.cpu().numpy(), (img.size[0], img.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 叠加热力图
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)

    # 保存结果
    save_path = f"heatmap_{os.path.basename(img_path)}"
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Heatmap")
    plt.savefig(save_path)
    print(f"热力图已保存至: {save_path}")

    # 移除钩子
    for hook in hooks:
        hook.remove()
    return save_path