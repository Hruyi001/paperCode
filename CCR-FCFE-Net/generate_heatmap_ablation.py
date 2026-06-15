# -*- coding: utf-8 -*-
"""
生成消融实验的注意力热力图
对比CIB处理前后的模型注意力差异

使用方法:
python generate_heatmap_ablation.py \
    --drone_img /path/to/drone/image.jpg \
    --satellite_img /path/to/satellite/image.jpg \
    --name FCFE_Model_University \
    --output_dir ./heatmap_results

注意：模型必须包含CIB组件，通过hook不同位置来展示CIB前后的差异
"""

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import yaml
import cv2
from utils import load_network
from datasets.queryDataset import Query_transforms


class AttentionExtractor:
    """提取模型注意力图的类"""
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.hooks = []
    
    def register_hooks(self, hook_before_cib=False):
        """注册hook来捕获中间层特征
        
        Args:
            hook_before_cib: 如果True，hook到CIB_layer之前；如果False，hook到CIB_layer之后
        """
        def cib_attentions_hook(module, input, output):
            # Hook到CIB内部的attentions模块的输出
            # output形状: [B, M, H, W] - attention maps
            # 注意：CIB的forward会循环调用attentions，我们只保存第一次的结果
            try:
                # 调试信息：打印output的类型和形状
                if len(self.activations) == 0:  # 只在第一次调用时打印，避免过多输出
                    print(f"DEBUG: CIB attentions hook called, output type: {type(output)}, shape: {getattr(output, 'shape', 'N/A')}")
                
                if isinstance(output, torch.Tensor):
                    if len(output.shape) == 4:  # [B, M, H, W]
                        # 对M维度求平均，得到空间注意力图
                        feat = torch.mean(output, dim=1, keepdim=True)  # [B, 1, H, W]
                        # 只保存第一次（如果已经保存过就跳过）
                        if len(self.activations) == 0:
                            self.activations.append(feat.clone().detach())
                            print(f"DEBUG: Saved activation from CIB attentions, shape: {feat.shape}")
                        return
                    elif len(output.shape) == 3:  # [B, H, W] 或其他形状
                        # 尝试处理3维输出
                        feat = output.unsqueeze(1) if len(output.shape) == 3 else output
                        if len(self.activations) == 0:
                            self.activations.append(feat.clone().detach())
                            print(f"DEBUG: Saved activation from CIB attentions (3D), shape: {feat.shape}")
                        return
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    # 如果输出是列表或元组，取第一个元素
                    if isinstance(output[0], torch.Tensor):
                        feat = output[0]
                        if len(feat.shape) >= 3:
                            if len(feat.shape) == 4:
                                feat = torch.mean(feat, dim=1, keepdim=True)
                            elif len(feat.shape) == 3:
                                feat = feat.unsqueeze(1)
                            if len(self.activations) == 0:
                                self.activations.append(feat.clone().detach())
                                print(f"DEBUG: Saved activation from CIB attentions (list/tuple), shape: {feat.shape}")
                            return
            except Exception as e:
                print(f"Warning in CIB attentions hook: {e}")
                import traceback
                traceback.print_exc()
        
        def cib_input_hook(module, input, output):
            # Hook到CIB_layer的输入: ADIB_attention_features [B, C, H, W, block]
            # 这是CIB处理前的特征
            try:
                if input and len(input) > 0:
                    adib_features = input[0]
                    if isinstance(adib_features, torch.Tensor):
                        if len(adib_features.shape) == 5:  # [B, C, H, W, block]
                            # 取第一个block并平均所有通道作为注意力图
                            feat = adib_features[:, :, :, :, 0]  # [B, C, H, W]
                            feat = torch.mean(feat, dim=1, keepdim=True)  # [B, 1, H, W]
                            self.activations.append(feat.clone().detach())
                            return
                        elif len(adib_features.shape) == 4:  # [B, C, H, W]
                            feat = torch.mean(adib_features, dim=1, keepdim=True)  # [B, 1, H, W]
                            self.activations.append(feat.clone().detach())
                            return
            except Exception as e:
                print(f"Warning in CIB input hook: {e}")
        
        def before_cib_hook(module, input, output):
            # ADIB_layer的输出: part_features (list of tensors)
            try:
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    feat = output[0]
                    if isinstance(feat, torch.Tensor):
                        if len(feat.shape) == 4:  # [B, C, H, W]
                            # 计算通道平均作为注意力图
                            feat = torch.mean(feat, dim=1, keepdim=True)  # [B, 1, H, W]
                            self.activations.append(feat.clone().detach())
                            return
                        elif len(feat.shape) == 3:  # [B, H, W]
                            feat = feat.unsqueeze(1)  # [B, 1, H, W]
                            self.activations.append(feat.clone().detach())
                            return
                # 如果输出是Tensor
                elif isinstance(output, torch.Tensor):
                    if len(output.shape) == 4:
                        feat = torch.mean(output, dim=1, keepdim=True)  # [B, 1, H, W]
                        self.activations.append(feat.clone().detach())
                        return
                    elif len(output.shape) == 3:
                        output = output.unsqueeze(1)
                        self.activations.append(output.clone().detach())
                        return
            except Exception as e:
                print(f"Warning in ADIB hook: {e}")
        
        # 查找CIB_layer或ADIB_layer
        target_layer = None
        for name, module in self.model.named_modules():
            if hook_before_cib:
                # Hook到ADIB_layer的输出（CIB之前）
                if 'ADIB_layer' in name or 'ADIB' in name:
                    target_layer = (name, module, before_cib_hook)
                    break
            else:
                # Hook到CIB之后的特征
                # 优先hook到CIB内部的attentions模块（这是CIB处理后的attention maps）
                if 'CIB_layer' in name or 'CIB' in name:
                    # 查找CIB内部的attentions模块
                    cib_module = module
                    for sub_name, sub_module in cib_module.named_modules():
                        if 'attentions' in sub_name.lower() and isinstance(sub_module, nn.Module):
                            # Hook到attentions模块的输出
                            target_layer = (f"{name}.{sub_name}", sub_module, cib_attentions_hook)
                            break
                    
                    # 如果找不到attentions模块，hook到CIB的输入作为备选
                    if target_layer is None:
                        target_layer = (name, module, cib_input_hook)
                    break
        
        if target_layer is not None:
            name, module, hook_func = target_layer
            handle = module.register_forward_hook(hook_func)
            self.hooks.append(handle)
            hook_location = 'before CIB' if hook_before_cib else 'after CIB (attentions)'
            print(f"Registered hook to layer: {name} ({hook_location})")
        else:
            # 如果找不到，尝试hook到backbone的输出
            print(f"Warning: Target layer not found, trying to hook to backbone...")
            for name, module in self.model.named_modules():
                if 'backbone' in name.lower():
                    def backbone_hook(module, input, output):
                        # backbone返回(gap_feature, part_features)
                        if isinstance(output, tuple) and len(output) >= 2:
                            part_features = output[1]
                            if isinstance(part_features, (list, tuple)) and len(part_features) > 0:
                                if len(part_features[0].shape) == 4:
                                    feat = torch.mean(part_features[0], dim=1, keepdim=True)
                                    self.activations.append(feat.clone().detach())
                    handle = module.register_forward_hook(backbone_hook)
                    self.hooks.append(handle)
                    print(f"Registered hook to layer: {name}")
                    break
    
    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def extract_attention(self, input_img, view_index=1, opt=None):
        """提取注意力图"""
        self.activations = []
        outputs = None
        
        self.model.eval()
        with torch.no_grad():
            # 根据view_index调用模型
            # 注意：hook会在前向传播过程中捕获中间特征
            # 即使前向传播失败，只要hook捕获到特征就可以
            try:
                if opt and opt.views == 2:
                    if view_index == 1:
                        outputs, _ = self.model(input_img, None)
                    elif view_index == 3:
                        _, outputs = self.model(None, input_img)
                elif opt and opt.views == 3:
                    if view_index == 1:
                        outputs, _, _ = self.model(input_img, None, None)
                    elif view_index == 2:
                        _, outputs, _ = self.model(None, input_img, None)
                    elif view_index == 3:
                        _, _, outputs = self.model(None, None, input_img)
                else:
                    outputs = self.model(input_img)
            except Exception as e:
                # 前向传播可能失败（比如在分类器部分），但hook应该已经捕获了特征
                # 只要activations不为空就可以
                pass
        
        # 优先使用hook捕获的激活值
        if self.activations:
            features = self.activations[-1]
            if len(features.shape) == 4:  # [B, C, H, W]
                # 计算通道平均作为注意力
                attention = torch.mean(features, dim=1)[0].cpu().numpy()
                # 归一化
                attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
                return attention
        
        # 如果没有捕获到激活值，尝试使用输出特征（如果前向传播成功）
        if outputs is not None and hasattr(outputs, 'shape'):
            if len(outputs.shape) == 4:
                attention = torch.mean(outputs, dim=1)[0].cpu().numpy()
                attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
                return attention
            elif len(outputs.shape) == 3:  # [B, H, W]
                attention = outputs[0].cpu().numpy()
                attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
                return attention
        
        return None


def generate_heatmap_overlay(image, attention_map, alpha=0.5):
    """将热力图叠加到原始图像上"""
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention_map, (w, h))
    
    # 转换为热力图颜色 (jet colormap: blue->green->yellow->red)
    heatmap = cm.jet(attention_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def load_and_preprocess_image(image_path, opt):
    """加载并预处理图像"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # 根据图像类型选择transform
    if 'query' in image_path.lower() or 'drone' in image_path.lower():
        transform = transforms.Compose([
            transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
            Query_transforms(pad=opt.pad, size=opt.w),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img_array


def create_ablation_visualization(drone_img_path, satellite_img_path, 
                                  model, opt, 
                                  output_dir='./heatmap_results', output_filename=None):
    """创建消融实验可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    print(f"Loading images:")
    print(f"  Drone: {drone_img_path}")
    print(f"  Satellite: {satellite_img_path}")
    
    if not os.path.exists(drone_img_path):
        raise FileNotFoundError(f"Drone image not found: {drone_img_path}")
    if not os.path.exists(satellite_img_path):
        raise FileNotFoundError(f"Satellite image not found: {satellite_img_path}")
    
    drone_tensor, drone_img = load_and_preprocess_image(drone_img_path, opt)
    satellite_tensor, satellite_img = load_and_preprocess_image(satellite_img_path, opt)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        drone_tensor = drone_tensor.cuda()
        satellite_tensor = satellite_tensor.cuda()
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    
    # 提取注意力图
    print("\nExtracting attention maps...")
    
    # 使用同一个模型，通过hook不同位置来展示CIB前后的差异
    # Before CIB: Hook到ADIB_layer的输出（CIB之前）
    # After CIB: Hook到CIB_layer内部的attentions模块（CIB处理后的attention maps）
    extractor_before = AttentionExtractor(model)
    extractor_before.register_hooks(hook_before_cib=True)  # Hook到CIB之前（ADIB输出）
    
    extractor_after = AttentionExtractor(model)
    extractor_after.register_hooks(hook_before_cib=False)  # Hook到CIB之后（CIB attentions输出）
    
    # Drone图像 (view_index=3)
    print("  Processing Drone image...")
    try:
        drone_attention_before = extractor_before.extract_attention(drone_tensor, view_index=3, opt=opt)
        print(f"    Before CIB: {'Success' if drone_attention_before is not None else 'Failed'}")
    except Exception as e:
        print(f"    Before CIB: Error - {e}")
        drone_attention_before = None
    
    try:
        drone_attention_after = extractor_after.extract_attention(drone_tensor, view_index=3, opt=opt)
        print(f"    After CIB: {'Success' if drone_attention_after is not None else 'Failed'}")
    except Exception as e:
        print(f"    After CIB: Error - {e}")
        drone_attention_after = None
    
    # Satellite图像 (view_index=1)
    print("  Processing Satellite image...")
    try:
        satellite_attention_before = extractor_before.extract_attention(satellite_tensor, view_index=1, opt=opt)
        print(f"    Before CIB: {'Success' if satellite_attention_before is not None else 'Failed'}")
    except Exception as e:
        print(f"    Before CIB: Error - {e}")
        satellite_attention_before = None
    
    try:
        satellite_attention_after = extractor_after.extract_attention(satellite_tensor, view_index=1, opt=opt)
        print(f"    After CIB: {'Success' if satellite_attention_after is not None else 'Failed'}")
    except Exception as e:
        print(f"    After CIB: Error - {e}")
        satellite_attention_after = None
    
    extractor_before.remove_hooks()
    extractor_after.remove_hooks()
    
    # 检查是否成功提取
    if drone_attention_before is None:
        raise RuntimeError("Failed to extract attention maps for drone image (before CIB). Hook may not be working correctly.")
    if drone_attention_after is None:
        raise RuntimeError("Failed to extract attention maps for drone image (after CIB). Hook may not be working correctly.")
    if satellite_attention_before is None:
        raise RuntimeError("Failed to extract attention maps for satellite image (before CIB). Hook may not be working correctly.")
    if satellite_attention_after is None:
        raise RuntimeError("Failed to extract attention maps for satellite image (after CIB). Hook may not be working correctly.")
    
    # 调整图像大小
    drone_img_resized = cv2.resize(drone_img, (opt.w, opt.h))
    satellite_img_resized = cv2.resize(satellite_img, (opt.w, opt.h))
    
    # 生成热力图叠加
    print("Generating heatmap overlays...")
    drone_overlay_before = generate_heatmap_overlay(drone_img_resized, drone_attention_before)
    drone_overlay_after = generate_heatmap_overlay(drone_img_resized, drone_attention_after)
    satellite_overlay_before = generate_heatmap_overlay(satellite_img_resized, satellite_attention_before)
    satellite_overlay_after = generate_heatmap_overlay(satellite_img_resized, satellite_attention_after)
    
    # 创建对比图
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：Drone
    axes[0, 0].imshow(drone_img_resized)
    axes[0, 0].set_title('Input (Drone)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(drone_overlay_before)
    axes[0, 1].set_title('Before FSM (Drone)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(drone_overlay_after)
    axes[0, 2].set_title('After FSM (Drone)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行：Satellite
    axes[1, 0].imshow(satellite_img_resized)
    axes[1, 0].set_title('Input (Satellite)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(satellite_overlay_before)
    axes[1, 1].set_title('Before FSM (Satellite)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(satellite_overlay_after)
    axes[1, 2].set_title('After FSM (Satellite)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if output_filename is None:
        output_filename = 'ablation_heatmap_comparison.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if not output_filename.startswith('ablation_heatmap_comparison'):  # 批量处理时不打印
        print(f"\nHeatmap saved to: {output_path}")
    
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate ablation heatmap visualization')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0')
    parser.add_argument('--name', default='FCFE_Model_University', type=str, help='model name')
    parser.add_argument('--which_epoch', default='last', type=str, help='model epoch')
    parser.add_argument('--drone_img', type=str, required=True, help='path to drone image')
    parser.add_argument('--satellite_img', type=str, required=True, help='path to satellite image')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--output_dir', default='./heatmap_results', type=str, help='output directory')
    parser.add_argument('--model', default='convnext_small', type=str, help='model type')
    
    opt = parser.parse_args()
    
    # 加载配置
    yaml.warnings({'YAMLLoadWarning': False})
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        opt.fp16 = config.get('fp16', False)
        opt.views = config.get('views', 2)
        opt.block = config.get('block', 4)
        opt.M = config.get('M', 4)
        opt.share = config.get('share', False)
        if 'h' in config:
            opt.h = config['h']
            opt.w = config['w']
        if 'nclasses' in config:
            opt.nclasses = config['nclasses']
        else:
            opt.nclasses = 729
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using default values...")
        opt.fp16 = False
        opt.views = 2
        opt.block = 4
        opt.M = 4
        opt.share = False
        opt.nclasses = 729
    
    # 设置GPU
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    # 加载模型（模型一定包含CIB）
    print(f"\nLoading model: {opt.name}")
    model, _, _ = load_network(opt.name, opt)
    model.eval()
    
    # 生成可视化
    # 使用同一个模型，通过hook不同位置来展示CIB前后的差异
    output_path = create_ablation_visualization(
        opt.drone_img,
        opt.satellite_img,
        model,
        opt,
        opt.output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Visualization completed!")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
