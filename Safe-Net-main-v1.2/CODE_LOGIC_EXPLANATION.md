# 实验图生成代码逻辑详解

本文档详细解释生成4x5实验图的完整代码逻辑流程。

## 整体流程图

```
输入图像 → 模型前向传播 → 提取中间参数 → 生成可视化 → 组合成网格图
   ↓            ↓              ↓              ↓            ↓
卫星图+4张    FAM/FPM模块    theta/boundaries   Aligned/Partition/  4x5网格
无人机图      输出中间结果    f_p_aligned        Heatmap图
```

## 一、模型中间参数保存机制

### 1.1 模型初始化 (`models/model.py`)

```python
class Safe_Net_model(nn.Module):
    def __init__(self, ..., save_intermediate=False):
        self.save_intermediate = save_intermediate  # 控制是否保存中间输出
        self.intermediate_outputs = {}  # 存储中间输出的字典
```

**关键点**：
- `save_intermediate` 标志控制是否保存中间参数
- `intermediate_outputs` 字典存储所有中间结果

### 1.2 前向传播中的参数保存 (`models/model.py:forward`)

```python
def forward(self, x):
    # 1. ViT骨干提取特征
    features, all_features = self.transformer(x)
    global_feature = features[:,0]      # [B, 768] 全局特征
    patch_features = features[:,1:]     # [B, 256, 768] patch特征
    
    # 2. FAM模块：特征对齐
    aligned_feature_map = self.feat_alignment(global_feature, patch_features)
    
    # 3. FPM模块：特征分区
    part_features = self.feat_partition(aligned_feature_map, pooling='avg')
    
    # 4. 【关键】保存中间输出供可视化使用
    if self.save_intermediate:
        A_theta = self.loc_net(global_feature)  # 获取affine变换矩阵
        boundaries = self.get_boundary(aligned_feature_map)  # 获取分区边界
        
        self.intermediate_outputs = {
            "theta": A_theta.detach().cpu(),        # [B, 2, 3] affine变换矩阵
            "boundaries": boundaries,                # list of lists，如[[b1,b2,b3], ...]
            "f_p_aligned": aligned_feature_map.detach().cpu()  # [B, C, H, W] 对齐后特征图
        }
```

**数据流**：
- `theta`: FAM模块的定位子网（Loc_Net）输出，6个参数组成2×3矩阵
  - 格式：`[[a, b, tx], [c, d, ty]]`，包含旋转、缩放、平移信息
- `boundaries`: FPM模块计算的分区边界
  - 格式：`[b1, b2, b3]`，表示3条边界，将特征图分为4个区域
- `f_p_aligned`: FAM模块输出的对齐后特征图
  - 尺寸：`[B, 768, 16, 16]`（16×16对应256×256图像的patch网格）

## 二、可视化工具函数逻辑

### 2.1 Aligned图生成 (`utils/visualize.py:align_image`)

**目标**：根据FAM输出的theta参数，对原始图像应用affine变换

```python
def align_image(original_img, theta, img_size=256):
    # 步骤1: 转换theta格式
    # theta形状: [2, 3]，表示affine变换矩阵
    # [[a, b, tx], [c, d, ty]]
    
    # 步骤2: 将theta从特征图空间转换到图像空间
    # 特征图是16×16，图像是256×256，缩放因子=16
    scale_factor = img_size / 16
    theta_img[0, 2] = theta[0, 2] * scale_factor  # 平移tx缩放
    theta_img[1, 2] = theta[1, 2] * scale_factor  # 平移ty缩放
    
    # 步骤3: 创建affine grid
    grid = F.affine_grid(theta_img.unsqueeze(0), img_tensor.size(), align_corners=True)
    
    # 步骤4: 应用grid_sample进行双线性采样
    aligned_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', ...)
    
    # 步骤5: 转换回numpy数组
    return aligned_img
```

**核心原理**：
- **Affine变换**：包含旋转、缩放、平移、剪切等线性变换
- **Grid Sample**：根据变换后的采样网格，从原图中采样像素值
- **空间映射**：theta基于16×16特征图，需要映射到256×256图像空间

**效果**：消除不同视角/距离造成的尺度偏差，统一图像方向

### 2.2 Partition图生成 (`utils/visualize.py:draw_partition`)

**目标**：在Aligned图上绘制FPM输出的分区边界

```python
def draw_partition(aligned_img, boundaries, img_size=256):
    # 步骤1: 解析boundaries
    # boundaries格式: [b1, b2, b3]（基于16×16特征图）
    # 例如: [4, 8, 12] 表示3条边界，将16×16特征图分为4个同心正方形环
    
    # 步骤2: 将特征图空间的边界映射到图像像素空间
    feat_size = 16  # 特征图尺寸
    scale_factor = min(h_orig, w_orig) / feat_size  # 缩放因子
    
    # 步骤3: 计算每个边界框的像素坐标
    for boundary in boundaries:
        boundary_pixels = int(boundary * scale_factor)  # 边界像素值
        
        # 计算矩形框（以图像中心为原点）
        center_h, center_w = h_orig // 2, w_orig // 2
        x1 = center_w - boundary_pixels
        y1 = center_h - boundary_pixels
        x2 = center_w + boundary_pixels
        y2 = center_h + boundary_pixels
        
        # 步骤4: 绘制矩形框（使用matplotlib绘制虚线）
        rect = patches.Rectangle(..., linestyle='--', edgecolor='cyan')
    
    return partition_img
```

**核心原理**：
- **同心正方形分区**：FPM将特征图分为从中心到边缘的同心正方形环
- **边界映射**：特征图边界（16×16空间）→ 图像像素边界（256×256空间）
- **可视化**：用蓝色虚线矩形框标记4个语义一致的区域

**效果**：可视化特征分区结果，展示模型如何将图像分为不同语义区域

### 2.3 Heatmap生成 (`utils/visualize.py:generate_heatmap`)

**目标**：根据FAM输出的对齐后特征图生成热力图

```python
def generate_heatmap(f_p_aligned, original_img_size=(256, 256)):
    # 步骤1: 特征图聚合
    # f_p_aligned形状: [C, H, W] = [768, 16, 16]
    # 沿通道维度求和，得到单通道特征图
    heatmap = f_p_aligned.sum(dim=0)  # [16, 16]
    
    # 步骤2: 归一化到0-255
    heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min) * 255
    
    # 步骤3: 上采样到图像尺寸
    heatmap_resized = cv2.resize(heatmap_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # 步骤4: 应用JET配色方案
    # JET: 蓝色(低值) → 青色 → 绿色 → 黄色 → 红色(高值)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    return heatmap_rgb
```

**核心原理**：
- **特征聚合**：768个通道的特征图 → 单通道激活图（求和操作）
- **空间映射**：16×16特征图 → 256×256图像（双线性插值上采样）
- **颜色编码**：JET配色方案，红色/橙色表示高激活区域（模型关注的重点）

**效果**：可视化模型关注的重点区域（如目标建筑），红色区域表示高响应

## 三、主脚本执行流程 (`visualize_demo.py`)

### 3.1 初始化阶段

```python
# 1. 加载模型（启用save_intermediate）
opt.save_intermediate = True
model = load_network(opt, gpu_ids)
model = model.eval()

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 3.2 单张图像处理流程 (`process_image`函数)

```python
def process_image(model, img_path, opt, is_satellite=False):
    # === 阶段1: 图像加载和预处理 ===
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).cuda()
    input_img = np.array(img)  # 保存原始图像用于显示
    
    # === 阶段2: 模型前向传播（关键步骤）===
    with torch.no_grad():
        # 根据模式选择输入（mode=1: drone->satellite）
        if opt.mode == 1:
            if is_satellite:
                outputs, _ = model(img_tensor, None)  # 卫星图作为query_satellite
            else:
                _, outputs = model(None, img_tensor)  # 无人机图作为query_drone
        
        # === 阶段3: 提取中间输出 ===
        if hasattr(model, 'module'):  # DataParallel包装
            intermediate_outputs = model.module.intermediate_outputs
        else:
            intermediate_outputs = model.intermediate_outputs
        
        theta = intermediate_outputs.get('theta', None)        # [B, 2, 3]
        boundaries = intermediate_outputs.get('boundaries', None)  # list
        f_p_aligned = intermediate_outputs.get('f_p_aligned', None)  # [B, C, H, W]
        
        # === 阶段4: 生成三种可视化图 ===
        
        # 4.1 Aligned图：使用theta对原图进行affine变换
        if theta is not None:
            aligned_img = align_image(img, theta[0], img_size=opt.h)
            aligned_img = Image.fromarray(aligned_img).resize((w_orig, h_orig))
        
        # 4.2 Partition图：在Aligned图上绘制边界
        if boundaries is not None:
            partition_img = draw_partition(aligned_img, boundaries, img_size=opt.h)
        
        # 4.3 Heatmap：从对齐后特征图生成
        if f_p_aligned is not None:
            heatmap_img = generate_heatmap(f_p_aligned[0], (h_orig, w_orig))
    
    return {
        'input': input_img,
        'aligned': aligned_img,
        'partition': partition_img,
        'heatmap': heatmap_img
    }
```

**关键点**：
1. **模型输入**：根据`opt.mode`决定输入到`model(x1, x2)`的哪个位置
2. **中间输出提取**：从`intermediate_outputs`字典中提取三个关键参数
3. **顺序依赖**：Partition图依赖Aligned图，Heatmap独立生成

### 3.3 批量处理和网格组合

```python
# === 处理所有图像 ===
satellite_results = process_image(model, opt.satellite_img, opt, is_satellite=True)
drone_results = []
for i, drone_path in enumerate(drone_img_paths[:4]):
    drone_result = process_image(model, drone_path, opt, is_satellite=False)
    drone_results.append(drone_result)

# === 创建4x5网格图 ===
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# 填充网格
for row in range(4):  # Input, Aligned, Partition, Heatmap
    for col in range(5):  # Satellite + 4个Drone
        if col == 0:
            data = satellite_results[row_type]
        else:
            data = drone_results[col-1][row_type]
        axes[row, col].imshow(data)
        axes[row, col].axis('off')

# 保存图像
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

## 四、数据流总结

### 4.1 完整数据流

```
原始图像 (256×256×3)
    ↓
[ViT骨干] → 全局特征 [B,768] + Patch特征 [B,256,768]
    ↓
[FAM模块]
    ├─→ Loc_Net → theta [B,2,3] ──────────┐
    └─→ Grid Sample → f_p_aligned [B,768,16,16] ─┐
                                                  │
[FPM模块]                                         │
    └─→ get_boundary → boundaries [[b1,b2,b3]] ──┤
                                                  │
[中间输出保存] ←──────────────────────────────────┘
    ├─ theta
    ├─ boundaries
    └─ f_p_aligned
    ↓
[可视化生成]
    ├─ align_image(theta) → Aligned图
    ├─ draw_partition(boundaries) → Partition图
    └─ generate_heatmap(f_p_aligned) → Heatmap图
    ↓
[网格组合] → 4×5实验图
```

### 4.2 关键参数说明

| 参数 | 形状/格式 | 来源 | 用途 |
|------|----------|------|------|
| `theta` | `[B, 2, 3]` | FAM的Loc_Net | Aligned图生成 |
| `boundaries` | `[[b1,b2,b3], ...]` | FPM的get_boundary | Partition图绘制 |
| `f_p_aligned` | `[B, 768, 16, 16]` | FAM的grid_sample | Heatmap生成 |

### 4.3 空间映射关系

```
特征图空间 (16×16)         图像空间 (256×256)
    ↓                          ↑
boundaries: [4,8,12]  →  边界像素: [64,128,192]
theta: [2,3] (特征空间) → theta_img: [2,3] (图像空间)
f_p_aligned: [16,16]   →  heatmap: [256,256] (上采样)
```

## 五、代码执行顺序

1. **模型加载** → 启用`save_intermediate=True`
2. **图像预处理** → Resize + Normalize
3. **模型前向传播** → 自动保存中间输出到`intermediate_outputs`
4. **提取中间参数** → theta, boundaries, f_p_aligned
5. **生成Aligned图** → 使用theta对原图affine变换
6. **生成Partition图** → 在Aligned图上绘制边界框
7. **生成Heatmap** → 从f_p_aligned生成热力图
8. **组合网格** → 4行×5列布局
9. **保存图像** → 300 DPI PNG格式

## 六、关键设计决策

1. **中间参数保存时机**：在FAM和FPM之后、分类器之前保存，确保参数完整
2. **空间映射策略**：使用缩放因子将特征图空间映射到图像空间
3. **可视化顺序**：Aligned → Partition → Heatmap，体现处理流程
4. **网格布局**：4行（处理阶段）× 5列（不同视角），清晰展示对比效果

## 七、常见问题

### Q1: 为什么Aligned图看起来和原图一样？
**A**: 如果theta接近单位矩阵（无变换），说明图像已经对齐或变换很小，这是正常的。

### Q2: Partition图的边界框数量不对？
**A**: 检查`opt.block`参数，block=4时应该有3条边界（4个区域）。

### Q3: Heatmap全黑或全白？
**A**: 检查特征图归一化，可能需要调整归一化方法或使用不同的聚合方式（如L2范数）。

### Q4: 中间输出为空？
**A**: 确保`save_intermediate=True`，且模型前向传播正常执行。
