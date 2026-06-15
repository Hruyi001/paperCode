# 消融实验热力图生成代码逻辑详解

## 整体流程概览

```
输入图像 → 模型前向传播 → Hook捕获特征 → 生成注意力图 → 叠加热力图 → 可视化对比
```

## 核心组件

### 1. AttentionExtractor 类（第34-188行）

这是核心类，负责提取模型的注意力图。

#### 1.1 Hook机制原理

**什么是Hook？**
- PyTorch的Hook机制允许在模型前向/反向传播过程中"拦截"中间层的输入/输出
- 类似于在函数执行过程中插入"监听器"

**为什么需要Hook？**
- 模型的最终输出是分类结果，无法直接看到中间层的注意力分布
- 需要通过Hook捕获中间层的特征图来可视化注意力

#### 1.2 Hook注册（register_hooks方法，第41-129行）

```python
def register_hooks(self, hook_before_cib=False):
```

**关键逻辑：**

1. **hook_before_cib=True（无CIB情况）**：
   ```python
   # Hook到 ADIB_layer 的输出
   # 位置：CIB_layer 之前
   # 含义：展示未经过CIB处理的基础特征注意力
   ```

2. **hook_before_cib=False（有CIB情况）**：
   ```python
   # Hook到 CIB_layer 的输入
   # 位置：CIB_layer 的输入（ADIB的输出，准备进入CIB）
   # 含义：展示准备进入CIB的特征注意力
   ```

**Hook函数详解：**

**CIB输入Hook（第47-65行）：**
```python
def cib_input_hook(module, input, output):
    # input[0] 是 ADIB_attention_features，形状为 [B, C, H, W, block]
    # 取第一个block: [B, C, H, W]
    # 计算通道平均: [B, 1, H, W] → 这就是注意力图
    feat = adib_features[:, :, :, :, 0]  # 取第一个block
    feat = torch.mean(feat, dim=1, keepdim=True)  # 通道平均
    self.activations.append(feat)  # 保存
```

**ADIB输出Hook（第67-93行）：**
```python
def before_cib_hook(module, input, output):
    # output 是 ADIB_layer 的输出，是一个特征列表
    # 取第一个特征: [B, C, H, W]
    # 计算通道平均: [B, 1, H, W] → 注意力图
    feat = output[0]
    feat = torch.mean(feat, dim=1, keepdim=True)
    self.activations.append(feat)
```

#### 1.3 注意力提取（extract_attention方法，第137-188行）

```python
def extract_attention(self, input_img, view_index=1, opt=None):
```

**执行流程：**

1. **清空之前的激活值**：
   ```python
   self.activations = []
   ```

2. **模型前向传播**（第143-165行）：
   ```python
   # 根据view_index调用模型
   if view_index == 1:  # Satellite
       outputs, _ = self.model(input_img, None)
   elif view_index == 3:  # Drone
       _, outputs = self.model(None, input_img)
   ```
   
   **关键点**：
   - 在前向传播过程中，Hook函数会被自动调用
   - Hook捕获的特征会保存到 `self.activations` 中
   - 即使前向传播在后续层失败（如分类器），Hook已经捕获了特征

3. **提取注意力图**（第168-175行）：
   ```python
   if self.activations:
       features = self.activations[-1]  # 取最后一个捕获的特征
       # features形状: [B, 1, H, W]
       attention = torch.mean(features, dim=1)[0].cpu().numpy()  # [H, W]
       # 归一化到 [0, 1]
       attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
       return attention
   ```

## 消融实验设计（第259-313行）

### 2.1 双模型对比策略

```python
# 创建两个提取器
extractor_without = AttentionExtractor(model_without_cib)
extractor_without.register_hooks(hook_before_cib=True)  # Hook到CIB之前

extractor_with = AttentionExtractor(model_with_cib)
extractor_with.register_hooks(hook_before_cib=False)  # Hook到CIB输入
```

**设计思路：**

1. **Without CIB**：
   - Hook位置：`ADIB_layer` 输出
   - 展示：未经过CIB处理的基础特征注意力
   - 代表：没有CIB组件时的模型行为

2. **With CIB**：
   - Hook位置：`CIB_layer` 输入（ADIB输出）
   - 展示：准备进入CIB的特征注意力
   - 代表：有CIB组件时的模型行为

**为什么这样设计？**

- CIB的输出是字典格式的1D特征，不适合直接可视化
- 通过对比CIB前后的特征，可以看到CIB对注意力的影响
- 即使使用同一个模型，通过hook不同位置也能展示差异

### 2.2 处理流程

```python
# 1. 处理Drone图像
drone_attention_without = extractor_without.extract_attention(drone_tensor, view_index=3)
drone_attention_with = extractor_with.extract_attention(drone_tensor, view_index=3)

# 2. 处理Satellite图像
satellite_attention_without = extractor_without.extract_attention(satellite_tensor, view_index=1)
satellite_attention_with = extractor_with.extract_attention(satellite_tensor, view_index=1)
```

## 热力图生成（第191-203行）

### 3.1 热力图叠加

```python
def generate_heatmap_overlay(image, attention_map, alpha=0.5):
```

**步骤：**

1. **调整尺寸**：
   ```python
   attention_resized = cv2.resize(attention_map, (w, h))
   # attention_map: [H, W] → 调整到图像尺寸
   ```

2. **颜色映射**：
   ```python
   heatmap = cm.jet(attention_resized)[:, :, :3]
   # jet colormap: 蓝色(低) → 绿色 → 黄色 → 红色(高)
   # 输出: [H, W, 3] RGB图像
   ```

3. **叠加**：
   ```python
   overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
   # 原图和热力图按比例混合
   # alpha=0.5 表示各占50%
   ```

## 可视化对比（第326-368行）

### 4.1 2×3网格布局

```
┌─────────────────┬──────────────────┬─────────────────┐
│  Input (Drone)  │ Without CIB      │  With CIB       │
│                 │ (Drone)          │ (Drone)         │
├─────────────────┼──────────────────┼─────────────────┤
│ Input (Satellite│ Without CIB      │  With CIB       │
│                 │ (Satellite)      │ (Satellite)     │
└─────────────────┴──────────────────┴─────────────────┘
```

### 4.2 生成过程

```python
# 1. 创建2×3子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 2. 第一行：Drone图像
axes[0, 0].imshow(drone_img_resized)  # 原始图像
axes[0, 1].imshow(drone_overlay_without)  # 无CIB热力图
axes[0, 2].imshow(drone_overlay_with)  # 有CIB热力图

# 3. 第二行：Satellite图像
axes[1, 0].imshow(satellite_img_resized)  # 原始图像
axes[1, 1].imshow(satellite_overlay_without)  # 无CIB热力图
axes[1, 2].imshow(satellite_overlay_with)  # 有CIB热力图

# 4. 保存
plt.savefig(output_path, dpi=300)
```

## 完整数据流

```
输入图像
    ↓
预处理（resize, normalize）
    ↓
模型前向传播
    ↓
    ├─→ ADIB_layer → [特征列表]
    │       ↓
    │   Hook捕获 (Without CIB)
    │       ↓
    │   注意力图1
    │
    └─→ CIB_layer输入 → [B,C,H,W,block]
            ↓
        Hook捕获 (With CIB)
            ↓
        注意力图2
            ↓
    归一化 [0,1]
            ↓
    调整尺寸到图像大小
            ↓
    Jet colormap (蓝→红)
            ↓
    叠加到原图 (alpha混合)
            ↓
    保存为PNG
```

## 关键技术点

### 1. Hook机制的优势

- **非侵入性**：不需要修改模型代码
- **灵活性**：可以hook到任意层
- **实时捕获**：在前向传播过程中自动捕获

### 2. 注意力图计算方法

- **通道平均**：`torch.mean(features, dim=1)`
  - 将多通道特征图压缩为单通道
  - 每个空间位置的值代表该位置的"重要性"

- **归一化**：`(x - min) / (max - min)`
  - 将值映射到[0, 1]区间
  - 便于可视化

### 3. 消融实验的巧妙设计

- **同一模型，不同Hook位置**：
  - 即使没有单独的无CIB模型，也能通过hook不同位置展示差异
  - Hook到CIB之前 = 无CIB效果
  - Hook到CIB输入 = 有CIB效果

## 代码执行顺序

1. **初始化**（main函数）：
   - 加载模型
   - 创建AttentionExtractor

2. **注册Hook**：
   - `register_hooks()` 注册hook函数

3. **提取注意力**：
   - `extract_attention()` 执行前向传播
   - Hook自动捕获特征
   - 返回注意力图

4. **生成热力图**：
   - `generate_heatmap_overlay()` 叠加热力图

5. **可视化**：
   - `create_ablation_visualization()` 创建对比图

6. **清理**：
   - `remove_hooks()` 移除hook

## 总结

这个代码的核心思想是：
1. **使用Hook机制**捕获模型中间层的特征
2. **通过对比不同位置的特征**来展示CIB组件的影响
3. **将特征图转换为热力图**并叠加到原图
4. **生成对比可视化**展示消融实验结果

整个过程是**非侵入性的**，不需要修改模型代码，只需要在运行时注册hook即可。
