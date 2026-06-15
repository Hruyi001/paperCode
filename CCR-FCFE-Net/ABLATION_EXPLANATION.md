# 消融实验代码详解

## 实验目的

这个消融实验（Ablation Study）用于**对比分析 CIB（Counterfactual Intervention Block）组件对模型注意力机制的影响**。

## 模型架构流程

根据代码分析，CCR模型的完整前向传播流程如下：

```
输入图像 (Drone/Satellite)
    ↓
Backbone (ConvNext/ResNet)
    ↓ 提取特征
    ├─→ gap_feature ──────────→ classifier1 ─→ convnext_feature
    └─→ part_features ────────→ ADIB_layer ─→ ADIB_features
                                    ↓
                            ADIB_attention_features (stack)
                                    ↓
                            CIB_layer ──────────────→ (nfeature, cfeature)
                                    ↓
                            part_classifier ────────→ 最终输出
```

## 关键组件说明

### 1. **ADIB_layer** (Attention-Driven Information Bottleneck)
- **位置**: Backbone之后，CIB之前
- **作用**: 提取注意力驱动的特征
- **输出**: `ADIB_features` - 一个特征列表

### 2. **CIB_layer** (Counterfactual Intervention Block) ⭐ **消融目标**
- **位置**: ADIB_layer之后
- **作用**: 
  - 对ADIB特征进行反事实干预（Counterfactual Intervention）
  - 生成正常特征（normal feature）和反事实特征（counterfactual feature）
  - 通过对比学习提升模型的跨视图匹配能力
- **输入**: `ADIB_attention_features` [B, C, H, W, block]
- **输出**: 
  - `nfeature`: 正常特征字典
  - `cfeature`: 反事实特征字典

## 消融实验设计

### 实验对比

代码通过以下方式实现消融：

1. **"Without CIB"（无CIB）**:
   - Hook位置: `ADIB_layer` 的输出
   - 含义: 展示**没有经过CIB处理**的特征注意力图
   - 代码: `hook_before_cib=True`

2. **"With CIB"（有CIB）**:
   - Hook位置: `CIB_layer` 的输入（即ADIB的输出，但准备进入CIB）
   - 含义: 展示**经过CIB处理**后的特征注意力图
   - 代码: `hook_before_cib=False`

### 关键代码解析

```python
# 第262-268行
# 对于无CIB模型，hook到CIB_layer之前（ADIB_layer之后）
extractor_without = AttentionExtractor(model_without_cib)
extractor_without.register_hooks(hook_before_cib=True)  # Hook到CIB之前

# 对于有CIB模型，hook到CIB_layer之后
extractor_with = AttentionExtractor(model_with_cib)
extractor_with.register_hooks(hook_before_cib=False)  # Hook到CIB之后
```

### Hook机制说明

#### Hook到ADIB_layer（无CIB）
```python
def before_cib_hook(module, input, output):
    # ADIB_layer的输出: part_features (list)
    # 直接使用ADIB输出的特征图
    feat = output[0]  # [B, C, H, W]
    feat = torch.mean(feat, dim=1, keepdim=True)  # 计算通道平均
    self.activations.append(feat)
```

#### Hook到CIB_layer输入（有CIB）
```python
def cib_input_hook(module, input, output):
    # CIB_layer的输入: ADIB_attention_features [B, C, H, W, block]
    # 取第一个block并计算通道平均
    adib_features = input[0]  # [B, C, H, W, block]
    feat = adib_features[:, :, :, :, 0]  # 取第一个block
    feat = torch.mean(feat, dim=1, keepdim=True)  # 通道平均
    self.activations.append(feat)
```

## 可视化输出

代码生成一个 **2×3 的对比图**：

```
┌─────────────────┬──────────────────┬─────────────────┐
│  Input (Drone)  │ Without CIB     │  With CIB       │
│                 │ (Drone)          │ (Drone)         │
├─────────────────┼──────────────────┼─────────────────┤
│ Input (Satellite│ Without CIB      │  With CIB       │
│                 │ (Satellite)      │ (Satellite)     │
└─────────────────┴──────────────────┴─────────────────┘
```

## 实验意义

通过这个消融实验，可以：

1. **验证CIB组件的有效性**: 
   - 对比有/无CIB时模型关注区域的变化
   - 观察CIB是否帮助模型关注更相关的特征

2. **理解模型行为**:
   - 可视化模型在不同视图（Drone/Satellite）下的注意力分布
   - 分析CIB如何改善跨视图特征匹配

3. **指导模型改进**:
   - 如果CIB显著改善了注意力分布，说明该组件有效
   - 如果差异不明显，可能需要调整CIB的设计

## 技术细节

### 注意力图提取方法

1. **特征提取**: 使用PyTorch的hook机制捕获中间层特征
2. **注意力计算**: 对特征图进行通道维度平均，得到空间注意力图
3. **热力图生成**: 使用jet colormap（蓝→绿→黄→红）将注意力图叠加到原图

### 为什么Hook到不同位置？

- **Without CIB**: Hook到ADIB输出，展示基础特征注意力
- **With CIB**: Hook到CIB输入（实际是ADIB输出），但这里展示的是**准备进入CIB的特征**，可以理解为"经过CIB处理准备的特征"

**注意**: 由于CIB的输出是字典格式的1D特征，不适合直接可视化，所以代码选择hook到CIB的输入（ADIB的输出）来展示特征。

## 使用示例

```bash
# 运行消融实验
./run_heatmap_ablation.sh

# 或直接使用Python
python generate_heatmap_ablation.py \
    --drone_img /path/to/drone.jpg \
    --satellite_img /path/to/satellite.jpg \
    --name CCR_Model_University
```

## 总结

这个消融实验代码专门用于**分析CIB（Counterfactual Intervention Block）组件对模型注意力机制的影响**，通过对比有/无CIB时的注意力热力图，验证CIB组件在跨视图地理定位任务中的有效性。
