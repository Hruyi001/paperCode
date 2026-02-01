# Safe-Net 热力图生成位置分析

## 模型结构流程图

```
输入图像 [B, 3, 256, 256]
    ↓
Transformer Backbone (ViT-S)
    ↓
features [B, 257, 768]
    ├─ global_feature [B, 768] (CLS token)
    └─ patch_features [B, 256, 768] (patch tokens)
         ↓
    【位置1: patch_features】← 原始patch特征热力图
         ↓
    FAM (Feature Alignment Module)
    - Loc_Net生成affine变换参数
    - 对patch_features进行空间对齐
         ↓
    aligned_feature_map [B, 768, 16, 16]
    【位置2: aligned_feature_map】← 当前热力图生成位置 ✓
         ↓
    FPM (Feature Partition Module)
    - 计算saliency values
    - 递归分区
         ↓
    part_features [B, 768, block]
    【位置3: part_features】← 分区特征热力图
         ↓
    分类器
    - Global分类器
    - Part分类器
```

## 各位置热力图的意义

### 位置1: patch_features (FAM之前)
**位置**: `models/model.py` line 126
```python
patch_features = features[:,1:]  # shape:[B, 256, 768]
```

**优点**:
- 显示Transformer提取的原始特征响应
- 可以看到模型在未对齐前关注哪些区域
- 与对齐后的热力图对比，可以验证FAM的效果

**缺点**:
- 特征图是1D序列，需要reshape到2D空间
- 可能包含一些噪声

**实现方式**:
```python
# 在feat_alignment之前保存
B, N, C = patch_features.size()
H = W = int(N**0.5)  # 16
patch_feature_map = patch_features.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, 768, 16, 16]
```

---

### 位置2: aligned_feature_map (FAM之后) ⭐ **推荐**
**位置**: `models/model.py` line 134, 186
```python
aligned_feature_map = self.feat_alignment(global_feature, patch_features)
self.aligned_feature_map = aligned_feature_map  # line 186
```

**优点**:
- ✅ **当前使用的位置，效果已经验证**
- 显示对齐后的特征响应，更符合模型的最终决策
- 特征图已经是2D空间结构 [B, 768, 16, 16]，便于可视化
- 与FPM的分区边界可以很好对应

**缺点**:
- 无法看到对齐前后的对比

**当前实现**: `utils/heatmap_utils.py` line 54-60

---

### 位置3: part_features (FPM之后)
**位置**: `models/model.py` line 136
```python
part_features = self.feat_partition(aligned_feature_map, pooling='avg')
# shape: [B, 768, block] 例如 [B, 768, 4]
```

**优点**:
- 显示不同分区的重要性
- 可以对比不同block的响应强度
- 与分类器的决策直接相关

**缺点**:
- 特征已经被pooling成1D向量，空间信息丢失
- 需要映射回原始空间位置才能生成热力图

**实现方式**:
```python
# 需要结合boundaries信息，将part_features映射回空间位置
# 在partition函数中，每个block对应一个空间区域
```

---

### 位置4: Transformer中间层特征
**位置**: Transformer的各个transformer block

**优点**:
- 可以看到不同深度的特征响应
- 理解特征提取的层次化过程

**缺点**:
- 需要修改Transformer代码添加hook
- 可能信息量过大，难以解释

---

## 推荐方案

### 方案1: 保持当前位置（推荐）⭐
**位置**: `aligned_feature_map` (FAM之后)

**理由**:
1. 这是模型的关键中间表示，直接影响最终分类
2. 特征已经对齐，空间结构清晰
3. 与FPM的分区边界可以很好对应
4. 当前实现已经验证有效

**改进建议**:
- 可以添加Grad-CAM方法，使用梯度信息增强热力图
- 可以对比不同epoch的aligned_feature_map，观察训练过程

---

### 方案2: 多位置对比（进阶）
**同时生成3个位置的热力图**:
1. **patch_features** (FAM之前) - 原始特征响应
2. **aligned_feature_map** (FAM之后) - 对齐后特征响应 ⭐
3. **part_features** (FPM之后) - 分区特征响应（映射回空间）

**优点**:
- 全面理解模型的工作流程
- 可以验证FAM和FPM的效果
- 适合论文可视化

**实现复杂度**: 中等

---

### 方案3: 基于Grad-CAM的增强（最佳）
**位置**: `aligned_feature_map` + 梯度信息

**方法**:
1. 在前向传播时保存 `aligned_feature_map`
2. 在反向传播时计算梯度
3. 使用Grad-CAM公式: `CAM = ReLU(Σ(α_k * A^k))`
   - `α_k = GAP(∂y^c/∂A^k)` (梯度全局平均池化)
   - `A^k` 是第k个通道的特征图

**优点**:
- 更准确地反映模型决策依据
- 结合了前向特征和反向梯度信息
- 是热力图生成的标准方法

**当前问题**: `utils/heatmap_utils.py` 中注释掉了梯度计算部分

---

## 具体实现建议

### 1. 在模型forward中保存多个中间特征
```python
# models/model.py
def forward(self, x):
    features, all_features = self.transformer(x)
    global_feature = features[:,0]
    patch_features = features[:,1:]
    
    # 保存FAM之前的特征
    if self.save_intermediate:
        B, N, C = patch_features.size()
        H = W = int(N**0.5)
        patch_feature_map = patch_features.view(B, H, W, C).permute(0, 3, 1, 2)
        self.intermediate_outputs['patch_feature_map'] = patch_feature_map.detach().cpu()
    
    # FAM
    aligned_feature_map = self.feat_alignment(global_feature, patch_features)
    
    # 保存FAM之后的特征（当前已有）
    if self.save_intermediate:
        self.intermediate_outputs['f_p_aligned'] = aligned_feature_map.detach().cpu()
    
    # FPM
    part_features = self.feat_partition(aligned_feature_map, pooling='avg')
    
    # 保存FPM之后的特征（需要映射回空间）
    if self.save_intermediate:
        # ... 映射part_features到空间位置
        pass
```

### 2. 增强热力图生成函数
```python
# utils/heatmap_utils.py
def generate_heatmap_from_feature_map(feature_map, method='l2_norm'):
    """
    从特征图生成热力图
    
    Args:
        feature_map: [B, C, H, W] 或 [C, H, W]
        method: 'l2_norm', 'max', 'mean', 'grad_cam'
    """
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]  # [C, H, W]
    
    if method == 'l2_norm':
        cam = torch.norm(feature_map, p=2, dim=0)  # [H, W]
    elif method == 'max':
        cam = torch.max(feature_map, dim=0)[0]  # [H, W]
    elif method == 'mean':
        cam = torch.mean(feature_map, dim=0)  # [H, W]
    elif method == 'grad_cam':
        # 需要梯度信息
        # cam = compute_grad_cam(feature_map, gradients)
        pass
    
    return cam
```

---

## 总结

**最佳热力图生成位置**: `aligned_feature_map` (FAM之后) ⭐

**原因**:
1. ✅ 特征已经对齐，空间结构清晰
2. ✅ 直接影响最终分类决策
3. ✅ 与FPM分区边界对应良好
4. ✅ 当前实现已验证有效

**改进方向**:
1. 添加Grad-CAM方法增强热力图质量
2. 可选：添加FAM前后的对比热力图
3. 可选：添加不同epoch的对比热力图（当前已有）

**当前实现状态**: ✅ 已经很好，位置选择合理
