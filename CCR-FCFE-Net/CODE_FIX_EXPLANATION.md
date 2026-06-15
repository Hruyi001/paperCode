# 代码修复说明

## 修复的问题

**原问题**：代码中两个hook都捕获的是CIB之前的特征，无法真正展示CIB处理前后的差异。

## 修复方案

### 修复前
- **Without CIB**: Hook到ADIB输出（CIB之前）✓
- **With CIB**: Hook到CIB输入（也是CIB之前）✗

### 修复后
- **Without CIB**: Hook到ADIB输出（CIB之前）✓
- **With CIB**: Hook到CIB内部的`attentions`模块输出（CIB处理后的attention maps）✓

## 技术细节

### CIB_layer的结构
```python
class CIB_block(nn.Module):
    def __init__(self):
        self.attentions = Attentions(...)  # 注意力模块
    
    def forward(self, x):
        for i in range(self.block):
            part_attention_maps[i] = self.attentions(part[i])  # [B, M, H, W]
            # ... 后续处理
```

### Hook策略

1. **Before CIB** (hook_before_cib=True):
   - Hook位置: `ADIB_layer` 的输出
   - 捕获: ADIB处理后的特征图 `[B, C, H, W]`
   - 含义: CIB处理前的特征注意力

2. **After CIB** (hook_before_cib=False):
   - Hook位置: `CIB_layer.attentions` 的输出
   - 捕获: CIB生成的attention maps `[B, M, H, W]`
   - 含义: CIB处理后的注意力分布
   - 处理: 对M维度求平均得到 `[B, 1, H, W]`

### 关键代码修改

```python
# 查找CIB内部的attentions模块
if 'CIB_layer' in name or 'CIB' in name:
    cib_module = module
    for sub_name, sub_module in cib_module.named_modules():
        if 'attentions' in sub_name.lower():
            # Hook到attentions模块的输出
            target_layer = (f"{name}.{sub_name}", sub_module, cib_attentions_hook)
            break
```

### Attention Maps处理

```python
def cib_attentions_hook(module, input, output):
    # output: [B, M, H, W] - CIB生成的attention maps
    # 对M维度求平均，得到空间注意力图
    feat = torch.mean(output, dim=1, keepdim=True)  # [B, 1, H, W]
    # 只保存第一次（CIB的forward会循环调用attentions）
    if len(self.activations) == 0:
        self.activations.append(feat.clone().detach())
```

## 数据流对比

### 修复前（错误）
```
ADIB输出 → Hook1 (Without CIB) ✓
    ↓
CIB输入 → Hook2 (With CIB) ✗  # 实际上还是CIB之前
```

### 修复后（正确）
```
ADIB输出 → Hook1 (Without CIB) ✓
    ↓
CIB处理
    ↓
CIB.attentions输出 → Hook2 (With CIB) ✓  # CIB处理后的attention maps
```

## 验证方法

运行代码后，应该看到：
- **Without CIB**: 显示ADIB输出的特征注意力
- **With CIB**: 显示CIB处理后的attention maps

两者应该有明显的差异，展示CIB组件对注意力分布的影响。
