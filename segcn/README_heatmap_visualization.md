# 对比热力图可视化使用说明

本目录提供了生成特征可视化对比热力图的完整工具，用于展示模型在不同特征提取阶段的关注区域。

## 📋 功能说明

### 1. 对比热力图内容

生成的对比热力图包含4个子图：
- **原始图像**: 输入图像
- **Backbone特征 (part_features)**: ConvNeXt backbone直接提取的原始空间特征
- **DSA热力图 (pfeat_align)**: DSA模块处理后的注意力热力图
- **DSA特征叠加 (pfeat_align)**: DSA对齐后特征与原图的叠加可视化

### 2. 为什么选择这些位置？

- **part_features (output[-1])**: 展示backbone的原始特征提取能力，保留完整的空间信息
- **pfeat_align (output[0])**: **最能代表DSA方法的核心效果**，融合了注意力机制和多尺度特征，直接关联到DSA_loss

## 🚀 快速开始

### 方法1: 使用Shell脚本（推荐）

1. **编辑脚本配置**
   
   打开 `run_heatmap_visualization.sh`，修改以下参数：
   
   ```bash
   # 数据集路径
   DATASET_PATH="sample4geo/dataset/U1652"
   
   # 模型权重路径（可选）
   CHECKPOINT_PATH="checkpoints/university/convnext_tiny.fb_in22k_ft_in1k_384/1002094438/weights_e1_0.9341.pth"
   
   # 输出目录
   OUTPUT_DIR="./heatmap_visualization_results"
   
   # 数据集类型
   DATASET_TYPE="U1652"  # 可选: U1652, SUES-200, custom
   
   # 每个类别采样数量
   NUM_SAMPLES=5
   ```

2. **运行脚本**
   
   ```bash
   bash run_heatmap_visualization.sh
   ```

### 方法2: 直接使用Python脚本

#### 单图像可视化

```bash
python visualize_single_image.py \
    --image_path path/to/your/image.jpg \
    --checkpoint path/to/checkpoint.pth \
    --output_path output_heatmap.png
```

#### 批量处理数据集

```bash
python visualize_comparison_heatmap.py \
    --dataset_path sample4geo/dataset/U1652 \
    --checkpoint checkpoints/your_model.pth \
    --output_dir ./heatmap_results \
    --num_samples 5 \
    --dataset_type U1652 \
    --img_size 384 \
    --num_classes 701 \
    --block 2 \
    --gpu_id 0
```

## 📁 数据集路径配置

### University-1652 数据集

```
sample4geo/dataset/U1652/
├── train/
│   ├── drone/
│   └── satellite/
└── test/
    ├── query_drone/
    ├── gallery_satellite/
    ├── query_satellite/
    └── gallery_drone/
```

在脚本中设置：
```bash
DATASET_PATH="sample4geo/dataset/U1652"
DATASET_TYPE="U1652"
```

### SUES-200 数据集

```
sample4geo/dataset/SUES-200/
├── Training/
│   ├── 150/
│   ├── 200/
│   ├── 250/
│   └── 300/
└── Testing/
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
```

在脚本中设置：
```bash
DATASET_PATH="sample4geo/dataset/SUES-200"
DATASET_TYPE="SUES-200"
```

## ⚙️ 参数说明

### 主要参数

- `--dataset_path`: 数据集根路径（必需）
- `--checkpoint`: 模型权重路径（可选，不提供则使用随机初始化）
- `--output_dir`: 输出目录（默认: `./heatmap_visualization_results`）
- `--num_samples`: 每个类别采样的图像数量（默认: 5）
- `--dataset_type`: 数据集类型，可选 `U1652`, `SUES-200`, `custom`（默认: `U1652`）
- `--img_size`: 图像尺寸（默认: 384）
- `--num_classes`: 类别数量（默认: 701）
- `--block`: block数量（默认: 2）
- `--gpu_id`: GPU ID（默认: 0）

## 📊 输出结果

生成的对比热力图保存在指定的输出目录中，文件名格式：
- `{class_id}_query_{idx}_comparison.png` - 查询图像的对比热力图
- `{class_id}_gallery_{idx}_comparison.png` - 图库图像的对比热力图

每个对比热力图包含4个子图，展示了从原始特征到DSA处理后的特征变化。

## 🔍 技术细节

### 模型输出结构

**训练模式** (`model.train()`):
```python
output = [pfeat_align, cls, features, gap_feature, part_features]
```

**评估模式** (`model.eval()`):
```python
output = [gap_feature, part_features]
```

**注意**: 为了获取 `pfeat_align`，脚本会临时设置 `model.train()`，但使用 `torch.no_grad()` 确保不更新梯度。

### 特征图尺寸

- 输入图像: 384×384
- ConvNeXt下采样: 32倍
- 特征图尺寸: 12×12
- `pfeat_align`: (B, 512, 144) → reshape为 (B, 512, 12, 12)
- `part_features`: (B, 768, 12, 12)

### 热力图生成流程

1. 提取特征图（`part_features` 或 `pfeat_align`）
2. 对通道维度求平均: `features.mean(dim=0)` → (H, W)
3. 归一化到 [0, 1]
4. 上采样到原图大小: `cv2.resize(heatmap, (384, 384))`
5. 应用colormap: `cv2.COLORMAP_JET`
6. 叠加到原图: `cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)`

## 🐛 常见问题

### 1. 模型权重加载失败

**问题**: 提示某些键不匹配

**解决**: 脚本会自动处理DataParallel保存的权重，并移除分类器权重。如果仍有问题，检查模型结构是否匹配。

### 2. 数据集路径不存在

**问题**: 提示数据集路径不存在

**解决**: 
- 检查 `DATASET_PATH` 是否正确
- 如果使用自定义数据集，设置 `DATASET_TYPE="custom"`

### 3. 内存不足

**问题**: GPU内存不足

**解决**: 
- 减少 `NUM_SAMPLES` 参数
- 使用更小的 `IMG_SIZE`
- 在CPU上运行（会自动检测）

### 4. 无法获取 pfeat_align

**问题**: 生成的对比图中没有DSA特征

**解决**: 
- 确保模型处于训练模式（脚本会自动设置）
- 检查模型输出结构是否正确

## 📝 示例

### 示例1: 可视化单张图像

```bash
python visualize_single_image.py \
    --image_path sample4geo/dataset/U1652/test/query_drone/0001/image1.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output_path my_heatmap.png
```

### 示例2: 批量处理数据集

```bash
# 修改 run_heatmap_visualization.sh 中的配置
# 然后运行
bash run_heatmap_visualization.sh
```

## 📚 相关文件

- `visualize_comparison_heatmap.py`: 批量处理脚本
- `visualize_single_image.py`: 单图像可视化脚本
- `run_heatmap_visualization.sh`: Shell运行脚本
- `模型结构分析与热力图生成位置建议.md`: 详细的技术分析文档
- `generate_heatmap.py`: 基础热力图生成工具

## 🎯 论文展示建议

1. **主图**: 使用 `pfeat_align` 的热力图，突出DSA方法的核心效果
2. **对比图**: 同时展示 `part_features` 和 `pfeat_align` 的对比，证明DSA模块的有效性
3. **消融实验**: 可以分别可视化有无DSA模块时的特征热力图

## 📧 联系

如有问题或建议，请提交Issue或联系项目维护者。
