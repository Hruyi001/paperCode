# 热力图可视化使用说明

本工具用于生成QDFL模型在University-1652数据集上的热力图可视化，展示模型关注的重点区域。

## 功能说明

- 从指定目录读取图像（支持jpg、png等格式）
- 自动提取模型AQEU模块输出的`x_fine_0`特征图
- 生成热力图可视化对比图（原图、热力图、叠加图）
- 支持批量处理和对比图生成

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 基本用法
bash scripts/visualize_heatmap.sh <image_dir>

# 完整参数
bash scripts/visualize_heatmap.sh \
    <image_dir> \
    <checkpoint_path> \
    <config_file> \
    <output_dir> \
    <num_images> \
    <img_size>

# 示例
bash scripts/visualize_heatmap.sh \
    /root/dataset/University-Release/test/query_drone \
    ./checkpoint/DINO_QDFL_U1652.pth \
    ./model_configs/dino_b_QDFL.yaml \
    ./heatmap_visualizations \
    10 \
    280 280
```

### 方法2: 直接使用Python脚本

```bash
python visualize_heatmap.py \
    --image_dir /root/dataset/University-Release/test/query_drone \
    --checkpoint ./checkpoint/DINO_QDFL_U1652.pth \
    --config ./model_configs/dino_b_QDFL.yaml \
    --output_dir ./heatmap_visualizations \
    --num_images 10 \
    --img_size 280 280
```

## 参数说明

- `--image_dir`: 图像目录路径（必需）
  - 例如：`/root/dataset/University-Release/test/query_drone`
  
- `--checkpoint`: 模型checkpoint路径（必需）
  - 例如：`./checkpoint/DINO_QDFL_U1652.pth`
  
- `--config`: 模型配置文件路径（可选，默认：`./model_configs/dino_b_QDFL.yaml`）

- `--output_dir`: 输出目录（可选，默认：`./heatmap_visualizations`）

- `--num_images`: 要可视化的图像数量（可选，默认：10）
  - 如果目录中图像少于指定数量，会使用所有可用图像

- `--img_size`: 输入图像大小（可选，默认：280 280）
  - 格式：`height width`

## 输出说明

脚本会在输出目录生成以下文件：

1. **单张图像可视化**：`{image_name}_heatmap.png`
   - 包含三列：原图、热力图、叠加图

2. **对比图**：`comparison_all.png`
   - 所有图像的可视化结果在一个大图中，便于对比

## 热力图生成原理

1. **特征提取位置**：AQEU模块的`x_fine_0`输出
   - 这是经过FFU（Fine Feature Enhancement）处理后的特征图
   - 空间分辨率保持，语义信息丰富

2. **热力图生成方法**：
   - 对特征图在通道维度求平均
   - 归一化到[0, 1]范围
   - 上采样到原图大小
   - 使用jet colormap可视化

3. **可视化内容**：
   - **原图**：输入的原始图像
   - **热力图**：模型关注区域的强度分布
   - **叠加图**：原图与热力图的叠加，更直观地显示关注区域

## 注意事项

1. 确保checkpoint文件存在且与配置文件匹配
2. 图像目录应包含至少一张图像
3. 如果GPU内存不足，可以减少`--num_images`参数
4. 图像预处理与测试时保持一致（Resize + Normalize）

## 故障排除

### 问题1: 无法捕获特征图

**现象**：输出"Warning: Failed to capture x_fine_0"

**解决**：
- 检查模型结构是否正确
- 确认checkpoint与配置文件匹配
- 查看错误日志了解详细信息

### 问题2: 图像加载失败

**现象**：某些图像无法加载

**解决**：
- 检查图像格式是否支持（jpg, png, bmp等）
- 确认图像文件未损坏
- 查看具体错误信息

### 问题3: 内存不足

**现象**：CUDA out of memory

**解决**：
- 减少`--num_images`参数
- 减小`--img_size`
- 使用CPU模式（修改代码中的device设置）

## 示例输出

运行成功后，会在输出目录看到类似以下文件：

```
heatmap_visualizations/
├── 0001_heatmap.png
├── 0002_heatmap.png
├── ...
├── 0010_heatmap.png
└── comparison_all.png
```

每个`*_heatmap.png`文件包含三列：
- 左：原始图像
- 中：热力图（红色表示高关注度）
- 右：叠加图（热力图叠加在原图上）

## 技术细节

- **特征图位置**：`model/components/QDFL.py` 第263行，AQEU模块的`x_fine_0`输出
- **Hook机制**：使用PyTorch的forward hook自动捕获中间特征
- **备用方案**：如果hook失败，会尝试直接调用AQEU模块提取特征
