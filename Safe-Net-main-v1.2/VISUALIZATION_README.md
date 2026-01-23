# 可视化工具使用说明

本工具用于生成论文实验图，展示Safe-Net模型在不同处理阶段（Input, Aligned, Partition, Heatmap）的可视化结果。

## 文件说明

### 核心文件

1. **`models/model.py`** - 修改后的模型代码
   - 添加了 `save_intermediate` 参数，用于保存中间输出
   - 在 `forward` 函数中保存 `theta`（affine变换参数）、`boundaries`（分区边界）、`f_p_aligned`（对齐后特征图）

2. **`models/create_model.py`** - 模型创建函数
   - 支持 `save_intermediate` 参数传递

3. **`utils/visualize.py`** - 可视化工具函数
   - `align_image()`: 根据FAM输出的theta参数生成对齐图
   - `draw_partition()`: 在Aligned图上绘制FPM输出的分区边界
   - `generate_heatmap()`: 根据FAM输出的对齐后特征图生成热力图

4. **`visualize_demo.py`** - 主可视化脚本
   - 生成4x5网格的实验图
   - 4行：Input, Aligned, Partition, Heatmap
   - 5列：Satellite-view + 4个Drone-view

## 使用方法

### 基本用法

```bash
python visualize_demo.py \
    --name SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \
    --satellite_img /path/to/satellite/image.jpg \
    --drone_imgs /path/to/drone1.jpg,/path/to/drone2.jpg,/path/to/drone3.jpg,/path/to/drone4.jpg \
    --epoch 119 \
    --gpu_ids 0 \
    --output experiment_figure.png
```

### 参数说明

- `--name`: 模型名称（用于查找checkpoint目录）
- `--satellite_img`: 卫星图像路径（必需）
- `--drone_imgs`: 无人机图像路径列表，用逗号分隔，至少4张（必需）
- `--epoch`: 使用的模型epoch（默认：119）
- `--gpu_ids`: GPU ID（默认：'0'）
- `--h`: 图像高度（默认：256）
- `--w`: 图像宽度（默认：256）
- `--checkpoint_dir`: checkpoint目录路径（可选，默认使用checkpoints/name）
- `--output`: 输出图像路径（默认：experiment_figure.png）

### 示例

```bash
# 使用默认checkpoint目录
python visualize_demo.py \
    --satellite_img data/satellite/001.jpg \
    --drone_imgs data/drone/001_1.jpg,data/drone/001_2.jpg,data/drone/001_3.jpg,data/drone/001_4.jpg \
    --epoch 119

# 指定checkpoint目录
python visualize_demo.py \
    --satellite_img data/satellite/001.jpg \
    --drone_imgs data/drone/001_1.jpg,data/drone/001_2.jpg,data/drone/001_3.jpg,data/drone/001_4.jpg \
    --checkpoint_dir /path/to/checkpoints/SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \
    --epoch 119 \
    --output my_experiment_figure.png
```

## 工作原理

### 1. Aligned图（对齐图）生成

- **输入**: 原始图像 + FAM模块输出的 `theta` 参数（affine变换矩阵）
- **过程**: 
  1. 模型前向传播，FAM模块生成 `theta`（2x3 affine变换矩阵）
  2. `align_image()` 函数将 `theta` 转换为图像空间的affine变换
  3. 对原始图像应用affine变换，生成对齐图
- **效果**: 消除不同视角/距离的尺度偏差，统一图像方向

### 2. Partition图（分区图）生成

- **输入**: Aligned图 + FPM模块输出的 `boundaries` 参数
- **过程**:
  1. FPM模块计算分区边界 `boundaries`（如 [b1, b2, b3]）
  2. `draw_partition()` 函数将特征图空间的边界映射到图像像素空间
  3. 在Aligned图上绘制蓝色虚线矩形框，标记4个语义一致的区域
- **效果**: 可视化特征分区结果

### 3. Heatmap（热力图）生成

- **输入**: FAM模块输出的对齐后特征图 `f_p_aligned`
- **过程**:
  1. 对特征图沿通道维度求和，得到16x16的单通道特征图
  2. 归一化到0-255范围
  3. 上采样到图像尺寸（如256x256）
  4. 应用JET配色方案（红/橙=高响应，蓝=低响应）
- **效果**: 可视化模型关注的重点区域（如目标建筑）

## 输出格式

生成的图像为4x5网格布局：

```
        Satellite  Drone-1  Drone-2  Drone-3  Drone-4
Input      [图]      [图]     [图]     [图]     [图]
Aligned    [图]      [图]     [图]     [图]     [图]
Partition  [图]      [图]     [图]     [图]     [图]
Heatmap    [图]      [图]     [图]     [图]     [图]
```

图像保存为PNG格式，默认DPI为300，适合论文使用。

## 注意事项

1. **模型要求**: 确保模型checkpoint文件存在且与代码版本匹配
2. **图像数量**: 至少需要1张卫星图和4张无人机图
3. **图像格式**: 支持常见图像格式（JPG, PNG等）
4. **GPU内存**: 如果GPU内存不足，可以减少batch size或使用CPU（修改代码）
5. **中间输出**: 脚本会自动设置 `save_intermediate=True`，无需手动配置

## 故障排除

### 问题1: 无法获取中间输出

**原因**: 模型未启用 `save_intermediate` 参数

**解决**: 确保使用修改后的模型代码，脚本会自动设置该参数

### 问题2: 对齐图效果不明显

**原因**: theta参数可能接近单位矩阵（无变换）

**解决**: 这是正常的，说明图像已经对齐或变换很小

### 问题3: 分区边界未显示

**原因**: boundaries可能为空或格式不正确

**解决**: 检查模型输出，确保FPM模块正常工作

### 问题4: 热力图全黑或全白

**原因**: 特征图值域异常

**解决**: 检查特征图归一化过程，可能需要调整归一化方法

## 代码结构

```
Safe-Net-main-v1.2/
├── models/
│   ├── model.py              # 修改：添加save_intermediate支持
│   └── create_model.py       # 修改：支持save_intermediate参数
├── utils/
│   ├── visualize.py          # 新增：可视化工具函数
│   └── utils_server.py       # 修改：load_network支持save_intermediate
├── visualize_demo.py         # 新增：主可视化脚本
└── VISUALIZATION_README.md    # 本文件
```

## 引用

如果使用本工具生成论文图片，请引用原始Safe-Net论文。
