# 原图与FAM对齐图对比工具使用说明

## 功能说明

`compare_alignment.py` 脚本用于将原始图像和FAM对齐后的图像并排显示，方便直观对比FAM模块的对齐效果。

## 使用方法

### 基本用法

```bash
# 对比单张图像
python compare_alignment.py \
    --img path/to/image.jpg \
    --epoch 119

# 对比多张图像（逗号分隔）
python compare_alignment.py \
    --img path/to/image1.jpg,path/to/image2.jpg,path/to/image3.jpg \
    --epoch 119
```

### 完整参数说明

```bash
python compare_alignment.py \
    --name SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \  # 模型名称
    --img path/to/image.jpg \                                 # 图像路径（单张或多张，逗号分隔）
    --epoch 119 \                                             # 模型epoch
    --gpu_ids 0 \                                             # GPU ID
    --h 256 \                                                 # 图像高度
    --w 256 \                                                 # 图像宽度
    --checkpoint_dir path/to/checkpoints \                    # checkpoint目录（可选）
    --output alignment_comparison.png \                      # 输出图像路径
    --mode 1 \                                                # 1: drone->satellite, 2: satellite->drone
    --debug                                                   # 打印详细theta参数信息
```

### 参数说明

- `--name`: 模型名称，默认使用 `SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652`
- `--img`: **必需**，图像路径。可以是单张图像，或多张图像用逗号分隔
- `--epoch`: 使用的模型epoch，默认119
- `--gpu_ids`: GPU ID，默认'0'
- `--h`, `--w`: 图像尺寸，默认256x256
- `--checkpoint_dir`: checkpoint目录路径，如果不指定则使用 `checkpoints/{name}`
- `--output`: 输出图像路径，默认 `alignment_comparison.png`
- `--mode`: 模式选择
  - `1`: drone->satellite（输入为无人机图像）
  - `2`: satellite->drone（输入为卫星图像）
- `--debug`: 打印详细的theta参数信息（旋转角度、缩放、平移等）

## 使用示例

### 示例1：对比单张无人机图像

```bash
python compare_alignment.py \
    --img data/University-Release/test/query_drone/001/001_001.jpg \
    --epoch 119 \
    --mode 1 \
    --output drone_alignment_comparison.png
```

### 示例2：对比多张无人机图像

```bash
python compare_alignment.py \
    --img data/University-Release/test/query_drone/001/001_001.jpg,\
           data/University-Release/test/query_drone/001/001_002.jpg,\
           data/University-Release/test/query_drone/001/001_003.jpg \
    --epoch 119 \
    --mode 1 \
    --output multiple_drone_comparison.png
```

### 示例3：对比卫星图像（通常对齐效果不明显）

```bash
python compare_alignment.py \
    --img data/University-Release/test/query_satellite/001/001.jpg \
    --epoch 119 \
    --mode 2 \
    --output satellite_alignment_comparison.png \
    --debug
```

### 示例4：使用自定义checkpoint目录

```bash
python compare_alignment.py \
    --img path/to/image.jpg \
    --checkpoint_dir /path/to/custom/checkpoints/SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \
    --epoch 119
```

## 输出说明

### 单张图像输出

- **左图**：原始图像（Original Image）
- **右图**：FAM对齐后的图像（Aligned Image）
- **底部信息**：显示旋转角度、缩放、平移等参数

### 多张图像输出

- **左列**：所有原始图像
- **右列**：所有对齐后的图像
- **每行下方**：显示该图像的对齐参数信息

### 对齐效果等级

- **不明显**：theta接近单位矩阵（diff < 0.01），图像几乎不变
- **中等**：theta有轻微变化（0.01 < diff < 0.1）
- **明显**：theta有明显变化（diff > 0.1），对齐效果显著

## 输出信息解读

### Theta参数

- **旋转角度（Rotation）**：图像旋转的角度（度）
  - 接近0°：无旋转
  - 偏离0°：有明显旋转校正
  
- **缩放（Scale）**：特征缩放因子
  - 接近1.0：无缩放
  - < 1.0：特征被放大（通常用于小目标）
  - > 1.0：特征被缩小（通常用于大目标）
  
- **平移（Translation）**：特征平移量
  - 接近(0, 0)：无平移
  - 偏离(0, 0)：有明显平移校正

### 对齐效果判断

- **倾斜视角的无人机图像**：通常有明显旋转和剪切校正
- **小目标建筑物图像**：通常有明显缩放校正（scale < 1.0）
- **接近正射的卫星图像**：通常theta接近单位矩阵，对齐效果不明显

## 常见问题

### Q1: 为什么对齐图和原图看起来一样？

**A**: 这是正常的！如果图像已经接近正射视角，FAM会输出接近单位矩阵的theta，变换效果很小。这说明图像已经对齐或不需要大的变换。

### Q2: 如何判断对齐效果是否明显？

**A**: 查看输出信息中的"Effect"等级：
- **明显**：对齐效果显著，图像有明显变化
- **中等**：对齐效果中等，图像有轻微变化
- **不明显**：对齐效果不明显，图像几乎不变（这是正常的）

### Q3: 如何查看详细的theta参数？

**A**: 使用 `--debug` 参数，会在控制台打印详细的theta矩阵和变换参数。

### Q4: 支持哪些图像格式？

**A**: 支持PIL Image支持的所有格式（JPG, PNG, BMP等）。

## 技术细节

### 工作原理

1. **加载模型**：从checkpoint加载训练好的Safe-Net模型
2. **图像预处理**：将图像resize到256x256并归一化
3. **前向传播**：通过模型获取FAM模块的theta参数
4. **生成对齐图**：使用theta对原图应用affine变换
5. **可视化对比**：将原图和对齐图并排显示

### 代码结构

- `process_single_image()`: 处理单张图像，获取原图和对齐图
- `align_image()`: 使用theta参数生成对齐图（来自utils/visualize.py）
- 使用matplotlib创建并排对比图

## 相关文件

- `compare_alignment.py`: 主脚本
- `utils/visualize.py`: 包含align_image函数
- `utils/utils_server.py`: 包含load_network函数
- `FAM_ALIGNMENT_DETAILED_EXPLANATION.md`: FAM对齐详细说明文档
