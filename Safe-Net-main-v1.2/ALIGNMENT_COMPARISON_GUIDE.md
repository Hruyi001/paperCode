# 原图与对齐图对比工具使用指南

本指南介绍如何使用便捷脚本来对比原图和FAM对齐后的图像。

## 两种使用方式

### 方式1: Shell脚本（推荐，简单快速）

使用 `run_compare_alignment.sh` 脚本，适合快速使用。

### 方式2: Python配置脚本（推荐，功能更强大）

使用 `compare_alignment_config.py` 脚本，支持更多配置选项。

---

## 方式1: Shell脚本使用

### 快速开始

1. **编辑脚本配置**（在脚本顶部）：

```bash
# 打开脚本
vim run_compare_alignment.sh

# 修改以下配置
DATASET_ROOT="/root/dataset/University-Release/test"  # 你的数据集路径
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
```

2. **选择使用方式**（在脚本中取消注释）：

**方式A: 从数据集自动选择（推荐）**
```bash
CLASS_ID="0001"      # 类别ID
VIEW_TYPE="drone"    # "drone" 或 "satellite"
NUM_IMAGES=3         # 选择几张图像
```

**方式B: 直接指定图像路径**
```bash
IMG_PATH="/path/to/image1.jpg,/path/to/image2.jpg"
```

3. **运行脚本**：
```bash
./run_compare_alignment.sh
```

### 使用示例

#### 示例1: 对比类别0001的3张无人机图像
```bash
# 在脚本中设置:
CLASS_ID="0001"
VIEW_TYPE="drone"
NUM_IMAGES=3

# 运行
./run_compare_alignment.sh
```

#### 示例2: 对比类别0001的卫星图像
```bash
# 在脚本中设置:
CLASS_ID="0001"
VIEW_TYPE="satellite"

# 运行
./run_compare_alignment.sh
```

#### 示例3: 直接指定图像路径
```bash
# 在脚本中设置:
IMG_PATH="/path/to/image1.jpg,/path/to/image2.jpg"

# 运行
./run_compare_alignment.sh
```

---

## 方式2: Python配置脚本使用

### 快速开始

1. **修改配置文件**（在脚本中的Config类）：

```python
class Config:
    # 数据集路径
    DATASET_ROOT = "/root/dataset/University-Release/test"
    
    # 模型配置
    MODEL_NAME = "SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
    EPOCH = 119
    
    # 自动选择图像
    CLASS_ID = "0001"      # 类别ID
    VIEW_TYPE = "drone"    # "drone" 或 "satellite"
    NUM_IMAGES = 3         # 选择几张图像
```

2. **运行脚本**：
```bash
python compare_alignment_config.py
```

### 命令行参数使用

Python脚本支持丰富的命令行参数，可以覆盖配置文件中的设置：

#### 示例1: 从数据集自动选择
```bash
python compare_alignment_config.py --class 0001 --view drone --num 3
```

#### 示例2: 指定图像路径
```bash
python compare_alignment_config.py --img path/to/image1.jpg,path/to/image2.jpg
```

#### 示例3: 修改数据集路径
```bash
python compare_alignment_config.py --dataset /path/to/dataset --class 0001 --view drone
```

#### 示例4: 修改模型配置
```bash
python compare_alignment_config.py --model "SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652" \
                                   --epoch 119 \
                                   --gpu 0 \
                                   --class 0001 --view drone
```

#### 示例5: 指定输出文件
```bash
python compare_alignment_config.py --class 0001 --view drone --output my_comparison.png
```

#### 示例6: 启用调试模式
```bash
python compare_alignment_config.py --class 0001 --view drone --debug
```

### 完整参数列表

```bash
python compare_alignment_config.py \
    --dataset /path/to/dataset \      # 数据集根目录
    --img path/to/image.jpg \         # 图像路径（单张或多张，逗号分隔）
    --class 0001 \                     # 类别ID
    --view drone \                     # 视图类型: drone 或 satellite
    --num 3 \                          # 选择几张图像（仅用于drone）
    --model "ModelName" \              # 模型名称
    --epoch 119 \                      # 模型epoch
    --gpu 0 \                          # GPU IDs
    --output_dir ./output \            # 输出目录
    --output comparison.png \          # 输出文件名
    --mode 1 \                         # 模式: 1=drone->satellite, 2=satellite->drone
    --debug                            # 调试模式
```

---

## 配置说明

### 数据集路径

数据集应该具有以下结构：

```
DATASET_ROOT/
├── query_satellite/
│   ├── 0001/
│   │   └── 0001.jpg
│   ├── 0002/
│   │   └── 0002.jpg
│   └── ...
└── query_drone/
    ├── 0001/
    │   ├── 0001_001.jpg
    │   ├── 0001_002.jpg
    │   └── ...
    ├── 0002/
    │   └── ...
    └── ...
```

### 模型配置

- **MODEL_NAME**: 模型名称，对应 `checkpoints/` 目录下的文件夹名
- **EPOCH**: 使用的模型epoch，通常是119或200
- **GPU_IDS**: GPU ID，多个GPU用逗号分隔，如 "0,1"

### 模式选择

- **MODE=1**: drone->satellite（输入为无人机图像）
- **MODE=2**: satellite->drone（输入为卫星图像）

脚本会自动根据VIEW_TYPE设置正确的MODE。

---

## 输出说明

### 输出位置

- 默认输出目录: `./alignment_comparisons/`
- 输出文件名格式:
  - 自动选择: `comparison_{CLASS_ID}_{VIEW_TYPE}.png`
  - 手动指定: `alignment_comparison.png` 或自定义名称

### 输出内容

- **单张图像**: 左右并排显示原图和对齐图，底部显示对齐参数
- **多张图像**: 每行显示一张图像的原图和对齐图，下方显示参数信息

### 对齐效果等级

- **明显**: theta有明显变化（diff > 0.1），对齐效果显著
- **中等**: theta有轻微变化（0.01 < diff < 0.1）
- **不明显**: theta接近单位矩阵（diff < 0.01），图像几乎不变（这是正常的）

---

## 常见使用场景

### 场景1: 快速对比某个类别的无人机图像

```bash
# Shell脚本方式
# 在脚本中设置 CLASS_ID="0001", VIEW_TYPE="drone", NUM_IMAGES=3
./run_compare_alignment.sh

# Python脚本方式
python compare_alignment_config.py --class 0001 --view drone --num 3
```

### 场景2: 对比卫星图像（通常对齐效果不明显）

```bash
# Shell脚本方式
# 在脚本中设置 CLASS_ID="0001", VIEW_TYPE="satellite"
./run_compare_alignment.sh

# Python脚本方式
python compare_alignment_config.py --class 0001 --view satellite
```

### 场景3: 对比自定义图像路径

```bash
# Shell脚本方式
# 在脚本中设置 IMG_PATH="path/to/img1.jpg,path/to/img2.jpg"
./run_compare_alignment.sh

# Python脚本方式
python compare_alignment_config.py --img path/to/img1.jpg,path/to/img2.jpg
```

### 场景4: 批量处理多个类别

创建一个简单的循环脚本：

```bash
#!/bin/bash
for class_id in 0001 0002 0003 0004 0005; do
    python compare_alignment_config.py --class $class_id --view drone --num 3
done
```

---

## 故障排除

### 问题1: 数据集路径不存在

**错误信息**: `错误: 数据集路径不存在: /path/to/dataset`

**解决方法**: 检查并修改脚本中的 `DATASET_ROOT` 配置

### 问题2: 找不到图像文件

**错误信息**: `错误: 在 xxx 中未找到图像文件`

**解决方法**: 
- 检查类别ID是否正确
- 检查数据集结构是否符合要求
- 检查图像文件扩展名（支持jpg, jpeg, png）

### 问题3: 模型文件不存在

**错误信息**: `模型文件不存在: xxx`

**解决方法**:
- 检查模型名称是否正确
- 检查epoch是否正确
- 检查checkpoint目录是否存在

### 问题4: 对齐图和原图看起来一样

**这不是问题！** 如果图像已经接近正射视角，FAM会输出接近单位矩阵的theta，这是正常行为。

---

## 文件说明

- `compare_alignment.py`: 核心对比脚本
- `run_compare_alignment.sh`: Shell便捷脚本
- `compare_alignment_config.py`: Python配置脚本
- `COMPARE_ALIGNMENT_README.md`: 详细使用说明
- `ALIGNMENT_COMPARISON_GUIDE.md`: 本指南

---

## 快速参考

### Shell脚本快速配置

```bash
# 编辑 run_compare_alignment.sh，修改以下变量：
DATASET_ROOT="/root/dataset/University-Release/test"
CLASS_ID="0001"
VIEW_TYPE="drone"
NUM_IMAGES=3
```

### Python脚本快速使用

```bash
# 最简单的方式
python compare_alignment_config.py --class 0001 --view drone --num 3

# 指定输出文件
python compare_alignment_config.py --class 0001 --view drone --output my_result.png

# 调试模式
python compare_alignment_config.py --class 0001 --view drone --debug
```

---

## 提示

1. **推荐使用Python脚本**：功能更强大，支持命令行参数，更灵活
2. **从数据集自动选择**：比手动指定路径更方便
3. **使用调试模式**：可以查看详细的theta参数信息
4. **对齐效果不明显是正常的**：如果图像已经对齐，theta会接近单位矩阵
