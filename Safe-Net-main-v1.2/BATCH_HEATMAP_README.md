# 批量生成热力图脚本使用说明

## 功能说明

该脚本用于批量生成热力图，基于 `aligned_feature_map`（FAM模块输出的对齐后特征图）生成。

**功能特点**:
- 支持处理任意指定目录下的所有图像
- 在shell脚本中配置输入目录和输出目录
- 保持原始目录结构
- 每张热力图上标注 "w/ DSA" 标识
- 输出格式为 PNG

## 使用方法

### 方法1: 使用Shell脚本（推荐）

1. **编辑 `run_batch_heatmap.sh`**，修改以下配置：

```bash
# ========== 用户配置区域 ==========
# 输入目录：包含所有要处理的图像
INPUT_DIR="./data/University-Release/test/query_drone"

# 输出目录：保存生成的热力图
OUTPUT_DIR="./safe-net-heatmap/query_drone"

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 模式设置
# 1 = drone->satellite (用于gallery_satellite)
# 2 = satellite->drone (用于query_drone)
MODE=2

# 是否使用query_transforms（query_drone需要设置为true，gallery_satellite设置为false）
USE_QUERY_TRANSFORM=true
# =================================
```

2. **运行脚本**：

```bash
bash run_batch_heatmap.sh
```

### 方法2: 直接运行Python脚本

```bash
python generate_batch_heatmap.py \
    --input_dir ./data/University-Release/test/query_drone \
    --output_dir ./safe-net-heatmap/query_drone \
    --mode 2 \
    --use_query_transform \
    --name SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652 \
    --epoch 119 \
    --gpu_ids 0
```

## 参数说明

### Python脚本参数

- `--input_dir` (必需): 输入图像目录路径
- `--output_dir` (必需): 输出目录路径
- `--mode` (可选): 模式设置
  - `1` = drone->satellite (用于gallery_satellite)
  - `2` = satellite->drone (用于query_drone)
  - 默认: 2
- `--use_query_transform` (可选): 是否使用query_transforms
  - query_drone: 需要添加此参数
  - gallery_satellite: 不需要此参数
- `--name`: 模型名称（默认: SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652）
- `--epoch`: 模型epoch编号（默认: 119）
- `--gpu_ids`: GPU ID（默认: 0）
- `--checkpoint_dir`: checkpoint目录路径（可选，默认使用 checkpoints/name）

## 输出目录结构

输出目录会保持输入目录的结构：

```
输入目录: ./data/University-Release/test/query_drone
├── 0000/
│   ├── image-01.jpg
│   ├── image-02.jpg
│   └── ...
└── 0001/
    └── ...

输出目录: ./safe-net-heatmap/query_drone
├── 0000/
│   ├── image-01.png  (热力图)
│   ├── image-02.png  (热力图)
│   └── ...
└── 0001/
    └── ...
```

## 使用示例

### 示例1: 处理query_drone

```bash
# 在run_batch_heatmap.sh中设置：
INPUT_DIR="./data/University-Release/test/query_drone"
OUTPUT_DIR="./safe-net-heatmap/query_drone"
MODE=2
USE_QUERY_TRANSFORM=true
```

### 示例2: 处理gallery_satellite

```bash
# 在run_batch_heatmap.sh中设置：
INPUT_DIR="./data/University-Release/test/gallery_satellite"
OUTPUT_DIR="./safe-net-heatmap/gallery_satellite"
MODE=1
USE_QUERY_TRANSFORM=false
```

### 示例3: 处理自定义目录

```bash
# 在run_batch_heatmap.sh中设置：
INPUT_DIR="./my_custom_images"
OUTPUT_DIR="./my_heatmaps"
MODE=2
USE_QUERY_TRANSFORM=false
```

## 技术细节

### 热力图生成位置

基于 `aligned_feature_map`（FAM模块输出）生成热力图：
- 位置: `models/model.py` line 134, 186
- 特征图形状: [B, 768, 16, 16]
- 计算方法: L2范数（对所有通道计算L2范数）

### 图像预处理

- **使用query_transform**: 使用 `Query_transforms`（pad=0）
- **不使用query_transform**: 使用标准 `transforms`

### Mode设置

- **mode=1**: drone->satellite模式（用于gallery_satellite）
- **mode=2**: satellite->drone模式（用于query_drone）

## 注意事项

1. 确保输入目录存在且包含图像文件
2. 确保checkpoint文件存在（格式: `net_XXX.pth`）
3. 如果输出目录已存在同名文件，会跳过（不会覆盖）
4. 处理大量图像时可能需要较长时间，请耐心等待
5. 输出文件格式统一为PNG

## 示例输出

处理完成后会显示统计信息：
```
============================================================
处理完成！
总计: 1000 张图像
成功: 998 张
失败: 2 张
输出目录: ./safe-net-heatmap/query_drone
============================================================
```
