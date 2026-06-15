# Shell脚本数据集配置说明

## 📋 数据集配置参数

所有数据集相关的参数都集中在 `run_heatmap_visualization.sh` 脚本的开头部分，方便修改。

### 主要配置参数

#### 1. 数据集根目录
```bash
DATA_FOLDER="sample4geo/dataset"
```
这是包含所有数据集的根目录路径。

#### 2. 数据集名称
```bash
DATASET_NAME="U1652"
```
可选值：
- `"U1652"` - University-1652 数据集
- `"SUES-200"` - SUES-200 数据集
- 自定义数据集文件夹名称

#### 3. 数据集类型
```bash
DATASET_TYPE="U1652"
```
用于告诉可视化脚本如何解析数据集结构：
- `"U1652"` - University-1652 标准结构
- `"SUES-200"` - SUES-200 标准结构
- `"custom"` - 自定义数据集结构

#### 4. 任务类型
```bash
DATASET="U1652-D2S"
```
定义检索任务方向：
- `"U1652-D2S"` - Drone to Satellite（无人机图像查询，卫星图像图库）
- `"U1652-S2D"` - Satellite to Drone（卫星图像查询，无人机图像图库）

#### 5. 高度参数（仅SUES-200）
```bash
ALTITUDE=300
```
仅当使用 SUES-200 数据集时有效，可选值：`150`, `200`, `250`, `300`

## 🔧 配置示例

### 示例1: University-1652, Drone to Satellite

```bash
DATA_FOLDER="sample4geo/dataset"
DATASET_NAME="U1652"
DATASET_TYPE="U1652"
DATASET="U1652-D2S"
```

**自动生成的路径**:
- 数据集路径: `sample4geo/dataset/U1652`
- 查询路径: `sample4geo/dataset/U1652/test/query_drone`
- 图库路径: `sample4geo/dataset/U1652/test/gallery_satellite`

### 示例2: University-1652, Satellite to Drone

```bash
DATA_FOLDER="sample4geo/dataset"
DATASET_NAME="U1652"
DATASET_TYPE="U1652"
DATASET="U1652-S2D"
```

**自动生成的路径**:
- 数据集路径: `sample4geo/dataset/U1652`
- 查询路径: `sample4geo/dataset/U1652/test/query_satellite`
- 图库路径: `sample4geo/dataset/U1652/test/gallery_drone`

### 示例3: SUES-200, 300米高度, Drone to Satellite

```bash
DATA_FOLDER="sample4geo/dataset"
DATASET_NAME="SUES-200"
DATASET_TYPE="SUES-200"
DATASET="U1652-D2S"
ALTITUDE=300
```

**自动生成的路径**:
- 数据集路径: `sample4geo/dataset/SUES-200`
- 查询路径: `sample4geo/dataset/SUES-200/Testing/300/query_drone`
- 图库路径: `sample4geo/dataset/SUES-200/Testing/300/gallery_satellite`

### 示例4: SUES-200, 200米高度, Satellite to Drone

```bash
DATA_FOLDER="sample4geo/dataset"
DATASET_NAME="SUES-200"
DATASET_TYPE="SUES-200"
DATASET="U1652-S2D"
ALTITUDE=200
```

**自动生成的路径**:
- 数据集路径: `sample4geo/dataset/SUES-200`
- 查询路径: `sample4geo/dataset/SUES-200/Testing/200/query_satellite`
- 图库路径: `sample4geo/dataset/SUES-200/Testing/200/gallery_drone`

### 示例5: 自定义数据集

```bash
DATA_FOLDER="sample4geo/dataset"
DATASET_NAME="my_custom_dataset"
DATASET_TYPE="custom"
DATASET="U1652-D2S"  # 这个参数对custom类型影响较小
```

**自动生成的路径**:
- 数据集路径: `sample4geo/dataset/my_custom_dataset`
- 查询路径: `sample4geo/dataset/my_custom_dataset/query`
- 图库路径: `sample4geo/dataset/my_custom_dataset/gallery`

## 📁 数据集目录结构要求

### University-1652 标准结构

```
sample4geo/dataset/U1652/
├── train/
│   ├── drone/
│   └── satellite/
└── test/
    ├── query_drone/      # D2S任务使用
    ├── gallery_satellite/ # D2S任务使用
    ├── query_satellite/   # S2D任务使用
    └── gallery_drone/     # S2D任务使用
```

### SUES-200 标准结构

```
sample4geo/dataset/SUES-200/
├── Training/
│   ├── 150/
│   ├── 200/
│   ├── 250/
│   └── 300/
└── Testing/
    ├── 150/
    │   ├── query_drone/
    │   ├── gallery_satellite/
    │   ├── query_satellite/
    │   └── gallery_drone/
    ├── 200/
    ├── 250/
    └── 300/
```

### 自定义数据集结构

```
your_dataset/
├── query/    # 查询图像
└── gallery/  # 图库图像
```

或者任意结构，脚本会尝试自动查找图像文件。

## 🚀 快速使用

1. **编辑配置**
   
   打开 `run_heatmap_visualization.sh`，修改数据集配置部分：
   
   ```bash
   DATA_FOLDER="sample4geo/dataset"
   DATASET_NAME="U1652"
   DATASET_TYPE="U1652"
   DATASET="U1652-D2S"
   ```

2. **运行脚本**
   
   ```bash
   bash run_heatmap_visualization.sh
   ```

3. **查看结果**
   
   结果保存在 `OUTPUT_DIR` 指定的目录中（默认: `./heatmap_visualization_results`）

## ⚙️ 其他配置参数

### 模型配置
```bash
CHECKPOINT_PATH="path/to/checkpoint.pth"  # 模型权重路径
NUM_CLASSES=701                            # 类别数量
BLOCK=2                                    # Block数量
```

### 可视化配置
```bash
OUTPUT_DIR="./heatmap_visualization_results"  # 输出目录
NUM_SAMPLES=5                                 # 每个类别采样数量
IMG_SIZE=384                                  # 图像尺寸
GPU_ID=0                                     # GPU ID
```

## 🔍 路径自动构建逻辑

脚本会根据配置参数自动构建正确的数据集路径：

1. **University-1652**:
   - 根据 `DATASET` 参数选择查询和图库路径
   - D2S: `test/query_drone` 和 `test/gallery_satellite`
   - S2D: `test/query_satellite` 和 `test/gallery_drone`

2. **SUES-200**:
   - 根据 `DATASET` 和 `ALTITUDE` 参数构建路径
   - 路径格式: `Testing/{ALTITUDE}/query_*` 和 `Testing/{ALTITUDE}/gallery_*`

3. **自定义数据集**:
   - 使用数据集根路径
   - 尝试查找 `query` 和 `gallery` 子目录

## 📝 注意事项

1. **路径检查**: 脚本会自动检查数据集路径是否存在，如果不存在会提示错误
2. **权重文件**: 如果模型权重文件不存在，脚本会使用随机初始化的模型（会有警告）
3. **GPU设置**: 确保 `GPU_ID` 对应的GPU可用
4. **采样数量**: `NUM_SAMPLES` 控制每个类别处理的图像数量，可以根据需要调整

## 🐛 常见问题

### Q: 提示数据集路径不存在
**A**: 检查 `DATA_FOLDER` 和 `DATASET_NAME` 是否正确，确保数据集文件夹存在

### Q: 提示查询/图库路径不存在
**A**: 检查数据集目录结构是否符合标准格式，或使用 `DATASET_TYPE="custom"`

### Q: 如何切换不同的数据集？
**A**: 只需修改脚本开头的配置参数，然后重新运行脚本即可
