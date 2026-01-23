# 批量生成实验图使用说明

## 脚本功能

`batch_visualize.sh` 脚本可以批量处理整个数据集，为每个类别自动生成实验图。

## 使用方法

### 1. 处理所有类别

```bash
./batch_visualize.sh
```

这会处理数据集中的所有类别，输出文件保存在 `experiment_figures/` 目录下。

### 2. 处理指定范围的类别

```bash
# 处理类别 0070 到 0100
./batch_visualize.sh 0070 0100

# 指定输出目录
./batch_visualize.sh 0070 0100 my_output_dir
```

### 3. 从指定类别开始处理所有

```bash
# 从类别 0100 开始处理所有后续类别
./batch_visualize.sh 0100 "" my_output_dir
```

## 参数说明

```bash
./batch_visualize.sh [起始类别] [结束类别] [输出目录]
```

- **起始类别**: 可选，类别ID（如 0070），如果不指定则从第一个类别开始
- **结束类别**: 可选，类别ID（如 0100），如果不指定则处理到最后一个类别
- **输出目录**: 可选，默认为 `experiment_figures`

## 输出文件命名

每个类别的实验图会保存为：
```
输出目录/experiment_figure_类别ID.png
```

例如：
- `experiment_figures/experiment_figure_0070.png`
- `experiment_figures/experiment_figure_0071.png`
- ...

## 脚本配置

可以在脚本顶部修改以下配置：

```bash
# 数据集路径
DATASET_ROOT="/datasets/University-Release/test"
SATELLITE_DIR="${DATASET_ROOT}/query_satellite"
DRONE_DIR="${DATASET_ROOT}/query_drone"

# 模型配置
MODEL_NAME="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
EPOCH=119
GPU_IDS="0"

# 每个类别选择的无人机图片数量
NUM_DRONE_IMGS=4
```

## 数据集结构要求

脚本期望的数据集结构：

```
/datasets/University-Release/test/
├── query_satellite/
│   ├── 0070/
│   │   └── 0070.jpg
│   ├── 0071/
│   │   └── 0071.jpg
│   └── ...
└── query_drone/
    ├── 0070/
    │   ├── image-01.jpeg
    │   ├── image-02.jpeg
    │   └── ...
    ├── 0071/
    │   └── ...
    └── ...
```

## 处理逻辑

1. **自动发现类别**: 脚本会扫描 `query_satellite` 目录，获取所有类别ID
2. **验证类别**: 对每个类别检查：
   - 卫星图是否存在：`query_satellite/类别ID/类别ID.jpg`
   - 无人机图目录是否存在：`query_drone/类别ID/`
   - 是否有足够的无人机图片（至少4张）
3. **选择图片**: 自动选择每个类别目录下前4张无人机图片（按文件名排序）
4. **生成实验图**: 调用 `visualize_demo.py` 生成4×5网格的实验图

## 统计信息

脚本运行结束后会显示：
- 总类别数
- 成功生成的类别数
- 失败的类别数
- 跳过的类别数（缺少必要文件）

## 示例输出

```
==========================================
批量生成实验图 - 处理类别范围: 0070 到 0100
==========================================
总类别数: 31
输出目录: experiment_figures

[1/31] 处理类别: 0070
==========================================
处理类别: 0070
==========================================
卫星图: /datasets/University-Release/test/query_satellite/0070/0070.jpg
无人机图: /datasets/University-Release/test/query_drone/0070/image-01.jpeg,/datasets/University-Release/test/query_drone/0070/image-02.jpeg,/datasets/University-Release/test/query_drone/0070/image-03.jpeg,/datasets/University-Release/test/query_drone/0070/image-04.jpeg
输出文件: experiment_figures/experiment_figure_0070.png
✓ 成功生成: experiment_figures/experiment_figure_0070.png

...

==========================================
批量处理完成
==========================================
总类别数: 31
成功: 30
失败: 0
跳过: 1
输出目录: experiment_figures
==========================================
```

## 注意事项

1. **处理时间**: 批量处理可能需要较长时间，取决于类别数量和GPU性能
2. **磁盘空间**: 确保有足够的磁盘空间存储所有生成的图片
3. **错误处理**: 如果某个类别处理失败，脚本会继续处理下一个类别
4. **中断恢复**: 如果脚本被中断，可以指定起始类别继续处理

## 故障排除

### 问题1: 找不到图片文件
**解决**: 检查数据集路径是否正确，确保 `DATASET_ROOT` 指向正确的目录

### 问题2: 某些类别被跳过
**解决**: 检查该类别的卫星图和无人机图是否存在，以及是否有足够的无人机图片

### 问题3: 内存不足
**解决**: 可以分批处理，每次处理一定范围的类别
