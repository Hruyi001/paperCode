# 批量生成消融热力图使用说明

## 功能说明

这个脚本可以批量处理数据集中的所有drone图像，为每张图像生成对应的消融热力图。

## 文件说明

- `generate_heatmap_batch.py`: 批量处理主脚本
- `run_heatmap_batch.sh`: 便捷运行脚本

## 使用方法

### 方法1: 使用便捷脚本（推荐）

```bash
# 基本用法（使用脚本中的默认路径）
./run_heatmap_batch.sh

# 指定路径
./run_heatmap_batch.sh \
    --drone_dir /datasets/University-Release/test/query_drone \
    --satellite_dir /datasets/University-Release/test/gallery_satellite

# 限制处理数量（用于测试）
./run_heatmap_batch.sh --max_images 10
```

### 方法2: 直接使用Python脚本

```bash
python generate_heatmap_batch.py \
    --drone_dir /datasets/University-Release/test/query_drone \
    --satellite_dir /datasets/University-Release/test/gallery_satellite \
    --name CCR_Model_University \
    --output_dir ./heatmap_results_batch
```

## 参数说明

### 必需参数

- `--drone_dir`: drone图像目录路径
- `--satellite_dir`: satellite图像目录路径

### 可选参数

- `--name`: 模型名称（默认: CCR_Model_University）
- `--model_without_cib`: 无CIB的模型名称（如果不同）
- `--gpu_ids`: GPU ID（默认: 0）
- `--output_dir`: 输出目录（默认: ./heatmap_results_batch）
- `--max_images`: 最大处理图像数量（用于测试，留空表示处理所有）
- `--h`: 图像高度（默认: 384）
- `--w`: 图像宽度（默认: 384）

## 图像匹配规则

脚本会自动匹配drone图像和对应的satellite图像：

- **Drone图像路径**: `query_drone/{location_id}/image-xx.jpeg`
- **Satellite图像路径**: `gallery_satellite/{location_id}/{location_id}.jpg`

例如：
- Drone: `/datasets/.../query_drone/0001/image-06.jpeg`
- Satellite: `/datasets/.../gallery_satellite/0001/0001.jpg`

如果找不到 `{location_id}.jpg`，脚本会尝试找该目录下的第一个jpg/jpeg文件。

## 输出结果

- **输出目录**: 在 `--output_dir` 指定的目录下
- **文件命名**: `{location_id}_{image_name}_ablation.png`
  - 例如: `0001_image-06_ablation.png`
- **文件格式**: PNG格式，2×3网格布局的热力图对比

## 处理流程

1. **扫描drone目录**: 递归查找所有jpg/jpeg/png图像
2. **匹配satellite图像**: 根据location_id找到对应的satellite图像
3. **生成热力图**: 为每对图像生成消融对比热力图
4. **跳过已存在**: 如果输出文件已存在，自动跳过
5. **统计报告**: 处理完成后显示成功/失败统计

## 示例

### 处理所有图像

```bash
./run_heatmap_batch.sh
```

### 测试模式（只处理前10张）

```bash
./run_heatmap_batch.sh --max_images 10
```

### 指定不同的模型

```bash
./run_heatmap_batch.sh \
    --name CCR_Model_University \
    --model_without_cib Model_Without_CIB
```

### 指定输出目录

```bash
./run_heatmap_batch.sh \
    --output_dir ./my_heatmap_results
```

## 注意事项

1. **处理时间**: 批量处理可能需要较长时间，取决于图像数量
2. **磁盘空间**: 确保输出目录有足够的磁盘空间
3. **GPU内存**: 如果GPU内存不足，可能需要减小batch size或使用CPU
4. **断点续传**: 如果处理中断，重新运行会自动跳过已生成的文件
5. **图像匹配**: 如果某些drone图像找不到对应的satellite图像，会在报告中列出

## 故障排除

1. **找不到satellite图像**: 检查路径是否正确，确认数据集结构
2. **内存不足**: 尝试使用 `--max_images` 限制处理数量
3. **处理速度慢**: 确保使用GPU，检查GPU使用情况

## 输出示例

```
==========================================
批量热力图生成配置
==========================================
Drone目录:    /datasets/University-Release/test/query_drone
Satellite目录: /datasets/University-Release/test/gallery_satellite
模型名称:     CCR_Model_University
GPU ID:       0
输出目录:     ./heatmap_results_batch
==========================================

找到 701 张drone图像

开始批量处理...
Processing: 100%|████████████| 701/701 [15:23<00:00,  1.32s/it]

============================================================
批量处理完成!
成功: 695 张
失败: 6 张
输出目录: ./heatmap_results_batch
============================================================
```
