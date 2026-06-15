# 消融实验热力图生成工具

这个工具用于生成跨视图地理定位模型的注意力热力图，对比有CIB和没有CIB组件的模型注意力差异。

## 文件说明

- `generate_heatmap_ablation.py`: 主脚本，用于生成热力图
- `run_heatmap_ablation.sh`: 便捷运行脚本

## 使用方法

### 方法1: 使用便捷脚本（推荐）

```bash
# 基本用法
./run_heatmap_ablation.sh /path/to/drone/image.jpg /path/to/satellite/image.jpg

# 指定模型名称
./run_heatmap_ablation.sh /path/to/drone.jpg /path/to/satellite.jpg --name CCR_Model_University

# 如果有单独的无CIB模型
./run_heatmap_ablation.sh /path/to/drone.jpg /path/to/satellite.jpg \
    --name CCR_Model_University \
    --model_without_cib Model_Without_CIB

# 指定GPU和输出目录
./run_heatmap_ablation.sh /path/to/drone.jpg /path/to/satellite.jpg \
    --gpu_ids 0 \
    --output_dir ./my_results
```

### 方法2: 直接使用Python脚本

```bash
python generate_heatmap_ablation.py \
    --drone_img /path/to/drone/image.jpg \
    --satellite_img /path/to/satellite/image.jpg \
    --name CCR_Model_University \
    --gpu_ids 0 \
    --output_dir ./heatmap_results
```

## 参数说明

### 必需参数

- `--drone_img`: 无人机图像路径
- `--satellite_img`: 卫星图像路径

### 可选参数

- `--name`: 模型名称（默认: CCR_Model_University）
- `--model_without_cib`: 无CIB的模型名称（如果与有CIB的模型不同）
- `--gpu_ids`: GPU ID（默认: 0）
- `--which_epoch`: 模型检查点（默认: last）
- `--h`: 图像高度（默认: 384，会从配置文件读取）
- `--w`: 图像宽度（默认: 384，会从配置文件读取）
- `--pad`: 填充大小（默认: 0）
- `--output_dir`: 输出目录（默认: ./heatmap_results）
- `--model`: 模型类型（默认: convnext_small）

## 输出结果

脚本会在指定的输出目录中生成一个 `ablation_heatmap_comparison.png` 文件，包含：

- **第一行**: Drone图像的输入、无CIB热力图、有CIB热力图
- **第二行**: Satellite图像的输入、无CIB热力图、有CIB热力图

## 工作原理

1. **加载模型**: 从 `./model/{name}/` 目录加载训练好的模型
2. **提取注意力图**: 使用hook机制捕获模型中间层的特征图
3. **生成热力图**: 将特征图转换为热力图并叠加到原始图像上
4. **对比可视化**: 生成2x3的对比图，展示有/无CIB的差异

## 注意事项

1. 确保模型配置文件 `./model/{name}/opts.yaml` 存在
2. 如果模型结构特殊，可能需要调整 `AttentionExtractor` 类中的hook注册位置
3. 如果无法提取注意力图，脚本会尝试使用输出特征图作为替代
4. 如果有无CIB的单独模型，使用 `--model_without_cib` 参数指定

## 示例

```bash
# 使用测试数据
./run_heatmap_ablation.sh \
    /datasets/University-Release/test/query_drone/0001/0001.jpg \
    /datasets/University-Release/test/gallery_satellite/0001/0001.jpg \
    --name CCR_Model_University \
    --output_dir ./heatmap_results
```

## 故障排除

1. **找不到模型**: 检查 `./model/{name}/` 目录是否存在，以及 `opts.yaml` 文件是否存在
2. **无法提取注意力图**: 检查模型结构，可能需要调整hook注册的层
3. **图像加载失败**: 检查图像路径是否正确，图像格式是否支持（JPG, PNG等）
4. **GPU内存不足**: 尝试使用CPU或减小图像尺寸
