# Shell脚本使用指南

## 快速开始

### 简化版脚本（推荐新手）

```bash
# 基本用法
./align_simple.sh source.jpg reference.jpg

# 指定输出文件名
./align_simple.sh source.jpg reference.jpg result.jpg
```

### 完整版脚本（功能更多）

```bash
# 基本用法
./align_images.sh source.jpg reference.jpg

# 使用SIFT检测器
./align_images.sh source.jpg reference.jpg -d SIFT

# 显示特征点匹配
./align_images.sh source.jpg reference.jpg --show-matches

# 指定输出目录
./align_images.sh source.jpg reference.jpg -o results

# 完整示例
./align_images.sh source.jpg reference.jpg \
    -d SIFT \
    -o output \
    --show-matches \
    --min-matches 10
```

## 脚本对比

### `align_simple.sh` - 简化版
- ✅ 使用简单，只需2个参数
- ✅ 自动生成输出文件名
- ✅ 适合快速测试

### `align_images.sh` - 完整版
- ✅ 更多选项和配置
- ✅ 自动生成对比图
- ✅ 详细的输出信息
- ✅ 支持自定义输出目录
- ✅ 支持显示匹配点

## 参数说明

### align_simple.sh
```
用法: ./align_simple.sh <源图像> <参考图像> [输出图像]
```

### align_images.sh
```
用法: ./align_images.sh <源图像> <参考图像> [选项]

选项:
  -d, --detector DETECTOR    特征检测器 (SIFT|ORB|AKAZE) [默认: ORB]
  -o, --output DIR           输出目录 [默认: output]
  -m, --show-matches         显示特征点匹配可视化
  -n, --min-matches NUM      最少匹配点数量 [默认: 4]
  -h, --help                 显示帮助信息
```

## 使用示例

### 示例1: 快速对齐
```bash
./align_simple.sh rotated_image.jpg template.jpg
```

### 示例2: 使用SIFT检测器
```bash
./align_images.sh image1.jpg image2.jpg -d SIFT
```

### 示例3: 生成完整效果图
```bash
./align_images.sh source.jpg reference.jpg \
    -d ORB \
    -o my_results \
    --show-matches
```

输出文件：
- `my_results/aligned_*.jpg` - 对齐后的图像
- `my_results/comparison_*.jpg` - 对比图（源图像|参考图像|对齐结果）
- `my_results/matches_*.jpg` - 特征点匹配可视化（如果启用）

### 示例4: 批量处理
```bash
# 创建输出目录
mkdir -p aligned_results

# 批量对齐
for img in input_images/*.jpg; do
    ./align_simple.sh "$img" template.jpg "aligned_results/$(basename "$img")"
done
```

## 输出文件说明

### align_simple.sh 输出
- 单个对齐后的图像文件

### align_images.sh 输出
在指定的输出目录（默认：`output/`）中生成：

1. **aligned_*.jpg** - 对齐后的图像
2. **comparison_*.jpg** - 三图对比（源图像 | 参考图像 | 对齐结果）
3. **matches_*.jpg** - 特征点匹配可视化（如果使用 `--show-matches`）
4. **transform_*.txt** - 变换矩阵信息（如果生成）

文件名包含时间戳，避免覆盖。

## 常见问题

### Q: 脚本没有执行权限？
```bash
chmod +x align_simple.sh
chmod +x align_images.sh
```

### Q: 找不到python3？
确保已安装Python 3：
```bash
python3 --version
```

### Q: 找不到OpenCV？
安装OpenCV：
```bash
pip install opencv-python
# 或使用SIFT时
pip install opencv-contrib-python
```

### Q: 对齐失败？
1. 检查图像是否相似
2. 尝试不同的检测器：`-d SIFT` 或 `-d AKAZE`
3. 减少最少匹配点：`-n 3`
4. 预处理图像（增强对比度、去噪等）

## 高级用法

### 结合其他工具使用

```bash
# 对齐后自动打开结果
./align_simple.sh source.jpg reference.jpg result.jpg && xdg-open result.jpg

# 对齐后转换为PDF
./align_simple.sh source.jpg reference.jpg result.jpg && \
convert result.jpg result.pdf

# 对齐后压缩
./align_simple.sh source.jpg reference.jpg result.jpg && \
jpegoptim --max=80 result.jpg
```

### 在脚本中使用

```bash
#!/bin/bash
# 批量处理脚本

TEMPLATE="template.jpg"
OUTPUT_DIR="aligned"

mkdir -p "$OUTPUT_DIR"

for img in images/*.jpg; do
    name=$(basename "$img" .jpg)
    echo "处理: $name"
    
    ./align_simple.sh "$img" "$TEMPLATE" "$OUTPUT_DIR/${name}_aligned.jpg"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ $name 成功"
    else
        echo "❌ $name 失败"
    fi
done
```

## 性能提示

1. **快速处理**: 使用ORB检测器（默认）
2. **高质量**: 使用SIFT检测器
3. **批量处理**: 考虑先缩小图像尺寸
4. **并行处理**: 使用GNU parallel加速批量处理

```bash
# 使用parallel并行处理
parallel -j 4 ./align_simple.sh {} template.jpg aligned_{/} ::: images/*.jpg
```
