# 脚本配置指南：在脚本中设置图像路径

## 概述

现在你可以在脚本中直接设置图像路径，无需每次都在命令行输入。脚本支持两种方式：
1. **命令行参数**（优先级高）
2. **脚本内配置**（方便重复使用）

## 方式1: 简化版脚本 (align_simple.sh)

### 编辑脚本

打开 `align_simple.sh`，找到配置区域：

```bash
# ============================================
# 配置区域：在这里直接设置图像路径
# ============================================
SOURCE_IMAGE=""      # 源图像路径（需要对齐的图像）
REFERENCE_IMAGE=""    # 参考图像路径（对齐目标）
OUTPUT_IMAGE=""       # 输出图像路径（可选，留空则自动生成）
# ============================================
```

### 设置路径

```bash
SOURCE_IMAGE="/path/to/your/source.jpg"
REFERENCE_IMAGE="/path/to/your/reference.jpg"
OUTPUT_IMAGE="/path/to/output/aligned.jpg"  # 可选
```

### 使用

设置好路径后，直接运行脚本：

```bash
./align_simple.sh
```

或者仍然可以使用命令行参数（会覆盖脚本中的设置）：

```bash
./align_simple.sh other_source.jpg other_reference.jpg
```

## 方式2: 完整版脚本 (align_images.sh)

### 编辑脚本

打开 `align_images.sh`，找到配置区域：

```bash
# ============================================
# 配置区域：在这里直接设置图像路径和参数
# ============================================
SOURCE_IMAGE=""      # 源图像路径
REFERENCE_IMAGE=""    # 参考图像路径

DETECTOR="ORB"        # 特征检测器: SIFT, ORB, AKAZE
OUTPUT_DIR="output"   # 输出目录
SHOW_MATCHES=false    # 是否显示匹配点
MIN_MATCHES=4         # 最少匹配点数量
# ============================================
```

### 设置路径和参数

```bash
SOURCE_IMAGE="/path/to/your/source.jpg"
REFERENCE_IMAGE="/path/to/your/reference.jpg"

DETECTOR="SIFT"           # 使用SIFT检测器
OUTPUT_DIR="my_results"   # 输出到my_results目录
SHOW_MATCHES=true         # 显示匹配点
MIN_MATCHES=10            # 最少10个匹配点
```

### 使用

```bash
# 直接运行（使用脚本中的配置）
./align_images.sh

# 或使用命令行参数（会覆盖脚本中的设置）
./align_images.sh source.jpg reference.jpg -d ORB
```

## 完整示例

### 示例1: 基本配置

```bash
# align_simple.sh 配置区域
SOURCE_IMAGE="./images/rotated_document.jpg"
REFERENCE_IMAGE="./images/template.jpg"
OUTPUT_IMAGE="./results/aligned_document.jpg"
```

运行：
```bash
./align_simple.sh
```

### 示例2: 完整配置

```bash
# align_images.sh 配置区域
SOURCE_IMAGE="./input/source.jpg"
REFERENCE_IMAGE="./input/reference.jpg"

DETECTOR="SIFT"
OUTPUT_DIR="./aligned_results"
SHOW_MATCHES=true
MIN_MATCHES=8
```

运行：
```bash
./align_images.sh
```

### 示例3: 使用相对路径

```bash
SOURCE_IMAGE="../data/image1.jpg"
REFERENCE_IMAGE="../data/image2.jpg"
```

### 示例4: 使用绝对路径

```bash
SOURCE_IMAGE="/home/user/images/source.jpg"
REFERENCE_IMAGE="/home/user/images/reference.jpg"
```

## 优先级说明

1. **命令行参数** - 最高优先级，会覆盖脚本中的设置
2. **脚本内配置** - 如果没有命令行参数，使用脚本中的设置
3. **默认值** - 如果都没有设置，显示帮助信息

## 路径格式

支持以下路径格式：

- **相对路径**: `./images/source.jpg` 或 `../data/image.jpg`
- **绝对路径**: `/home/user/images/source.jpg`
- **带空格路径**: `"/path/with spaces/image.jpg"` (使用引号)
- **环境变量**: `"$HOME/images/source.jpg"`

## 常见问题

### Q: 如何快速切换不同的图像？

**方法1**: 在脚本中注释/取消注释不同的配置
```bash
# SOURCE_IMAGE="./images/img1.jpg"
SOURCE_IMAGE="./images/img2.jpg"  # 使用这个
```

**方法2**: 使用命令行参数覆盖
```bash
./align_simple.sh img1.jpg ref.jpg
./align_simple.sh img2.jpg ref.jpg
```

**方法3**: 创建多个配置脚本
```bash
cp align_simple.sh align_img1.sh
cp align_simple.sh align_img2.sh
# 分别编辑每个脚本的配置区域
```

### Q: 路径中包含空格怎么办？

使用引号：
```bash
SOURCE_IMAGE="/path/with spaces/image.jpg"
```

### Q: 可以使用通配符吗？

不可以，必须使用完整路径。如果需要批量处理，使用命令行参数配合循环：
```bash
for img in images/*.jpg; do
    ./align_simple.sh "$img" template.jpg
done
```

### Q: 如何验证路径是否正确？

脚本会自动检查文件是否存在，如果路径错误会显示错误信息。

## 最佳实践

1. **为不同项目创建不同的脚本副本**
   ```bash
   cp align_simple.sh project1_align.sh
   # 编辑 project1_align.sh 的配置区域
   ```

2. **使用相对路径便于项目迁移**
   ```bash
   SOURCE_IMAGE="./data/source.jpg"
   ```

3. **保留一个通用脚本用于临时测试**
   ```bash
   # 保持 SOURCE_IMAGE 和 REFERENCE_IMAGE 为空
   # 总是使用命令行参数
   ```

4. **使用版本控制时注意**
   - 不要提交包含个人路径的配置
   - 或使用 `.gitignore` 忽略配置脚本
