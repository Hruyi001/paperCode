# 方案5使用指南：scikit-image仿射变换

## 概述

方案5使用scikit-image的transform模块进行仿射变换，支持：
- 参数化变换（旋转、平移、缩放、剪切）
- 从对应点估计变换
- 从变换矩阵应用变换
- 高质量双三次插值

## 安装依赖

```bash
pip install scikit-image opencv-python numpy
```

## 使用方法

### 方式1: 简化版脚本（推荐快速使用）

```bash
# 基本用法：旋转图像
./align_solution5_simple.sh image.jpg 15

# 指定输出
./align_solution5_simple.sh image.jpg 15 result.jpg
```

或在脚本中设置：
```bash
SOURCE_IMAGE="./images/source.jpg"
ROTATION_ANGLE=15
OUTPUT_IMAGE="./results/output.jpg"
```

### 方式2: 完整版脚本（更多功能）

#### 参数变换

```bash
# 旋转
./align_solution5.sh image.jpg -r 15 -o result.jpg

# 旋转 + 平移
./align_solution5.sh image.jpg -r 15 -t 20,10 -o result.jpg

# 旋转 + 平移 + 缩放
./align_solution5.sh image.jpg -r 15 -t 20,10 -s 1.1 -o result.jpg

# 完整变换（旋转+平移+缩放+剪切）
./align_solution5.sh image.jpg -r 15 -t 20,10 -s 1.1 --shear 5 -o result.jpg

# 非均匀缩放
./align_solution5.sh image.jpg -s 1.2,0.9 -o result.jpg
```

#### 从对应点估计

1. 创建点文件 `points.json`:
```json
{
    "source_points": [
        [100, 100],
        [300, 100],
        [100, 300]
    ],
    "destination_points": [
        [120, 80],
        [320, 120],
        [80, 320]
    ]
}
```

2. 使用点文件：
```bash
./align_solution5.sh image.jpg -p points.json -o result.jpg
```

#### 从变换矩阵

1. 创建矩阵文件 `matrix.txt` (2x3或3x3):
```
0.866 -0.5 100
0.5   0.866 50
```

2. 使用矩阵文件：
```bash
./align_solution5.sh image.jpg -m matrix.txt -o result.jpg
```

#### 指定输出尺寸

```bash
./align_solution5.sh image.jpg -r 15 -w 800 -h 600 -o result.jpg
```

## 在脚本中配置

编辑 `align_solution5.sh` 的配置区域：

```bash
SOURCE_IMAGE="./images/source.jpg"

# 方式1: 参数变换
ROTATION=15
TRANSLATION_X=20
TRANSLATION_Y=10
SCALE=1.1
SHEAR=5

# 方式2: 从对应点
# POINTS_FILE="./points.json"

# 方式3: 从矩阵
# MATRIX_FILE="./matrix.txt"

OUTPUT_IMAGE="./results/output.jpg"
OUTPUT_WIDTH=800
OUTPUT_HEIGHT=600
```

然后运行：
```bash
./align_solution5.sh
```

## 参数说明

### 变换参数

- **旋转 (rotation)**: 角度（度），正值为逆时针
- **平移 (translation)**: X,Y坐标，例如 `20,10`
- **缩放 (scale)**: 
  - 单个值：均匀缩放，例如 `1.1`
  - 两个值：非均匀缩放，例如 `1.2,0.9`
- **剪切 (shear)**: 角度（度）

### 输出选项

- **输出路径**: 可以是文件路径或目录
- **输出宽度/高度**: 指定输出图像尺寸

## 使用示例

### 示例1: 简单旋转

```bash
./align_solution5_simple.sh photo.jpg 30 rotated.jpg
```

### 示例2: 复杂变换

```bash
./align_solution5.sh image.jpg \
    -r 15 \
    -t 50,30 \
    -s 1.2 \
    --shear 5 \
    -o transformed.jpg
```

### 示例3: 从对应点

```bash
# 创建点文件
cat > my_points.json << EOF
{
    "source_points": [[0, 0], [100, 0], [0, 100]],
    "destination_points": [[10, 10], [110, 5], [5, 110]]
}
EOF

# 应用变换
./align_solution5.sh image.jpg -p my_points.json -o result.jpg
```

### 示例4: 批量处理

```bash
for angle in 15 30 45 60; do
    ./align_solution5_simple.sh input.jpg $angle "output_${angle}deg.jpg"
done
```

## 与方案1的区别

| 特性 | 方案1 | 方案5 |
|------|-------|-------|
| 变换方式 | 自动特征匹配 | 手动指定参数 |
| 需要参考图像 | 是 | 否 |
| 适用场景 | 图像对齐 | 几何变换 |
| 插值质量 | 线性 | 双三次 |
| 速度 | 较慢（特征检测） | 较快 |

## 常见问题

### Q: 如何知道需要什么变换参数？

- **旋转角度**: 测量或估算图像的倾斜角度
- **平移**: 需要移动的像素数
- **缩放**: 需要的放大/缩小倍数
- **对应点**: 手动选择源图像和目标位置的对应点

### Q: 如何获得对应点？

可以使用图像编辑软件（如GIMP、Photoshop）或Python脚本手动选择点。

### Q: 如何从OpenCV矩阵转换？

方案5可以直接使用OpenCV的2x3变换矩阵，只需保存为文本文件即可。

### Q: 输出图像质量如何？

方案5使用双三次插值（order=3），质量比线性插值更好，但速度稍慢。

## 高级用法

### 组合多个变换

```bash
# 先旋转
./align_solution5.sh img.jpg -r 15 -o temp1.jpg

# 再缩放
./align_solution5.sh temp1.jpg -s 1.2 -o final.jpg
```

### 使用Python API

```python
from solution5_skimage import align_with_skimage_affine
import cv2

img = cv2.imread('image.jpg')

# 参数变换
params = {
    'rotation': 15,
    'translation': (20, 10),
    'scale': 1.1
}
transformed, transform = align_with_skimage_affine(img, params)
cv2.imwrite('result.jpg', transformed)
```
