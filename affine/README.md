# 图像仿射变换对齐方案

本项目提供了多种对倾斜图像进行仿射变换和对齐的实现方案。

## 方案概述

### 方案1: 基于特征点匹配的自动对齐 (OpenCV)
- 使用SIFT/ORB特征检测器找到特征点
- 通过特征匹配找到对应关系
- 计算仿射变换矩阵
- 适用于有参考图像的情况

### 方案2: 基于边缘检测的倾斜校正 (OpenCV)
- 使用Canny边缘检测找到图像边缘
- 通过霍夫变换检测直线
- 计算倾斜角度
- 进行旋转校正

### 方案3: 手动指定控制点的仿射变换 (OpenCV)
- 手动选择源图像和目标图像的对应点
- 计算仿射变换矩阵
- 应用变换
- 适用于已知对应点的情况

### 方案4: 基于投影变换的透视校正 (OpenCV)
- 检测文档/矩形的四个角点
- 使用透视变换进行校正
- 适用于文档扫描等场景

### 方案5: 使用scikit-image的仿射变换
- 使用scikit-image的transform模块
- 提供更高级的变换接口
- 支持多种插值方法

## 依赖安装

```bash
pip install opencv-python numpy scikit-image matplotlib
```

## 快速使用（Shell脚本）

### 简化版（推荐）

```bash
# 给脚本添加执行权限
chmod +x align_simple.sh

# 使用
./align_simple.sh source.jpg reference.jpg
```

### 完整版（更多功能）

```bash
# 给脚本添加执行权限
chmod +x align_images.sh

# 基本使用
./align_images.sh source.jpg reference.jpg

# 使用SIFT检测器并显示匹配点
./align_images.sh source.jpg reference.jpg -d SIFT --show-matches

# 指定输出目录
./align_images.sh source.jpg reference.jpg -o results
```

详细说明请查看 [SHELL_SCRIPT_USAGE.md](SHELL_SCRIPT_USAGE.md)
