# 倾斜无人机视图转正射视图校正

基于传统计算机视觉方法，将倾斜的无人机视图校正为垂直正射视图。

## 功能特点

- ✅ 自动特征匹配（SIFT/ORB）
- ✅ 单应性矩阵计算
- ✅ 倾斜角度提取
- ✅ 视角校正（不改变方向和尺度）
- ✅ 完全自动化处理

## 实现方案

```
特征匹配 → 计算单应性矩阵 → 提取角度 → 构建视角校正变换 → 应用变换
```

### 核心步骤

1. **特征匹配**: 从倾斜图和参考正射图中提取并匹配特征点
2. **单应性矩阵计算**: 使用RANSAC算法计算单应性矩阵
3. **角度提取**: 从单应性矩阵中提取俯仰角（pitch）和翻滚角（roll）
4. **构建校正变换**: 基于提取的角度构建只校正视角的变换矩阵
5. **应用变换**: 将倾斜图校正为正射视角

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### Shell脚本使用（推荐）

最简单方便的使用方式：

```bash
./ortho_correct.sh <倾斜无人机图> <参考正射图> [输出路径]
```

示例：
```bash
# 基本使用（输出默认为 corrected_output.jpg）
./ortho_correct.sh drone_tilted.jpg satellite_reference.jpg

# 指定输出路径
./ortho_correct.sh drone_tilted.jpg satellite_reference.jpg my_output.jpg
```

**特点：**
- ✅ 自动检查依赖包
- ✅ 友好的错误提示
- ✅ 彩色输出，易于查看
- ✅ 自动处理文件路径

### Python命令行使用

```bash
python ortho_correction.py <倾斜图路径> <参考正射图路径> [输出路径]
```

示例：
```bash
python ortho_correction.py tilted.jpg reference.jpg output.jpg
```

### Python代码使用

```python
from ortho_correction import OrthoCorrection
import cv2

# 读取图像
tilted_image = cv2.imread("tilted.jpg")
reference_image = cv2.imread("reference.jpg")

# 创建校正器
corrector = OrthoCorrection(feature_type='SIFT')  # 或 'ORB'

# 执行校正
corrected_image = corrector.process(tilted_image, reference_image)

# 保存结果
cv2.imwrite("corrected.jpg", corrected_image)
```

## 参数说明

### OrthoCorrection 初始化参数

- `feature_type`: 特征提取类型，可选 'SIFT' 或 'ORB'
  - SIFT: 精度高，对尺度/旋转/光照鲁棒，速度较慢
  - ORB: 速度快，免费使用，对尺度变化敏感
- `ratio_threshold`: 特征匹配的ratio test阈值（默认0.75）

## 技术细节

### 角度提取方法

从单应性矩阵中提取倾斜角度有两种方法：

1. **几何分析方法**: 分析单应性矩阵对图像角点的变换效果
2. **矩阵分解方法**: 将单应性矩阵分解为旋转矩阵，提取欧拉角

### 视角校正变换

基于提取的俯仰角和翻滚角，构建透视变换矩阵：
- 只校正视角（消除透视畸变）
- 不改变图像的方向和尺度
- 保持原图的相对位置关系

## 注意事项

1. **匹配点数量**: 至少需要4个匹配点才能计算单应性矩阵
2. **特征点质量**: 图像需要有足够的纹理特征（如建筑物、道路等）
3. **倾斜角度**: 适合中等倾斜角度（<60°），过大角度可能匹配失败
4. **参考图要求**: 参考正射图应与倾斜图有重叠区域

## 可能的问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 匹配点太少 | 特征点不足或差异大 | 尝试使用SIFT，调整ratio_threshold |
| 角度估计不准 | 匹配点分布不均 | 检查匹配点质量，增加RANSAC迭代次数 |
| 变换结果扭曲 | 单应性矩阵计算错误 | 检查输入图像质量，确保有足够重叠区域 |

## 文件结构

```
affine-stn/
├── ortho_correction.py  # 核心实现
├── ortho_correct.sh     # Shell脚本（推荐使用）
├── example.py           # Python使用示例
├── requirements.txt     # 依赖包
└── README.md            # 说明文档
```

## 许可证

MIT License
