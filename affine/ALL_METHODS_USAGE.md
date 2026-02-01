# 所有倾斜校正方法使用指南

本文档介绍如何使用所有6种自适应倾斜校正方法。

## 文件说明

- `all_skew_correction_methods.py` - 包含所有6种方法的实现
- `test_all_methods.py` - 测试脚本，可以一次性测试所有方法并生成结果图

## 快速开始

### 方法1: 使用测试脚本（推荐）

```bash
# 基本用法
python test_all_methods.py your_image.jpg

# 指定输出目录
python test_all_methods.py your_image.jpg output_folder
```

这将：
1. 使用所有6种方法对图像进行校正
2. 生成每个方法的单独结果图
3. 生成一个对比图，显示原图和所有校正结果
4. 打印详细的检测信息

### 方法2: 在Python代码中使用

```python
import cv2
from all_skew_correction_methods import correct_skew_all_methods, METHODS

# 读取图像
image = cv2.imread('your_image.jpg')

# 使用所有方法进行校正
results = correct_skew_all_methods(image)

# 访问特定方法的结果
for method_key, (method_name, _) in METHODS.items():
    if method_key in results:
        corrected, info = results[method_key]
        angle = info.get('detected_angle', 0.0)
        print(f"{method_name}: 检测角度 = {angle:.2f}度")
        
        # 保存结果
        cv2.imwrite(f'result_{method_key}.jpg', corrected)
```

### 方法3: 使用单个方法

```python
import cv2
from all_skew_correction_methods import (
    correct_skew_method1,  # 投影轮廓法
    correct_skew_method2,  # 霍夫直线检测法
    correct_skew_method3,  # 最小外接矩形法
    correct_skew_method4,  # 投影变换+旋转组合法
    correct_skew_method5,  # 频域分析法
    correct_skew_method6,  # 组合方法
)

image = cv2.imread('your_image.jpg')

# 使用方案1（投影轮廓法）
corrected1, info1 = correct_skew_method1(image)
print(f"方案1检测角度: {info1['detected_angle']:.2f}度")

# 使用方案2（霍夫直线检测法）
corrected2, info2 = correct_skew_method2(image)
print(f"方案2检测角度: {info2['detected_angle']:.2f}度")

# ... 其他方法类似
```

## 6种方法说明

### 方案1: 投影轮廓法（Projection Profile）

**原理**: 通过旋转图像并计算水平投影的方差，找到方差最大的角度。

**优点**:
- 速度快
- 对文档、文字图像效果好

**适用场景**:
- 文档扫描
- 文字图像
- 表格图像

**使用示例**:
```python
corrected, info = correct_skew_method1(image, angle_range=(-30, 30), angle_step=0.5)
```

---

### 方案2: 霍夫直线检测法（Hough Line Detection）

**原理**: 使用霍夫变换检测图像中的直线，统计角度分布。

**优点**:
- 鲁棒性好
- 适用场景广泛

**适用场景**:
- 通用场景
- 有直线特征的图像

**使用示例**:
```python
corrected, info = correct_skew_method2(image)
```

---

### 方案3: 最小外接矩形法（Minimum Bounding Rectangle）

**原理**: 检测图像中的主要轮廓，计算最小外接矩形的角度。

**优点**:
- 实现简单
- 对有明显边界的对象效果好

**适用场景**:
- 文档扫描（有明显边界）
- 矩形对象

**使用示例**:
```python
corrected, info = correct_skew_method3(image)
```

---

### 方案4: 投影变换+旋转组合法（Perspective + Rotation）

**原理**: 先使用透视变换校正透视变形，再检测并校正倾斜。

**优点**:
- 能处理复杂的变形
- 适合文档扫描场景

**适用场景**:
- 文档扫描（有透视变形）
- 需要全面校正的场景

**使用示例**:
```python
corrected, info = correct_skew_method4(image)
```

---

### 方案5: 频域分析法（Frequency Domain Analysis）

**原理**: 使用FFT变换将图像转换到频域，分析主方向。

**优点**:
- 对周期性纹理敏感
- 不受局部噪声影响

**适用场景**:
- 有周期性纹理的图像
- 文字行明显的文档

**使用示例**:
```python
corrected, info = correct_skew_method5(image)
```

---

### 方案6: 组合方法（Combined Methods）

**原理**: 组合多种方法，取中位数作为最终角度。

**优点**:
- 最准确
- 最鲁棒

**适用场景**:
- 对准确度要求高的场景
- 不确定哪种方法最适合时

**使用示例**:
```python
corrected, info = correct_skew_method6(image)
# info['all_angles'] 包含所有方法检测到的角度
# info['methods_used'] 包含成功使用的方法列表
```

---

## 输出说明

### 测试脚本输出

运行 `test_all_methods.py` 后，会在输出目录中生成：

```
output_directory/
├── 00_original.jpg                    # 原始图像
├── 1_method1_投影轮廓法.jpg           # 方案1结果
├── 2_method2_霍夫直线检测法.jpg       # 方案2结果
├── 3_method3_最小外接矩形法.jpg       # 方案3结果
├── 4_method4_投影变换_旋转组合法.jpg  # 方案4结果
├── 5_method5_频域分析法.jpg           # 方案5结果
├── 6_method6_组合方法.jpg             # 方案6结果
└── comparison_all_methods.jpg         # 对比图（所有结果）
```

### 信息字典结构

每个方法返回的信息字典包含：

```python
{
    'method': 'method1',              # 方法标识
    'detected_angle': 5.23,          # 检测到的角度（度）
    'corrected_angle': -5.23,        # 应用的校正角度（度）
    'success': True,                 # 是否成功
    # ... 方法特定的信息
}
```

**方案1额外信息**:
- `max_variance`: 最大方差值
- `angle_range`: 搜索的角度范围
- `angle_step`: 角度搜索步长

**方案3额外信息**:
- `rect_size`: 最小外接矩形尺寸

**方案4额外信息**:
- `perspective_corrected`: 是否进行了透视校正
- `hough_info`: 霍夫检测的详细信息

**方案6额外信息**:
- `methods_used`: 成功使用的方法列表
- `all_angles`: 所有方法检测到的角度列表
- `method_results`: 每个方法的详细结果

---

## 参数调优

### 方案1（投影轮廓法）参数

```python
corrected, info = correct_skew_method1(
    image,
    angle_range=(-45, 45),  # 角度搜索范围
    angle_step=0.5          # 角度搜索步长（越小越精确但越慢）
)
```

### 角度阈值

所有方法都支持角度阈值，小于此值不进行校正：

```python
# 在内部实现中，如果检测角度小于0.5度，通常不进行校正
# 可以通过修改代码来调整这个阈值
```

---

## 性能对比

| 方法 | 速度 | 准确度 | 适用场景 |
|------|------|--------|---------|
| 方案1 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 文档、文字 |
| 方案2 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 通用场景 |
| 方案3 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 有明显边界 |
| 方案4 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 复杂变形 |
| 方案5 | ⭐⭐⭐ | ⭐⭐⭐ | 周期性纹理 |
| 方案6 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 高精度需求 |

---

## 常见问题

### Q: 哪种方法最好？

A: 取决于您的图像类型：
- **文档/文字**: 方案1或方案2
- **有明显边界**: 方案3
- **复杂变形**: 方案4
- **不确定**: 方案6（组合方法）

### Q: 如何选择最佳方法？

A: 使用 `test_all_methods.py` 测试所有方法，然后根据结果选择最适合的。

### Q: 检测角度不准确怎么办？

A: 
1. 尝试方案6（组合方法），它通常最准确
2. 调整方案1的参数（减小angle_step）
3. 预处理图像（增强对比度、去噪）

### Q: 处理速度慢怎么办？

A:
1. 使用方案1（最快）
2. 缩小输入图像尺寸
3. 调整方案1的angle_step参数（增大步长）

---

## 示例代码

### 完整示例

```python
#!/usr/bin/env python3
import cv2
import sys
from all_skew_correction_methods import correct_skew_all_methods

def main():
    if len(sys.argv) < 2:
        print("用法: python example.py <图像路径>")
        sys.exit(1)
    
    # 读取图像
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        sys.exit(1)
    
    # 使用所有方法
    results = correct_skew_all_methods(image)
    
    # 保存结果
    for method_key, (corrected, info) in results.items():
        angle = info.get('detected_angle', 0.0)
        output_path = f'corrected_{method_key}.jpg'
        cv2.imwrite(output_path, corrected)
        print(f"{method_key}: 角度={angle:.2f}度, 已保存到 {output_path}")

if __name__ == '__main__':
    main()
```

---

## 依赖要求

```bash
pip install opencv-python numpy
```

可选（用于方案4的透视校正）:
```bash
# 如果使用现有的solution4_perspective_correction.py
# 确保该文件在同一目录下
```

---

## 更多信息

- 详细方案分析: 查看 `ADAPTIVE_SKEW_CORRECTION_PLANS.md`
- 现有方案: 查看 `solution2_edge_detection.py` 和 `solution4_perspective_correction.py`
