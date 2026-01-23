# Affine变换详解

## 一、什么是Affine变换？

**Affine变换（仿射变换）**是一种保持直线平行性的几何变换，包括：
- 旋转（Rotation）
- 缩放（Scaling）
- 平移（Translation）
- 剪切（Shear）

### 1.1 数学定义

Affine变换可以用矩阵表示：

```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
[1 ]   [0  0  1 ] [1]
```

简化为：
```
x' = a*x + b*y + tx
y' = c*x + d*y + ty
```

其中：
- `a, b, c, d`：控制旋转、缩放、剪切
- `tx, ty`：控制平移

### 1.2 2×3矩阵形式

在PyTorch中，通常使用2×3矩阵（省略最后一行`[0,0,1]`）：

```
theta = [[a, b, tx],
         [c, d, ty]]
```

## 二、Affine变换的类型

### 2.1 单位矩阵（无变换）

```
[[1, 0, 0],
 [0, 1, 0]]
```
- 不改变图像
- 所有点保持不变

### 2.2 旋转（Rotation）

```
角度θ的旋转：
[[cos(θ), -sin(θ), 0],
 [sin(θ),  cos(θ), 0]]
```

**示例**：旋转30度
```
[[0.866, -0.5,  0],
 [0.5,    0.866, 0]]
```

### 2.3 缩放（Scaling）

```
x方向缩放sx，y方向缩放sy：
[[sx, 0,  0],
 [0,  sy, 0]]
```

**示例**：x方向放大2倍，y方向缩小0.5倍
```
[[2,  0,   0],
 [0,  0.5, 0]]
```

### 2.4 平移（Translation）

```
x方向平移tx，y方向平移ty：
[[1, 0, tx],
 [0, 1, ty]]
```

**示例**：向右平移10像素，向下平移5像素
```
[[1, 0, 10],
 [0, 1, 5]]
```

### 2.5 剪切（Shear）

```
x方向剪切shx，y方向剪切shy：
[[1,  shx, 0],
 [shy, 1,   0]]
```

**示例**：x方向剪切0.2
```
[[1,  0.2, 0],
 [0,  1,   0]]
```

### 2.6 组合变换

通常affine变换是多种变换的组合：

```
旋转 + 缩放 + 平移：
[[sx*cos(θ), -sy*sin(θ), tx],
 [sx*sin(θ),  sy*cos(θ), ty]]
```

## 三、在Safe-Net中的应用

### 3.1 Loc_Net输出的theta

```python
# Loc_Net输出6个参数，组成2×3矩阵
A_theta = loc_net(global_feature)  # [B, 2, 3]

# 示例theta值：
theta = [[0.95, -0.05, 0.1],   # 轻微旋转+平移
         [0.05,  0.95, 0.1]]
```

### 3.2 应用变换

```python
# 1. 创建affine grid
grid = F.affine_grid(theta.unsqueeze(0), feature_map.size())

# 2. 应用grid_sample进行采样
aligned_feature = F.grid_sample(feature_map, grid)
```

### 3.3 变换的效果

**原始特征图**（16×16）：
```
[特征图，可能有旋转、缩放等]
```

**应用theta后**：
```
[对齐后的特征图，消除了视角差异]
```

## 四、可视化示例

### 4.1 图像变换示例

假设有一张倾斜的建筑物图像：

**原始图像**：
```
    /\
   /  \
  /____\
```

**应用旋转theta = [[0.7, -0.7, 0], [0.7, 0.7, 0]]**：
```
   ____
  |    |
  |____|
```
（旋转45度，变成正视图）

### 4.2 在Safe-Net中的实际效果

**无人机图像（倾斜视角）**：
- 建筑物有透视变形
- 角度不是正射

**应用FAM的theta变换后**：
- 消除透视变形
- 对齐到接近正射视角
- 便于与卫星图匹配

## 五、为什么使用Affine变换？

### 5.1 优势

1. **线性变换**：计算高效，可以用矩阵乘法实现
2. **保持平行性**：平行线变换后仍平行
3. **可逆**：可以计算逆变换
4. **参数少**：只需要6个参数（2×3矩阵）

### 5.2 局限性

1. **不能处理非线性变形**：如透视投影、鱼眼畸变
2. **不能处理局部变形**：所有区域使用相同的变换

### 5.3 为什么Safe-Net使用Affine？

- **视角差异主要是旋转、缩放、平移**：无人机和卫星图的差异主要是这些
- **计算效率高**：6个参数，计算快速
- **足够表达**：对于大多数场景，affine变换已经足够

## 六、代码示例

### 6.1 创建affine变换

```python
import torch
import torch.nn.functional as F

# 定义theta矩阵
theta = torch.tensor([
    [0.9, -0.1, 10],   # 旋转+平移
    [0.1,  0.9, 5]
], dtype=torch.float32)

# 应用到特征图
feature_map = torch.randn(1, 768, 16, 16)  # [B, C, H, W]
grid = F.affine_grid(theta.unsqueeze(0), feature_map.size())
aligned = F.grid_sample(feature_map, grid)
```

### 6.2 应用到图像

```python
from PIL import Image
import torchvision.transforms as transforms

# 加载图像
img = Image.open("drone_image.jpg")
img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]

# 应用affine变换
grid = F.affine_grid(theta.unsqueeze(0), img_tensor.size())
aligned_img = F.grid_sample(img_tensor, grid)
```

## 七、常见问题

### Q1: Affine变换和透视变换的区别？

**Affine变换**：
- 保持平行性
- 6个参数（2×3矩阵）
- 不能处理透视投影

**透视变换（Homography）**：
- 不保持平行性
- 8个参数（3×3矩阵）
- 可以处理透视投影

### Q2: 为什么theta是2×3而不是3×3？

PyTorch的`F.affine_grid`使用2×3矩阵，最后一行`[0,0,1]`是固定的，不需要存储。

### Q3: 如何理解theta的值？

- **接近单位矩阵**：变换很小，图像几乎不变
- **旋转角度大**：图像会明显旋转
- **缩放值≠1**：图像会放大或缩小
- **平移值大**：图像会明显移动

### Q4: 如何可视化theta的效果？

```python
# 打印theta值
print(f"Theta: {theta}")

# 计算旋转角度
angle = math.atan2(theta[1,0], theta[0,0]) * 180 / math.pi
print(f"旋转角度: {angle:.2f}°")

# 计算缩放
scale = math.sqrt(theta[0,0]**2 + theta[1,0]**2)
print(f"缩放: {scale:.4f}")

# 平移
print(f"平移: ({theta[0,2]:.2f}, {theta[1,2]:.2f})")
```

## 八、总结

### 8.1 核心概念

- **Affine变换**：保持平行性的线性几何变换
- **6个参数**：旋转、缩放、平移、剪切的组合
- **2×3矩阵**：PyTorch中的表示形式

### 8.2 在Safe-Net中的作用

1. **Loc_Net学习theta**：根据图像特征学习变换参数
2. **应用到特征图**：消除视角差异
3. **统一特征空间**：便于匹配和检索

### 8.3 实际效果

- 将不同视角的无人机图像对齐到标准空间
- 消除旋转、缩放、平移等差异
- 使特征表示更一致，便于匹配
