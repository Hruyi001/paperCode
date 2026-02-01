# 快速开始 - 所有倾斜校正方法

## 🚀 最快使用方式

```bash
# 1. 给脚本添加执行权限
chmod +x test_all_methods.py

# 2. 运行测试脚本（自动使用所有方法并生成结果图）
python test_all_methods.py your_image.jpg
```

就这么简单！脚本会自动：
- ✅ 使用所有6种方法进行校正
- ✅ 生成每个方法的单独结果图
- ✅ 生成对比图
- ✅ 显示检测角度等信息

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `all_skew_correction_methods.py` | **核心实现** - 包含所有6种方法的完整实现 |
| `test_all_methods.py` | **测试脚本** - 一键测试所有方法并生成结果图 |
| `example_use_all_methods.py` | **示例代码** - 演示如何使用 |
| `ALL_METHODS_USAGE.md` | **详细文档** - 完整的使用指南 |
| `ADAPTIVE_SKEW_CORRECTION_PLANS.md` | **方案分析** - 各方案的原理和对比 |

---

## 🎯 6种方法快速对比

| 方法 | 名称 | 速度 | 适用场景 |
|------|------|------|---------|
| **方法1** | 投影轮廓法 | ⭐⭐⭐⭐⭐ | 文档、文字 |
| **方法2** | 霍夫直线检测法 | ⭐⭐⭐ | 通用场景 |
| **方法3** | 最小外接矩形法 | ⭐⭐⭐⭐ | 有明显边界 |
| **方法4** | 投影变换+旋转 | ⭐⭐ | 复杂变形 |
| **方法5** | 频域分析法 | ⭐⭐⭐ | 周期性纹理 |
| **方法6** | 组合方法 | ⭐⭐ | **最准确**（推荐） |

---

## 💻 Python代码使用

### 方式1: 使用所有方法（推荐）

```python
import cv2
from all_skew_correction_methods import correct_skew_all_methods

image = cv2.imread('your_image.jpg')
results = correct_skew_all_methods(image)

# 访问结果
for method_key, (corrected, info) in results.items():
    angle = info.get('detected_angle', 0.0)
    print(f"{method_key}: {angle:.2f}度")
    cv2.imwrite(f'result_{method_key}.jpg', corrected)
```

### 方式2: 使用单个方法

```python
from all_skew_correction_methods import correct_skew_method6  # 组合方法（最准确）

image = cv2.imread('your_image.jpg')
corrected, info = correct_skew_method6(image)

print(f"检测角度: {info['detected_angle']:.2f}度")
cv2.imwrite('corrected.jpg', corrected)
```

---

## 📊 输出结果

运行 `test_all_methods.py` 后，输出目录结构：

```
skew_correction_results_xxx/
├── 00_original.jpg                    # 原图
├── 1_method1_投影轮廓法.jpg          # 方法1结果
├── 2_method2_霍夫直线检测法.jpg      # 方法2结果
├── 3_method3_最小外接矩形法.jpg      # 方法3结果
├── 4_method4_投影变换_旋转组合法.jpg # 方法4结果
├── 5_method5_频域分析法.jpg         # 方法5结果
├── 6_method6_组合方法.jpg            # 方法6结果
└── comparison_all_methods.jpg        # 对比图
```

---

## ⚙️ 参数调整

### 方法1（投影轮廓法）- 可调参数

```python
corrected, info = correct_skew_method1(
    image,
    angle_range=(-30, 30),  # 搜索范围（默认-45到45）
    angle_step=0.5          # 搜索步长（默认0.5，越小越精确但越慢）
)
```

其他方法通常不需要调整参数。

---

## 🎨 使用示例

### 示例1: 处理单张图像

```bash
python test_all_methods.py image.jpg
```

### 示例2: 指定输出目录

```bash
python test_all_methods.py image.jpg my_results
```

### 示例3: 运行示例代码

```bash
python example_use_all_methods.py
# 或处理真实图像
python example_use_all_methods.py your_image.jpg
```

---

## ❓ 常见问题

**Q: 哪种方法最好？**  
A: 如果不确定，使用**方法6（组合方法）**，它最准确。

**Q: 如何选择方法？**  
A: 运行 `test_all_methods.py` 测试所有方法，然后根据结果选择。

**Q: 处理速度慢？**  
A: 使用方法1（最快）或增大方法1的 `angle_step` 参数。

**Q: 检测不准确？**  
A: 使用方法6（组合方法）或预处理图像（增强对比度、去噪）。

---

## 📚 更多信息

- **详细使用指南**: 查看 `ALL_METHODS_USAGE.md`
- **方案原理分析**: 查看 `ADAPTIVE_SKEW_CORRECTION_PLANS.md`
- **代码示例**: 查看 `example_use_all_methods.py`

---

## 🔧 依赖安装

```bash
pip install opencv-python numpy
```

---

## ✅ 快速检查清单

- [ ] 已安装 `opencv-python` 和 `numpy`
- [ ] 已准备好测试图像
- [ ] 运行 `python test_all_methods.py your_image.jpg`
- [ ] 查看输出目录中的结果图
- [ ] 根据结果选择最适合的方法

---

**开始使用吧！** 🎉
