# 改进的自动裁剪功能

## 问题

之前的裁剪功能可能不够彻底，校正后的图像仍有留白。

## 改进内容

### 1. 更智能的裁剪算法

新的裁剪算法使用**行/列扫描**方法，而不是简单的边界框检测：

- **逐行扫描**：从上到下找到第一个有足够非白色像素的行
- **逐列扫描**：从左到右找到第一个有足够非白色像素的列
- **更敏感**：只要行/列中有1%的像素是非白色，就认为该行/列有内容

### 2. 更低的阈值

- **之前**：threshold=250（只裁剪非常白的区域）
- **现在**：threshold=240（更敏感，也会裁剪浅灰色区域）

### 3. 零边距裁剪

- **之前**：可能有边距
- **现在**：margin=0，完全裁剪到内容边缘

## 效果

**之前**：
```
图像: 1000x800
内容: 800x600（被留白包围）
```

**现在**：
```
图像: 约 800x600（完全裁剪，无留白）✅
```

## 使用方法

自动裁剪默认启用，无需额外操作：

```python
from all_skew_correction_methods import correct_skew_method6

# 自动裁剪（默认，已改进）
corrected, info = correct_skew_method6(image)
```

## 技术细节

### 裁剪算法流程

1. **转换为灰度图**
2. **创建掩码**：非白色区域（gray < 240）
3. **行扫描**：
   - 从上到下：找到第一个 `row_sums[i] > width * 0.01` 的行
   - 从下到上：找到第一个 `row_sums[i] > width * 0.01` 的行
4. **列扫描**：
   - 从左到右：找到第一个 `col_sums[i] > height * 0.01` 的列
   - 从右到左：找到第一个 `col_sums[i] > height * 0.01` 的列
5. **裁剪**：使用找到的边界裁剪图像

### 参数说明

```python
def crop_white_borders(image, threshold=240, margin=0):
    """
    threshold: 白色阈值（0-255）
        - 更小的值：更严格（只裁剪非常白的区域）
        - 更大的值：更宽松（也会裁剪浅灰色区域）
        - 默认240：平衡效果
    
    margin: 保留的边距（像素）
        - 0：完全裁剪到内容边缘（默认）
        - >0：保留指定像素的边距
    """
```

## 如果仍有留白

如果改进后的裁剪仍有留白，可以：

### 方法1: 调整阈值

```python
# 在 all_skew_correction_methods.py 中修改
# 将 threshold=240 改为更小的值，如 threshold=230
```

### 方法2: 手动裁剪

```python
import cv2
from all_skew_correction_methods import correct_skew_method6, crop_white_borders

corrected, info = correct_skew_method6(image)

# 使用更激进的裁剪
more_cropped = crop_white_borders(corrected, threshold=230, margin=0)
```

### 方法3: 检查图像

如果图像边缘有接近白色的内容（如浅色建筑物、云朵等），可能被误认为是留白。这种情况下：
- 降低阈值（如 threshold=220）
- 或禁用自动裁剪，手动处理

## 测试

重新运行测试脚本，查看效果：

```bash
python test_all_methods.py your_image.jpg
```

或使用Shell脚本：

```bash
./correct_skew_all_methods.sh your_image.jpg
```

现在所有校正结果都应该没有留白了！
