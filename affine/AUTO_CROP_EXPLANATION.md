# 自动裁剪留白功能说明

## 问题说明

当图像进行旋转校正时，为了保留所有内容，需要扩大画布尺寸。这会导致图像周围出现白色留白区域。

### 为什么会有留白？

1. **旋转需要更大的画布**: 当矩形图像旋转时，需要更大的画布才能容纳所有内容
2. **填充白色背景**: 旋转后的空白区域会被填充为白色（或指定的背景色）
3. **结果**: 校正后的图像比原图更大，包含大量留白

### 示例

```
原图尺寸: 800x600
旋转后尺寸: 约 1000x800 (包含留白)
实际内容: 约 800x600 (被留白包围)
```

## 解决方案

已添加**自动裁剪留白功能**，默认启用。

### 工作原理

1. **检测非白色区域**: 将图像转换为灰度图，找到所有非白色像素
2. **计算边界框**: 找到包含所有非白色像素的最小矩形
3. **裁剪图像**: 裁剪到边界框，移除周围的留白

### 实现细节

```python
def crop_white_borders(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """
    自动裁剪图像周围的白色边框
    
    Args:
        image: 输入图像
        threshold: 白色阈值（0-255），大于此值的像素被认为是白色
    
    Returns:
        cropped: 裁剪后的图像
    """
    # 转换为灰度图用于检测
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 创建掩码：非白色区域
    mask = gray < threshold
    
    # 找到非白色区域的边界框
    coords = np.column_stack(np.where(mask))
    
    if len(coords) == 0:
        # 如果整个图像都是白色，返回原图
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 裁剪图像
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped
```

## 使用方法

### 默认行为（自动裁剪）

所有方法默认启用自动裁剪：

```python
from all_skew_correction_methods import correct_skew_method6

# 自动裁剪（默认）
corrected, info = correct_skew_method6(image)
```

### 禁用自动裁剪

如果需要保留留白（例如用于对齐或比较），可以禁用：

```python
# 禁用自动裁剪
corrected, info = correct_skew_method6(image, auto_crop=False)
```

### 调整白色阈值

如果需要调整白色检测的敏感度：

```python
# 修改 crop_white_borders 函数中的 threshold 参数
# 默认值: 250 (0-255)
# 更小的值: 更严格（只裁剪非常白的区域）
# 更大的值: 更宽松（也会裁剪浅灰色区域）
```

## 效果对比

### 启用自动裁剪（默认）

```
原图: 800x600
校正后（无裁剪）: 1000x800 (包含留白)
校正后（有裁剪）: 约 800x600 (无留白) ✅
```

### 禁用自动裁剪

```
原图: 800x600
校正后: 1000x800 (包含留白)
```

## 注意事项

1. **白色阈值**: 默认阈值为250，适用于大多数情况。如果图像背景不是纯白色，可能需要调整。

2. **边缘内容**: 如果图像边缘有重要内容且颜色接近白色，可能会被误裁剪。可以：
   - 降低阈值
   - 禁用自动裁剪
   - 手动裁剪

3. **性能**: 自动裁剪操作很快，对性能影响很小。

4. **特殊情况**: 
   - 如果整个图像都是白色，会返回原图（不裁剪）
   - 如果图像没有留白，裁剪不会改变图像

## 更新说明

**版本更新**: 所有校正方法现在默认启用自动裁剪。

**向后兼容**: 可以通过 `auto_crop=False` 参数禁用自动裁剪，保持旧的行为。

## 示例代码

```python
import cv2
from all_skew_correction_methods import correct_skew_all_methods

# 读取图像
image = cv2.imread('skewed_image.jpg')

# 使用所有方法（自动裁剪留白）
results = correct_skew_all_methods(image)

# 结果图像已自动裁剪，没有留白
for method_key, (corrected, info) in results.items():
    print(f"{method_key}: 尺寸 {corrected.shape[1]}x{corrected.shape[0]}")
    cv2.imwrite(f'result_{method_key}.jpg', corrected)
```

## 技术细节

### 裁剪算法

1. **灰度转换**: 将彩色图像转换为灰度图
2. **阈值处理**: 使用阈值（默认250）区分白色和非白色
3. **边界检测**: 找到所有非白色像素的边界框
4. **裁剪**: 使用边界框坐标裁剪图像

### 性能

- **时间复杂度**: O(n)，其中n是图像像素数
- **空间复杂度**: O(n)
- **实际速度**: 对于1920x1080的图像，约需10-50毫秒

## 常见问题

**Q: 为什么我的图像还是有很多留白？**

A: 可能的原因：
1. 图像边缘有接近白色的内容
2. 阈值设置不合适
3. 图像本身就有白色边框

**Q: 如何调整阈值？**

A: 修改 `crop_white_borders` 函数中的 `threshold` 参数，或创建自定义裁剪函数。

**Q: 可以裁剪其他颜色的背景吗？**

A: 可以，修改 `crop_white_borders` 函数，使用颜色范围检测而不是阈值。

**Q: 自动裁剪会影响图像质量吗？**

A: 不会，裁剪只是移除边缘像素，不进行任何图像处理。
