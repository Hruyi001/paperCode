# 快速参考卡片

## Shell脚本快速使用

### 最简单的方式
```bash
./align_simple.sh source.jpg reference.jpg
```

### 完整功能
```bash
./align_images.sh source.jpg reference.jpg -d SIFT --show-matches -o results
```

## Python API快速使用

```python
from solution1_feature_matching import align_image_with_features
import cv2

source = cv2.imread('source.jpg')
reference = cv2.imread('reference.jpg')

aligned, transform = align_image_with_features(source, reference, detector_type='ORB')
cv2.imwrite('aligned.jpg', aligned)
```

## 特征检测器选择

- **ORB** (默认): 快速，免费，适合大多数场景
- **SIFT**: 最准确，需要 opencv-contrib-python
- **AKAZE**: 平衡速度和准确性

## 常见问题快速解决

| 问题 | 解决方案 |
|------|---------|
| 对齐失败 | 尝试 `-d SIFT` 或 `-d AKAZE` |
| 速度慢 | 使用 `-d ORB` |
| 精度不够 | 使用 `-d SIFT` |
| 找不到python3 | 安装Python 3 |
| 找不到cv2 | `pip install opencv-python` |
