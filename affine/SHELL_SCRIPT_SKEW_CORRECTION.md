# Shell脚本使用指南 - 倾斜校正

本文档介绍如何使用倾斜校正的Shell脚本。

## 脚本列表

| 脚本 | 功能 | 说明 |
|------|------|------|
| `correct_skew_all_methods.sh` | 使用所有6种方法 | 生成所有方法的校正结果图 |
| `correct_skew_single_method.sh` | 使用单个方法 | 只使用指定的一个方法 |

---

## 脚本1: correct_skew_all_methods.sh

### 功能
使用所有6种倾斜校正方法处理图像，生成每个方法的单独结果图和对比图。

### 用法

#### 方式1: 命令行参数（推荐）

```bash
# 基本用法（自动生成输出目录）
./correct_skew_all_methods.sh image.jpg

# 指定输出目录
./correct_skew_all_methods.sh image.jpg output_folder
```

#### 方式2: 在脚本中设置路径

编辑脚本顶部的配置区域：

```bash
INPUT_IMAGE="/path/to/your/image.jpg"  # 输入图像路径
OUTPUT_DIR="output"                      # 输出目录（可选）
```

然后直接运行：

```bash
./correct_skew_all_methods.sh
```

### 输出结果

脚本会在输出目录中生成：

```
output_directory/
├── 00_original.jpg                    # 原始图像
├── 1_method1_投影轮廓法.jpg          # 方案1结果
├── 2_method2_霍夫直线检测法.jpg      # 方案2结果
├── 3_method3_最小外接矩形法.jpg      # 方案3结果
├── 4_method4_投影变换_旋转组合法.jpg # 方案4结果
├── 5_method5_频域分析法.jpg         # 方案5结果
├── 6_method6_组合方法.jpg            # 方案6结果
└── comparison_all_methods.jpg         # 对比图（所有结果）
```

### 示例

```bash
# 示例1: 处理单张图像
./correct_skew_all_methods.sh /path/to/skewed_document.jpg

# 示例2: 指定输出目录
./correct_skew_all_methods.sh /path/to/skewed_document.jpg /path/to/results

# 示例3: 使用相对路径
./correct_skew_all_methods.sh ./images/test.jpg ./output
```

---

## 脚本2: correct_skew_single_method.sh

### 功能
使用指定的单个方法进行倾斜校正。

### 方法编号

| 编号 | 方法名称 | 特点 |
|------|---------|------|
| 1 | 投影轮廓法 | 最快，适合文档 |
| 2 | 霍夫直线检测法 | 通用场景 |
| 3 | 最小外接矩形法 | 有明显边界 |
| 4 | 投影变换+旋转组合法 | 复杂变形 |
| 5 | 频域分析法 | 周期性纹理 |
| 6 | 组合方法 | **最准确（推荐）** |

### 用法

#### 方式1: 命令行参数（推荐）

```bash
# 使用方案6（组合方法，默认）
./correct_skew_single_method.sh image.jpg

# 使用指定方法
./correct_skew_single_method.sh image.jpg 1

# 指定输出文件
./correct_skew_single_method.sh image.jpg 6 result.jpg

# 指定输出目录
./correct_skew_single_method.sh image.jpg 6 output_folder
```

#### 方式2: 在脚本中设置路径

编辑脚本顶部的配置区域：

```bash
INPUT_IMAGE="/path/to/your/image.jpg"  # 输入图像路径
METHOD_NUM="6"                          # 方法编号 (1-6)
OUTPUT_IMAGE="output"                   # 输出路径（可选）
```

然后直接运行：

```bash
./correct_skew_single_method.sh
```

### 输出结果

生成单个校正后的图像文件。

### 示例

```bash
# 示例1: 使用方案6（组合方法，最准确）
./correct_skew_single_method.sh image.jpg 6

# 示例2: 使用方案1（最快）
./correct_skew_single_method.sh image.jpg 1 result.jpg

# 示例3: 保存到指定目录
./correct_skew_single_method.sh image.jpg 6 ./results
```

---

## 使用场景推荐

### 场景1: 不知道哪种方法最好

**推荐**: 使用 `correct_skew_all_methods.sh`

```bash
./correct_skew_all_methods.sh image.jpg
```

这会生成所有方法的结果，您可以比较后选择最好的。

### 场景2: 需要快速处理

**推荐**: 使用 `correct_skew_single_method.sh` 配合方法1

```bash
./correct_skew_single_method.sh image.jpg 1
```

### 场景3: 需要最高准确度

**推荐**: 使用 `correct_skew_single_method.sh` 配合方法6

```bash
./correct_skew_single_method.sh image.jpg 6
```

### 场景4: 批量处理

可以编写循环脚本：

```bash
#!/bin/bash
for img in *.jpg; do
    ./correct_skew_single_method.sh "$img" 6 "output/$img"
done
```

---

## 输出路径说明

### 自动生成文件名

如果不指定输出路径，脚本会自动生成文件名：

- **所有方法脚本**: `skew_correction_results_<原文件名>_<时间戳>/`
- **单个方法脚本**: `corrected_method<编号>_<原文件名>_<时间戳>.jpg`

### 指定目录

如果指定目录路径（不以文件扩展名结尾），会在该目录中自动生成文件名：

```bash
./correct_skew_single_method.sh image.jpg 6 output_folder
# 结果: output_folder/corrected_method6_image_20240125_123456.jpg
```

### 指定文件

如果指定完整文件路径，会保存到该文件：

```bash
./correct_skew_single_method.sh image.jpg 6 result.jpg
# 结果: result.jpg
```

---

## 常见问题

### Q: 脚本没有执行权限？

A: 添加执行权限：

```bash
chmod +x correct_skew_all_methods.sh
chmod +x correct_skew_single_method.sh
```

### Q: 提示找不到Python脚本？

A: 确保以下文件在同一目录下：
- `all_skew_correction_methods.py`
- `test_all_methods.py` (仅所有方法脚本需要)

### Q: 提示找不到opencv-python？

A: 安装依赖：

```bash
pip install opencv-python numpy
```

### Q: 处理速度慢？

A: 
1. 使用方法1（最快）
2. 缩小输入图像尺寸
3. 只使用单个方法而不是所有方法

### Q: 检测不准确？

A: 
1. 使用方法6（组合方法，最准确）
2. 预处理图像（增强对比度、去噪）
3. 使用所有方法脚本，比较结果后选择最好的

---

## 完整示例

### 示例1: 处理文档图像

```bash
# 使用所有方法，比较结果
./correct_skew_all_methods.sh document.jpg results

# 查看结果
ls -lh results/
```

### 示例2: 快速校正

```bash
# 使用最快的方法
./correct_skew_single_method.sh document.jpg 1 corrected.jpg
```

### 示例3: 高精度校正

```bash
# 使用最准确的方法
./correct_skew_single_method.sh document.jpg 6 high_quality_result.jpg
```

### 示例4: 批量处理

```bash
#!/bin/bash
# 批量处理脚本

INPUT_DIR="./input_images"
OUTPUT_DIR="./corrected_images"
METHOD=6  # 使用方法6

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.jpg; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        echo "处理: $filename"
        ./correct_skew_single_method.sh "$img" "$METHOD" "$OUTPUT_DIR/$filename"
    fi
done

echo "批量处理完成！结果在: $OUTPUT_DIR"
```

---

## 与Python代码的对应关系

| Shell脚本 | Python代码 |
|-----------|-----------|
| `correct_skew_all_methods.sh` | `test_all_methods.py` |
| `correct_skew_single_method.sh` | `all_skew_correction_methods.py` 中的单个方法函数 |

---

## 注意事项

1. **路径**: 支持绝对路径和相对路径
2. **扩展名**: 输出图像会自动添加 `.jpg` 扩展名（如果没有指定）
3. **目录**: 如果输出目录不存在，会自动创建
4. **错误处理**: 脚本会检查输入文件是否存在，Python环境是否正确

---

## 更多信息

- **详细方法说明**: 查看 `ALL_METHODS_USAGE.md`
- **方案原理分析**: 查看 `ADAPTIVE_SKEW_CORRECTION_PLANS.md`
- **Python API**: 查看 `all_skew_correction_methods.py`
