# Shell脚本快速参考

## 🚀 最快使用方式

### 使用所有方法（推荐首次使用）

```bash
./correct_skew_all_methods.sh image.jpg
```

### 使用单个方法（最快）

```bash
./correct_skew_single_method.sh image.jpg 6
```

---

## 📋 命令格式

### 所有方法脚本

```bash
./correct_skew_all_methods.sh [输入图像] [输出目录]
```

**示例**:
```bash
./correct_skew_all_methods.sh image.jpg
./correct_skew_all_methods.sh image.jpg output_folder
```

### 单个方法脚本

```bash
./correct_skew_single_method.sh [输入图像] [方法编号] [输出路径]
```

**方法编号**:
- `1` - 投影轮廓法（最快）
- `2` - 霍夫直线检测法
- `3` - 最小外接矩形法
- `4` - 投影变换+旋转组合法
- `5` - 频域分析法
- `6` - 组合方法（**最准确，推荐**）

**示例**:
```bash
./correct_skew_single_method.sh image.jpg 6
./correct_skew_single_method.sh image.jpg 1 result.jpg
./correct_skew_single_method.sh image.jpg 6 output_folder
```

---

## 🎯 使用场景

| 场景 | 推荐命令 |
|------|---------|
| 不知道哪种方法好 | `./correct_skew_all_methods.sh image.jpg` |
| 需要最快速度 | `./correct_skew_single_method.sh image.jpg 1` |
| 需要最高准确度 | `./correct_skew_single_method.sh image.jpg 6` |
| 批量处理 | 编写循环脚本（见文档） |

---

## 📁 输出文件

### 所有方法脚本输出

```
output_directory/
├── 00_original.jpg
├── 1_method1_*.jpg
├── 2_method2_*.jpg
├── 3_method3_*.jpg
├── 4_method4_*.jpg
├── 5_method5_*.jpg
├── 6_method6_*.jpg
└── comparison_all_methods.jpg
```

### 单个方法脚本输出

单个校正后的图像文件。

---

## ⚙️ 配置方式

### 方式1: 命令行参数（推荐）

```bash
./correct_skew_all_methods.sh image.jpg output_folder
```

### 方式2: 编辑脚本

编辑脚本顶部的配置区域：

```bash
INPUT_IMAGE="/path/to/image.jpg"
OUTPUT_DIR="output"
```

然后运行：

```bash
./correct_skew_all_methods.sh
```

---

## 🔧 常见问题

**没有执行权限?**
```bash
chmod +x correct_skew_all_methods.sh
chmod +x correct_skew_single_method.sh
```

**找不到Python模块?**
```bash
pip install opencv-python numpy
```

---

## 📚 详细文档

- **完整使用指南**: `SHELL_SCRIPT_SKEW_CORRECTION.md`
- **方法说明**: `ALL_METHODS_USAGE.md`
- **方案分析**: `ADAPTIVE_SKEW_CORRECTION_PLANS.md`
