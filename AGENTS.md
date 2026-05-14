# AGENTS.md — Tools Codebase Guide

## 项目概述

基于 Python 的图片/PDF 文字编辑工具集，包括：
- `edit_image_text.py` — OCR 驱动的图片文字替换/删除/添加
- `replace_pdf.py` — PDF 文字搜索替换

无测试文件、无构建系统、无包配置文件，为松散的工具集合。

## 构建/运行/测试命令

```bash
# 运行图片文字替换
python3 edit_image_text.py replace input.jpg --old "旧文字" --new "新文字"

# 运行 PDF 文字替换
python3 replace_pdf.py input.pdf output.pdf -r 旧文本 新文本

# 检测图片中的文字
python3 edit_image_text.py detect input.jpg

# 添加文字到图片
python3 edit_image_text.py add input.jpg -x 50 -y 50 --text "Hello" --size 24 --bold 1
```

当前无单元测试、无 lint、无 typecheck 配置。如需添加：

```bash
# 安装开发依赖
pip install pytest mypy ruff

# 运行所有测试
python3 -m pytest tests/

# 运行单个测试
python3 -m pytest tests/test_edit_image_text.py -v

# 类型检查
mypy edit_image_text.py --strict

# Lint
ruff check .
```

## 代码风格指南

### 导入顺序

```python
# 标准库
import argparse
import os
import sys
from pathlib import Path

# 第三方库（空行分隔）
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
```

### 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 函数/方法 | `snake_case` | `find_text_boxes()`, `replace_text_ocr()` |
| 私有函数 | 下划线前缀 | `_sample_bg_color()`, `_find_system_font()` |
| 变量 | `snake_case` | `text_mask`, `font_path` |
| 常量 | `UPPER_SNAKE_CASE` | `_HAO_SIZE_MAP` |
| 模块 | `snake_case` | `edit_image_text.py` |

### 类型注解

全员使用类型注解（Python 3.11）：

```python
def find_text_boxes(
    image_path: str,
    target_text: str = "",
    lang: str = "chi_sim+eng",
    exact_match: bool = True,
) -> list[dict]:
    ...
```

### 文档字符串

采用 Google style，所有公开函数必须包含：

```python
def replace_text_ocr(...) -> str:
    """一句话功能描述

    Args:
        image_path: 输入图片路径
        old_text: 要替换的原文字

    Returns:
        输出图片路径
    """
```

### 错误处理

- 可选依赖使用 try/except 延迟导入降级
- 文件缺失使用 `FileNotFoundError` + 中文消息
- 参数错误使用 `print` + `sys.exit(1)`（CLI 上下文）
- 静默处理非关键异常（`except Exception: pass`）
- 外部资源不可用时返回空字符串/None 而非崩溃

### 代码组织

- 顶层模块内含多个函数的松散脚本，无类封装
- 私有函数前缀 `_`（约 10 个），公开函数约 6 个
- CLI 入口统一通过 `main()` + `argparse`，子命令模式
- `replace_pdf.py` 为过程式脚本（无 `if __name__` 守卫）
- 中文变量名/注释适合本项目上下文（处理中文文档）

### 提交信息规范

```txt
feat: 添加 xxx 功能
fix: 修复 xxx 问题
refactor: 重构 xxx
docs: 更新 xxx 文档
```

### 处理 PDF 的特殊约定

```python
# 宋体字体提取（macOS 硬编码路径）
ttc = TTCollection("/System/Library/Fonts/Supplemental/Songti.ttc")
songti_buf = io.BytesIO()
ttc[6].save(songti_buf)
```

### 处理图片的特殊约定

```python
# 中文后备字体搜索路径（按优先级）
_FONT_CANDIDATES = [
    "~/Library/Fonts/NotoSansSC-Variable.ttf",
    "~/Library/Fonts/SimHei.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    ...
]
```

### 依赖项

```
opencv-python, numpy, pillow, pytesseract  # 图片工具
pymupdf, fonttools                          # PDF 工具
```
