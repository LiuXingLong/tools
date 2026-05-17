# AGENTS.md — Tools Codebase Guide

## 项目概述

基于 Python 的图片/PDF 文字编辑 + DeepSeek Web API 工具集。

**图片/PDF 工具：**
- `edit_image_text.py` — OCR 驱动的图片文字替换/删除/添加
- `replace_pdf.py` — PDF 文字搜索替换

**DeepSeek 工具（`deepseek/` 目录）：**
- `deepseek/deepseek_responses_api_sdk.py` — Python SDK 封装
- `deepseek/deepseek_responses_api_server.py` — FastAPI HTTP 服务
- `deepseek/deepseek_chat_cli.py` — CLI 工具

## 运行命令

### DeepSeek 工具

```bash
# 启动 HTTP 服务
python3 deepseek/deepseek_responses_api_server.py

# CLI 对话
python3 deepseek/deepseek_chat_cli.py "你好"

# 直接在 Python 中使用 SDK
python3 -c "
from deepseek.deepseek_responses_api_sdk import DeepSeekResponses
client = DeepSeekResponses()
resp = client.create(model='deepseek-chat', input='你好')
print(resp)
"
```

### 图片/PDF 工具

```bash
python3 edit_image_text.py replace input.jpg --old "旧文字" --new "新文字"
python3 replace_pdf.py input.pdf output.pdf -r 旧文本 新文本
```

## 项目结构

```
tools/
├── deepseek/
│   ├── __init__.py
│   ├── .env                    # 环境变量（不提交）
│   ├── .env.example
│   ├── deepseek_chat_cli.py
│   ├── deepseek_responses_api_sdk.py
│   ├── deepseek_responses_api_server.py
│   └── deepseek.log            # 日志文件（不提交）
├── sha3_wasm_bg.wasm           # PoW 解算依赖
├── edit_image_text.py
├── replace_pdf.py
├── .gitignore
├── AGENTS.md
└── README.md
```

## DeepSeek 代码规范

所有 DeepSeek 核心逻辑在 `deepseek/deepseek_responses_api_sdk.py` 中维护，
`deepseek/deepseek_responses_api_server.py` 仅负责 HTTP 路由和 SSE 格式转换。

```bash
# 依赖
pip install opendeep fastapi uvicorn python-dotenv
```

### 跨文件引用

```python
# SDK 引用
from deepseek_responses_api_sdk import DeepSeekResponses  # 同目录

# 通过项目 root 引用
from deepseek.deepseek_responses_api_sdk import DeepSeekResponses
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

### 提交信息规范

```txt
feat: 添加 xxx 功能
fix: 修复 xxx 问题
refactor: 重构 xxx
docs: 更新 xxx 文档
```

### 依赖项

```
opencv-python, numpy, pillow, pytesseract  # 图片工具
pymupdf, fonttools                          # PDF 工具
opendeep, fastapi, uvicorn, python-dotenv   # DeepSeek 工具
```
