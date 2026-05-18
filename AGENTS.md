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
# 运行测试（SDK + CLI + HTTP 服务）
python3 deepseek/test_deepseek.py

# 只测 SDK + CLI（不需要 HTTP 服务）
python3 deepseek/test_deepseek.py --no-server

# 启动 HTTP 服务（端口通过 PORT 环境变量配置，默认 8888）
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
│   ├── test_deepseek.py        # 测试脚本
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

### Codex CLI 代理原则

DeepSeek Responses 代理只做协议适配：

- 只支持 Responses 模式 `/v1/responses`，不新增其他兼容入口，除非用户明确要求。
- Codex CLI 请求中带了哪些 `tools`，就只转换这些工具给 DeepSeek；不得新增、替换、过滤或按语义启停工具。
- 不要在代理层根据天气、实时信息、搜索等语义修改工具调用策略。
- 不要注入与原始请求无关的控制性提示，例如强制不再调用工具、强制改用某种搜索方式。
- `function_call`、`function_call_output` 只做 OpenAI Responses 与 DeepSeek Web API 之间的结构转换；是否继续调用工具由 DeepSeek 的真实返回决定。
- 允许保留 DeepSeek Web API 必需的协议字段（例如 `search_enabled`、`thinking_enabled`），但不得用这些字段改变 Codex CLI 原有工具语义。

### DeepSeek 工具调用兼容问题

Codex CLI 到代理这一段的 OpenAI Responses 工具协议是固定的；不稳定点在代理到 DeepSeek Web API 这一段：DeepSeek Web API 没有原生 Responses tools schema，工具调用目前依赖 prompt 约束模型输出文本，再由代理解析文本并转换为 Responses `function_call`。

因此调试工具调用问题时要区分两段协议：

- Codex CLI 请求中的 `tools`、`function_call`、`function_call_output` 是固定结构，只能按字段转换，不能按语义改写。
- DeepSeek 返回的工具调用可能是模型生成文本，常见变体包括纯 JSON、`<tool_call>...</tool_call>`、`工具：name({...})`、前置说明加裸 JSON。
- 如果 DeepSeek 文本中包含工具调用，代理应提取并转换成 Responses `function_call`，不要把原始工具 JSON 或标签作为 `output_text` 透给 Codex。
- 如果解析失败，优先查看 `deepseek/deepseek.log` 中的 `DSREQ`、`DSRES`、`RESP`，确认是 DeepSeek 未调用工具、返回了新格式，还是代理转换失败。
- 修复这类问题只应扩展文本到结构的解析兼容性，不应新增工具、不应过滤 Codex 工具、不应强制 DeepSeek 停止或继续调用工具。

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
