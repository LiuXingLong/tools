# DeepSeek Web API 工具集

基于 DeepSeek 浏览器端 Web API（非官方）的工具集合，绕过官方 API 计费，通过 PoW 挑战认证。

## 目录结构

```
deepseek/
  __init__.py
  .env                     # 环境变量（不提交）
  .env.example             # 环境变量模板
  deepseek_chat_cli.py     # CLI 工具
  deepseek_responses_api_sdk.py   # Python SDK 封装
  deepseek_responses_api_server.py # FastAPI HTTP 服务
  deepseek.log             # 日志文件（不提交）
sha3_wasm_bg.wasm          # PoW 解算依赖（项目根目录）
```

## 环境变量（deepseek/.env）

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | **必需**。浏览器中获取的 `userToken` | `8vHJX8O4m...` |
| `DEEPSEEK_WASM_PATH` | PoW 解算用的 WASM 文件路径，默认 `sha3_wasm_bg.wasm` | `sha3_wasm_bg.wasm` |

## 脚本说明

所有命令需在项目根目录执行：

### `deepseek_chat_cli.py`

```bash
# 基础对话
python3 deepseek/deepseek_chat_cli.py "你好"

# 非流式
python3 deepseek/deepseek_chat_cli.py "你好" --no-stream

# 深度思考（R1 模型）
python3 deepseek/deepseek_chat_cli.py "9.9 和 9.11 谁大" --thinking

# 联网搜索
python3 deepseek/deepseek_chat_cli.py "今天的新闻" --search

# 工具调用
python3 deepseek/deepseek_chat_cli.py "北京天气怎么样" --tool
```

### `deepseek_responses_api_sdk.py`

```python
from deepseek.deepseek_responses_api_sdk import DeepSeekResponses

client = DeepSeekResponses()
resp = client.create(model="deepseek-chat", input="你好")
print(resp["output"])
```

### `deepseek_responses_api_server.py`

```bash
# 启动服务
python3 deepseek/deepseek_responses_api_server.py
# 默认 http://0.0.0.0:8888

# 非流式
curl http://localhost:8888/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","input":"你好","stream":false}'

# 流式
curl -N http://localhost:8888/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","input":"你好","stream":true}'

# 健康检查
curl http://localhost:8888/health
```

API key 传递方式（按优先级）：
1. HTTP `Authorization: Bearer <token>` 头
2. `deepseek/.env` 文件中 `DEEPSEEK_API_KEY`

#### 日志系统

- JSON 格式日志，双输出：
  - **控制台**：INFO 及以上（含 `REQ/RESP/DSREQ/DSRES/ERR` 类型标记）
  - **文件** `deepseek/deepseek.log`：DEBUG 及以上（含详细步骤）

#### SSE 流式事件生命周期

```text
response.created → response.in_progress → response.output_item.added
→ response.content_part.added → response.output_text.delta (xN)
→ response.output_text.done → response.content_part.done
→ response.output_item.done → response.completed
```

### 其他脚本

- `edit_image_text.py` — OCR 驱动的图片文字编辑
- `replace_pdf.py` — PDF 文字搜索替换

## 注意事项

- token 来自 DeepSeek 浏览器的 `userToken`
- PoW 需要每次请求重新解算，依赖 `sha3_wasm_bg.wasm`
- 工具调用通过 prompt 注入实现，非原生 `tools` 参数
- 专家模式（`model_type: "expert"`）当前不可用
