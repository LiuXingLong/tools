"""
DeepSeek Responses API HTTP 服务
OpenAI /v1/responses 兼容接口，支持流式和非流式

启动：
    python3 deepseek_responses_api_server.py
    # 默认 http://0.0.0.0:8888

测试用请求示例（替换 TOKEN 为你的 userToken）：

    # 1. 基础非流式
    curl http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"model":"deepseek-chat","input":"你好","stream":false}'

    # 2. 流式
    curl -N http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"model":"deepseek-chat","input":"你好","stream":true}'

    # 3. 联网搜索（隐式，始终开启）
    curl http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"model":"deepseek-chat","input":"今天的日期","stream":false}'

    # 4. 深度思考（R1 模型）
    curl -N http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"model":"deepseek-reasoner","input":"9.9和9.11谁大","stream":true}'

    # 5. 工具调用
    curl http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "model":"deepseek-chat",
        "input":"计算 12345 * 67890",
        "stream":false,
        "tool_choice":"required",
        "tools":[{
          "type":"function",
          "function":{
            "name":"calculator",
            "description":"计算数学表达式",
            "parameters":{
              "type":"object",
              "properties":{
                "expr":{"type":"string","description":"数学表达式"}
              },
              "required":["expr"]
            }
          }
        }]
      }'

    # 6. 流式 + 工具
    curl -N http://localhost:8888/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "model":"deepseek-chat",
        "input":"北京和上海今天的天气对比",
        "stream":true,
        "tools":[{
          "type":"function",
          "function":{
            "name":"get_weather",
            "description":"获取城市天气",
            "parameters":{
              "type":"object",
              "properties":{
                "city":{"type":"string"}
              },
              "required":["city"]
            }
          }
        }]
      }'

    # 7. 健康检查
    curl http://localhost:8888/health
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from deepseek_responses_api_sdk import DeepSeekResponses, DeepSeekAPIError

load_dotenv(dotenv_path=_HERE / ".env")


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            obj.update(record.msg)
        else:
            obj["msg"] = record.getMessage()
        return json.dumps(obj, ensure_ascii=False)


_formatter = _JSONFormatter()

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_formatter)

_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.log")
_file = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_formatter)

logger = logging.getLogger("deepseek")
logger.setLevel(logging.DEBUG)
logger.addHandler(_console)
logger.addHandler(_file)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

app = FastAPI(title="DeepSeek Responses API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client(api_key: str) -> DeepSeekResponses:
    return DeepSeekResponses(api_key=api_key)


def _non_stream_response(api_key: str, body: dict) -> dict:
    client = _get_client(api_key)
    result = client.create(**body)
    return result


async def _stream_response(api_key: str, body: dict):
    import asyncio

    body.pop("stream", None)
    try:
        client = _get_client(api_key)
        events = await asyncio.to_thread(client.create, **body, stream=True)
        for ev in events:
            ev_type = ev.get("type", "")
            data = json.dumps(ev, ensure_ascii=False)
            yield f"event: {ev_type}\ndata: {data}\n\n"
    except DeepSeekAPIError as e:
        logger.warning({"type": "ERR", "error": e.message, "status": e.status})
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': e.message, 'status': e.status}}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"流式处理异常: {type(e).__name__}: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': f'{type(e).__name__}: {e}', 'status': 500}}, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    client_ip = request.client.host if request.client else "unknown"
    logger.error("[%s] <<< 全局未捕获异常: %s %s", client_ip, type(exc).__name__, exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": f"服务器内部错误: {type(exc).__name__}: {exc}"}},
    )


@app.post("/responses")
@app.post("/v1/responses")
async def responses_endpoint(request: Request):
    client_ip = request.client.host if request.client else "unknown"

    auth = request.headers.get("Authorization", "")
    api_key = auth.removeprefix("Bearer ").strip() or os.environ.get(
        "DEEPSEEK_API_KEY", ""
    )
    if not api_key:
        logger.warning("[%s] <<< 401 无 API key", client_ip)
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Missing API key. Provide via Authorization: Bearer <token>"
                }
            },
        )

    try:
        body = await request.json()
    except Exception as e:
        logger.error("[%s] <<< 400 JSON 解析失败: %s", client_ip, e)
        return JSONResponse(
            status_code=400, content={"error": {"message": f"请求体不是有效 JSON: {e}"}}
        )

    logger.info({"type": "REQ", "client_ip": client_ip, "body": body})

    stream = body.get("stream", False)
    if stream:
        return StreamingResponse(
            _stream_response(api_key, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await run_in_thread_pool(_non_stream_response, api_key, body)
        logger.info({"type": "RESP", "client_ip": client_ip, "resp": result})
        return JSONResponse(content=result)
    except DeepSeekAPIError as e:
        logger.warning(
            {
                "type": "ERR",
                "client_ip": client_ip,
                "status": e.status,
                "error": e.message,
            }
        )
        return JSONResponse(
            status_code=e.status,
            content={"error": {"message": e.message, "status": e.status}},
        )
    except Exception as e:
        logger.error(
            "[%s] <<< 非流式未知异常: %s %s",
            client_ip,
            type(e).__name__,
            e,
        )
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"服务器内部错误: {type(e).__name__}: {e}"}},
        )


import asyncio
from concurrent.futures import ThreadPoolExecutor

_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread_pool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, func, *args)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    print(f"启动 DeepSeek Responses API 服务 http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)
