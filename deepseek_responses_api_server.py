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
    curl http://localhost:8888/v1/responses \
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
import re
import time
import uuid

from dotenv import load_dotenv

import opendeep as genai
from opendeep import pow as deepseek_pow
from opendeep.config import config as ds_config

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

load_dotenv()


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

# 控制台输出（INFO 及以上）
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_formatter)

# 文件输出（DEBUG 及以上，包含详细步骤）
_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.log")
_file = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_formatter)

logger = logging.getLogger("deepseek")
logger.setLevel(logging.DEBUG)
logger.addHandler(_console)
logger.addHandler(_file)

# 抑制 uvicorn 访问日志
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

TOOL_SYSTEM_PROMPT = """
你有以下工具可用，需要获取实时信息时请严格按 JSON 格式返回工具调用：

{tools_desc}

需要调用工具时，只返回以下格式（不要包含其他文字）：
{{"tool": "函数名", "args": {{"参数名": "参数值"}}}}

不需要工具时正常回答即可。
"""

app = FastAPI(title="DeepSeek Responses API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 wasm 路径（从 .env 读取，默认 sha3_wasm_bg.wasm）
deepseek_pow.WASM_PATH = os.environ.get("DEEPSEEK_WASM_PATH", "sha3_wasm_bg.wasm")


def _build_tool_desc(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        func = t.get("function", t)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        args_str = ", ".join(
            f"{k}: {v.get('type', 'str')}{' (必填)' if k in required else ''}"
            for k, v in props.items()
        )
        lines.append(f"- {name}({args_str}): {desc}")
    return "\n".join(lines)


def _gen_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _parse_input(inp) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        texts = []
        for item in inp:
            if isinstance(item, dict):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "input_text":
                            texts.append(c["text"])
                        elif isinstance(c, str):
                            texts.append(c)
                elif "content" in item:
                    content = item["content"]
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                texts.append(c["text"])
                            elif isinstance(c, str):
                                texts.append(c)
                    elif isinstance(content, str):
                        texts.append(content)
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(inp)


def _merge_events(events: list[dict]) -> str:
    text = ""
    target = "content"
    for ev in events:
        if "v" not in ev:
            continue
        p = ev.get("p", "")
        v = ev["v"]
        if p:
            target = p.removeprefix("response/")
        if isinstance(v, str) and target in ("content", "thinking_content"):
            text += v
    return text


def _detect_tool_call(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return {
                "name": obj["tool"],
                "arguments": json.dumps(obj.get("args", {}), ensure_ascii=False),
            }
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _build_response(
    model_name: str, full_text: str, tool_call: dict | None, has_tools: bool
) -> dict:
    created = int(time.time())
    resp_id = _gen_id("resp")
    output = []

    if tool_call:
        output.append(
            {
                "type": "function_call",
                "id": _gen_id("fc"),
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
                "status": "completed",
                "call_id": uuid.uuid4().hex[:12],
            }
        )

    content_parts = []
    if full_text:
        content_parts.append(
            {
                "type": "output_text",
                "text": full_text,
                "annotations": [],
            }
        )

    if content_parts:
        output.append(
            {
                "type": "message",
                "id": _gen_id("msg"),
                "role": "assistant",
                "content": content_parts,
            }
        )

    return {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model_name,
        "output": output,
        "parallel_tool_calls": has_tools,
        "temperature": 1.0,
        "tool_choice": "auto" if has_tools else None,
        "tools": None,
        "top_p": 1.0,
        "truncation": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "user": None,
    }


class DeepSeekAPIError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def _call_deepseek(
    api_key: str,
    model_name: str,
    prompt: str,
    tools: list,
    tool_choice: str,
    client_ip: str = "unknown",
) -> tuple[list[dict], str, bool]:
    """调用 DeepSeek Web API，返回 (events, model_name, has_tools)"""
    logger.debug(
        "[%s] _call_deepseek 开始 | model=%s | prompt_len=%d",
        client_ip,
        model_name,
        len(prompt),
    )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("deepseek-chat")
    session = model.session
    headers = model._get_headers()
    thinking_enabled = model_name == "deepseek-reasoner"

    # PoW
    try:
        pow_solver = deepseek_pow.DeepSeekPOW()
        logger.debug("[%s] PoW 请求开始", client_ip)
        pow_resp = session.post(
            f"{ds_config.base_url}/chat/create_pow_challenge",
            headers=headers,
            json={"target_path": "/api/v0/chat/completion"},
        )
        logger.debug("[%s] PoW 响应 status=%s", client_ip, pow_resp.status_code)
        logger.info(
            {
                "type": "DSREQ",
                "client_ip": client_ip,
                "target": "PoW",
                "status": pow_resp.status_code,
            }
        )
        if pow_resp.ok:
            cd = pow_resp.json()
            if cd and isinstance(cd, dict):
                cd = cd.get("data", {}).get("biz_data", {}).get("challenge")
                if cd:
                    logger.debug("[%s] PoW 解算开始", client_ip)
                    headers["x-ds-pow-response"] = pow_solver.solve_challenge(cd)
                    logger.debug("[%s] PoW 解算完成", client_ip)
    except Exception as e:
        logger.error("[%s] PoW 异常: %s", client_ip, e)
        raise DeepSeekAPIError(401, f"PoW 认证失败（token 可能无效）: {e}")

    chat_session_id = model._create_session()
    has_tools = bool(tools) and tool_choice != "none"

    final_prompt = prompt
    if has_tools:
        tools_desc = _build_tool_desc(tools)
        final_prompt = (
            TOOL_SYSTEM_PROMPT.format(tools_desc=tools_desc) + "\n\n" + prompt
        )

    payload = {
        "chat_session_id": chat_session_id,
        "prompt": final_prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": True,
        "model_type": "default",
    }

    logger.info(
        {
            "type": "DSREQ",
            "client_ip": client_ip,
            "target": "chat/completion",
            "payload": payload,
        }
    )

    try:
        logger.debug(
            "[%s] chat/completion 请求开始 | thinking=%s", client_ip, thinking_enabled
        )
        resp = session.post(
            f"{ds_config.base_url}/chat/completion",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )
        logger.debug("[%s] chat/completion 响应 status=%s", client_ip, resp.status_code)
        resp.raise_for_status()
        logger.info(
            {
                "type": "DSRES",
                "client_ip": client_ip,
                "target": "chat/completion",
                "status": resp.status_code,
                "ok": True,
            }
        )
    except Exception as e:
        status = getattr(resp, "status_code", 500) if "resp" in dir() else 500
        detail = str(e)
        if "resp" in dir():
            try:
                detail = resp.json()
            except Exception:
                detail = getattr(resp, "text", str(e))[:500]
        logger.error(
            "[%s] chat/completion 异常: status=%s err=%s",
            client_ip,
            status,
            str(detail)[:300],
        )
        logger.warning(
            {
                "type": "DSRES",
                "client_ip": client_ip,
                "target": "chat/completion",
                "status": status,
                "error": detail,
            }
        )
        raise DeepSeekAPIError(status, f"DeepSeek API 调用失败: {detail}") from e

    logger.debug("[%s] 开始读取 SSE 事件流", client_ip)
    events = []
    line_count = 0
    for line in resp.iter_lines():
        if line:
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                raw = decoded[6:].strip()
                if raw:
                    events.append(json.loads(raw))
                    line_count += 1
    logger.debug("[%s] SSE 事件读取完成: %d 条", client_ip, line_count)

    logger.info(
        {
            "type": "DSRES",
            "client_ip": client_ip,
            "target": "chat/completion",
            "events_count": len(events),
            "model": model_name,
            "events_preview": events[:3] if events else [],
        }
    )
    return events, model_name, has_tools


# ─── 非流式 ─────────────────────────────────────────────


def _non_stream_response(api_key: str, body: dict, client_ip: str = "unknown") -> dict:
    model_name = body.get("model", "deepseek-chat")
    prompt = _parse_input(body.get("input", ""))
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")

    events, model_name, has_tools = _call_deepseek(
        api_key, model_name, prompt, tools, tool_choice, client_ip
    )

    full_text = _merge_events(events)
    full_text = re.sub(r"\s*\[citation:\d+\]", "", full_text)
    tool_call = _detect_tool_call(full_text)

    return _build_response(model_name, full_text, tool_call, has_tools)


# ─── 流式 ───────────────────────────────────────────────


async def _stream_response(api_key: str, body: dict, client_ip: str = "unknown"):
    model_name = body.get("model", "deepseek-chat")
    prompt = _parse_input(body.get("input", ""))
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")
    logger.debug(
        "[%s] _stream_response 开始 | model=%s | prompt=%s",
        client_ip,
        model_name,
        prompt[:50],
    )

    try:
        logger.debug("[%s] asyncio.to_thread(_call_deepseek) 开始", client_ip)
        events, model_name, has_tools = await asyncio.to_thread(
            _call_deepseek, api_key, model_name, prompt, tools, tool_choice, client_ip
        )
        logger.debug("[%s] asyncio.to_thread 完成 | events=%d", client_ip, len(events))
    except DeepSeekAPIError as e:
        logger.error("[%s] DeepSeekAPIError 捕获: %s", client_ip, e.message)
        logger.warning(
            {
                "type": "ERR",
                "client_ip": client_ip,
                "model": model_name,
                "error": e.message,
            }
        )
        yield f"event: error\ndata: {json.dumps({'error': {'message': e.message, 'status': e.status}, 'type': 'error'}, ensure_ascii=False)}\n\n"
        return
    except Exception as e:
        logger.error(
            "[%s] _call_deepseek 未知异常: %s %s", client_ip, type(e).__name__, e
        )
        yield f"event: error\ndata: {json.dumps({'error': {'message': f'{type(e).__name__}: {e}', 'status': 500}, 'type': 'error'}, ensure_ascii=False)}\n\n"
        return

    try:
        full_text = _merge_events(events)
        full_text = re.sub(r"\s*\[citation:\d+\]", "", full_text)
        tool_call = _detect_tool_call(full_text)
        final = _build_response(model_name, full_text, tool_call, has_tools)
        msg_id = (
            final.get("output", [{}])[0].get("id", "msg_unknown")
            if final.get("output")
            else "msg_unknown"
        )
        seq = 0

        # 1. response.created
        logger.debug("[%s] 发送 response.created", client_ip)
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': final, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 2. response.in_progress
        yield f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': final, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 3. response.output_item.added
        yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item_id': msg_id, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 4. response.content_part.added
        yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'content_index': 0, 'item_id': msg_id, 'output_index': 0, 'part_id': 'part_0', 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 5. response.output_text.delta
        logger.debug("[%s] 发送 delta 事件 | 共 %d 条", client_ip, len(events))
        delta_count = 0
        for ev in events:
            if "v" not in ev:
                continue
            p = ev.get("p", "")
            v = ev["v"]
            if not isinstance(v, str):
                continue
            target = "content"
            if p:
                target = p.removeprefix("response/")
            if target in ("content", "thinking_content"):
                yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'delta': v, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
                seq += 1
                delta_count += 1
        logger.debug("[%s] delta 事件发送完成 | 共 %d 条", client_ip, delta_count)

        # 6. response.output_text.done
        yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'text': full_text, 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 7. response.content_part.done
        yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'content_index': 0, 'item_id': msg_id, 'output_index': 0, 'part_id': 'part_0', 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 8. response.output_item.done
        yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item_id': msg_id, 'sequence_number': seq}, ensure_ascii=False)}\n\n"
        seq += 1

        # 9. response.completed
        logger.debug("[%s] 发送 response.completed", client_ip)
        yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': final, 'sequence_number': seq}, ensure_ascii=False)}\n\n"

        logger.info({"type": "RESP", "client_ip": client_ip, "resp": final})
        logger.debug("[%s] _stream_response 正常完成", client_ip)
    except Exception as e:
        logger.error("[%s] delta/done 发送异常: %s %s", client_ip, type(e).__name__, e)
        logger.warning(
            {
                "type": "ERR",
                "client_ip": client_ip,
                "model": model_name,
                "error": f"流式处理异常: {e}",
            }
        )
        yield f"event: error\ndata: {json.dumps({'error': {'message': str(e), 'status': 500}, 'type': 'error'}, ensure_ascii=False)}\n\n"


# ─── 路由 ───────────────────────────────────────────────


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
    """OpenAI /v1/responses 兼容接口"""
    client_ip = request.client.host if request.client else "unknown"
    logger.debug(
        "[%s] >>> 收到请求 | method=%s | path=%s",
        client_ip,
        request.method,
        request.url.path,
    )

    # 获取 API key：优先 Authorization 头，其次 .env 文件
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

    model = body.get("model", "deepseek-chat")
    stream = body.get("stream", False)
    logger.debug("[%s] REQ body | model=%s | stream=%s", client_ip, model, stream)
    logger.info({"type": "REQ", "client_ip": client_ip, "body": body})

    if stream:
        logger.debug("[%s] >>> 返回 StreamingResponse", client_ip)
        return StreamingResponse(
            _stream_response(api_key, body, client_ip),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await run_in_thread_pool(
            _non_stream_response, api_key, body, client_ip
        )
        logger.debug("[%s] <<< 非流式响应 | status=%s", client_ip, result.get("status"))
        logger.info({"type": "RESP", "client_ip": client_ip, "resp": result})
        return JSONResponse(content=result)
    except DeepSeekAPIError as e:
        logger.warning(
            "[%s] <<< 非流式 DeepSeekAPIError | status=%d | %s",
            client_ip,
            e.status,
            e.message,
        )
        logger.warning(
            {
                "type": "ERR",
                "client_ip": client_ip,
                "model": model,
                "status": e.status,
                "error": e.message,
            }
        )
        return JSONResponse(
            status_code=e.status,
            content={"error": {"message": e.message, "status": e.status}},
        )
    except Exception as e:
        logger.error("[%s] <<< 非流式未知异常: %s %s", client_ip, type(e).__name__, e)
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
