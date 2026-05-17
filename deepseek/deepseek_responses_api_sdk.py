"""
OpenAI /v1/responses 兼容的 DeepSeek Web API 封装

用法：
    from deepseek_responses_api_sdk import DeepSeekResponses

    client = DeepSeekResponses(api_key="your_token")
    resp = client.create(
        model="deepseek-chat", // deepseek-reasoner
        input="北京天气怎么样",
        tools=[{"type": "function", "name": "get_weather", "description": "获取天气", "parameters": {...}}],
    )
    print(resp["output"])
"""

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

import opendeep as genai
from opendeep import pow as deepseek_pow
from opendeep.config import config as ds_config

_HERE = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=_HERE / ".env")

_wasm_env = os.environ.get("DEEPSEEK_WASM_PATH")
deepseek_pow.WASM_PATH = str(_HERE / (_wasm_env or "sha3_wasm_bg.wasm"))

logger = logging.getLogger("deepseek.sdk")

TOOL_SYSTEM_PROMPT = """
你有以下工具可用，需要获取实时信息时请严格按 JSON 格式返回工具调用：

{tools_desc}

需要调用工具时，只返回以下格式（不要包含其他文字）：
{{"tool": "函数名", "args": {{"参数名": "参数值"}}}}

不需要工具时正常回答即可。
"""


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


class DeepSeekAPIError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


class DeepSeekResponses:
    """OpenAI /v1/responses 兼容的 DeepSeek API 封装"""

    def __init__(self, api_key: str = ""):
        self._model = None
        self._session = None
        self._headers = None
        self.responses = self  # client.responses.create(...) 兼容写法
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            genai.configure(api_key=key)

    def _ensure_session(self):
        self._model = genai.GenerativeModel("deepseek-chat")
        self._session = self._model.session
        self._headers = self._model._get_headers()

    def _solve_pow(self):
        try:
            pow_solver = deepseek_pow.DeepSeekPOW()
            pow_resp = self._session.post(
                f"{ds_config.base_url}/chat/create_pow_challenge",
                headers=self._headers,
                json={"target_path": "/api/v0/chat/completion"},
            )
            logger.debug("PoW status=%s", pow_resp.status_code)
            if pow_resp.ok:
                cd = pow_resp.json()
                if cd and isinstance(cd, dict):
                    cd = cd.get("data", {}).get("biz_data", {}).get("challenge")
                    if cd:
                        self._headers["x-ds-pow-response"] = pow_solver.solve_challenge(
                            cd
                        )
        except Exception as e:
            logger.error("PoW 异常: %s", e)
            raise DeepSeekAPIError(401, f"PoW 认证失败（token 可能无效）: {e}")

    def create(self, **kwargs) -> dict | list[dict]:
        """
        OpenAI Responses API 兼容入口
        """
        self._ensure_session()
        self._kwargs = kwargs
        self._model_name = kwargs.get("model", "deepseek-chat")
        self._tools = kwargs.get("tools", [])
        self._tool_choice = kwargs.get("tool_choice", "auto")
        self._has_tools = bool(self._tools) and self._tool_choice != "none"

        model_info = f"model={self._model_name} stream={kwargs.get('stream', False)}"
        logger.info(
            {
                "type": "REQ",
                "model": self._model_name,
                "input": str(kwargs.get("input", ""))[:80],
            }
        )

        if kwargs.get("stream", False):
            return list(self._stream())
        try:
            result = self._non_stream()
            logger.info({"type": "RESP", "model": self._model_name, "resp": result})
            return result
        except DeepSeekAPIError as e:
            logger.warning(
                {
                    "type": "ERR",
                    "model": self._model_name,
                    "status": e.status,
                    "error": e.message,
                }
            )
            return {"error": {"message": e.message, "status": e.status}}

    def _create_session(self) -> str:
        url = f"{ds_config.base_url}/chat_session/create"
        payload = {"character_id": None}
        resp = self._session.post(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()
        logger.debug(f"_create_session raw response: {raw}")
        if not raw or not isinstance(raw, dict):
            raise DeepSeekAPIError(
                500, f"创建 session 失败：无效响应: {resp.text[:200]}"
            )
        inner = raw.get("data")
        if not inner or not isinstance(inner, dict):
            raise DeepSeekAPIError(500, f"创建 session 失败：data 字段无效: {raw}")
        biz = inner.get("biz_data")
        if not biz or not isinstance(biz, dict):
            raise DeepSeekAPIError(500, f"创建 session 失败：biz_data 字段无效: {raw}")
        session_id = biz.get("id")
        if not session_id:
            raise DeepSeekAPIError(500, f"创建 session 失败：无 id 字段: {raw}")
        return session_id

    def _prepare(self) -> tuple[str, str, dict]:
        user_input = self._parse_input(self._kwargs.get("input", ""))
        thinking_enabled = self._model_name == "deepseek-reasoner"

        prompt = user_input
        if self._has_tools:
            tools_desc = _build_tool_desc(self._tools)
            prompt = TOOL_SYSTEM_PROMPT.format(tools_desc=tools_desc) + "\n\n" + prompt

        chat_session_id = self._create_session()
        self._solve_pow()

        payload = {
            "chat_session_id": chat_session_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": True,
            "model_type": "default",
        }
        logger.info({"type": "DSREQ", "target": "chat/completion", "payload": payload})
        return prompt, self._model_name, payload

    def _do_request(self, payload: dict) -> list[dict]:
        resp = None
        try:
            resp = self._session.post(
                f"{ds_config.base_url}/chat/completion",
                headers=self._headers,
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as e:
            status = getattr(resp, "status_code", 500) if resp is not None else 500
            detail = str(e)
            if resp is not None:
                try:
                    detail = resp.json()
                except Exception:
                    detail = getattr(resp, "text", str(e))[:500]
            logger.warning(
                {
                    "type": "DSRES",
                    "target": "chat/completion",
                    "status": status,
                    "error": detail,
                }
            )
            raise DeepSeekAPIError(status, f"DeepSeek API 调用失败: {detail}") from e

        events = []
        logger.debug("开始读取 SSE 事件流")
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    raw = decoded[6:].strip()
                    if raw:
                        events.append(json.loads(raw))
        logger.info(
            {"type": "DSRES", "target": "chat/completion", "events_count": len(events)}
        )
        return events

    def _non_stream(self) -> dict:
        prompt, model_name, payload = self._prepare()
        events = self._do_request(payload)
        full_text = self._merge_events(events)
        full_text = re.sub(r"\s*\[citation:\d+\]", "", full_text)
        tool_call = self._detect_tool_call(full_text)
        return self._build_response(model_name, full_text, tool_call, self._has_tools)

    def _stream(self):
        try:
            prompt, model_name, payload = self._prepare()
            events = self._do_request(payload)

            full_text = self._merge_events(events)
            full_text = re.sub(r"\s*\[citation:\d+\]", "", full_text)
            tool_call = self._detect_tool_call(full_text)
            final = self._build_response(
                model_name, full_text, tool_call, self._has_tools
            )
            msg_id = (
                final.get("output", [{}])[0].get("id", "msg_unknown")
                if final.get("output")
                else "msg_unknown"
            )
            seq = 0

            yield {
                "type": "response.created",
                "response": final,
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.in_progress",
                "response": final,
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item_id": msg_id,
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": msg_id,
                "output_index": 0,
                "part_id": "part_0",
                "sequence_number": seq,
            }
            seq += 1

            for ev in events:
                item = self._emit_delta(ev)
                if item:
                    item["item_id"] = msg_id
                    item["sequence_number"] = seq
                    yield item
                    seq += 1

            yield {
                "type": "response.output_text.done",
                "text": full_text,
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": msg_id,
                "output_index": 0,
                "part_id": "part_0",
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.output_item.done",
                "output_index": 0,
                "item_id": msg_id,
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.completed",
                "response": final,
                "sequence_number": seq,
            }

            logger.info({"type": "RESP", "model": model_name, "resp": final})
        except DeepSeekAPIError as e:
            logger.warning(
                {"type": "ERR", "model": self._model_name, "error": e.message}
            )
            yield {"type": "error", "error": {"message": e.message, "status": e.status}}
        except Exception as e:
            logger.error(f"_stream 未知异常: {type(e).__name__}: {e}")
            yield {
                "type": "error",
                "error": {"message": f"{type(e).__name__}: {e}", "status": 500},
            }

    def _emit_delta(self, ev: dict) -> dict | None:
        if "v" not in ev:
            return None
        p = ev.get("p", "")
        v = ev["v"]
        if not isinstance(v, str):
            return None
        target = "content"
        if p:
            target = p.removeprefix("response/")
        if target not in ("content", "thinking_content"):
            return None
        return {
            "type": "response.output_text.delta",
            "delta": v,
            "item_id": "",
            "output_index": 0,
            "content_index": 0,
        }

    def _parse_input(self, inp) -> str:
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

    def _merge_events(self, events: list[dict]) -> str:
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

    def _detect_tool_call(self, text: str) -> dict | None:
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
        self, model_name: str, full_text: str, tool_call: dict | None, has_tools: bool
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
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "user": None,
        }


# CLI 测试入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="你好")
    parser.add_argument(
        "--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""), help="userToken"
    )
    parser.add_argument("--tools", action="store_true", help="启用默认工具")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    args = parser.parse_args()

    client = DeepSeekResponses(api_key=args.api_key)

    tools = []
    if args.tools:
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "获取指定城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名"}},
                    "required": ["city"],
                },
            }
        ]

    if args.stream:
        for event in client.create(
            model="deepseek-chat", input=args.prompt, tools=tools, stream=True
        ):
            if isinstance(event, dict):
                print(json.dumps(event, ensure_ascii=False))
    else:
        result = client.create(model="deepseek-chat", input=args.prompt, tools=tools)
        print(json.dumps(result, ensure_ascii=False, indent=2))
