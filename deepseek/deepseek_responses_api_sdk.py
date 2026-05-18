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
                "input": kwargs.get("input", ""),
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
            msg_item = next(
                (
                    item
                    for item in final.get("output", [])
                    if item.get("type") == "message"
                ),
                {
                    "id": "msg_unknown",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            )
            msg_id = msg_item.get("id", "msg_unknown")
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

            if tool_call:
                call_item = final["output"][0]
                yield {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item_id": call_item["id"],
                    "item": {
                        **call_item,
                        "arguments": "",
                    },
                    "sequence_number": seq,
                }
                seq += 1

                yield {
                    "type": "response.function_call_arguments.delta",
                    "item_id": call_item["id"],
                    "output_index": 0,
                    "delta": call_item["arguments"],
                    "sequence_number": seq,
                }
                seq += 1

                yield {
                    "type": "response.function_call_arguments.done",
                    "item_id": call_item["id"],
                    "output_index": 0,
                    "arguments": call_item["arguments"],
                    "sequence_number": seq,
                }
                seq += 1

                yield {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item_id": call_item["id"],
                    "item": call_item,
                    "sequence_number": seq,
                }
                seq += 1

                yield {
                    "type": "response.completed",
                    "response": final,
                    "sequence_number": seq,
                }
                logger.info({"type": "RESP", "model": model_name, "resp": final})
                return

            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item_id": msg_id,
                "item": {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "content": [],
                },
                "sequence_number": seq,
            }
            seq += 1

            content_part = {
                "type": "output_text",
                "text": "",
                "annotations": [],
            }
            yield {
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": msg_id,
                "output_index": 0,
                "part_id": "part_0",
                "part": content_part,
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
                "part": {
                    "type": "output_text",
                    "text": full_text,
                    "annotations": [],
                },
                "sequence_number": seq,
            }
            seq += 1

            yield {
                "type": "response.output_item.done",
                "output_index": 0,
                "item_id": msg_id,
                "item": msg_item,
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
        if target != "content":
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
            messages = []
            fallback_texts = []
            tool_outputs = []
            for item in inp:
                if isinstance(item, dict):
                    role = item.get("role")
                    if item.get("type") == "message":
                        text = self._extract_content_text(item.get("content"))
                        if role in ("user", "assistant") and text:
                            messages.append({"role": role, "text": text})
                        elif not role and text:
                            fallback_texts.append(text)
                    elif "content" in item:
                        text = self._extract_content_text(item.get("content"))
                        if role in ("user", "assistant") and text:
                            messages.append({"role": role, "text": text})
                        elif text:
                            fallback_texts.append(text)
                    elif item.get("type") == "function_call_output":
                        output = item.get("output", "")
                        if isinstance(output, str) and output.strip():
                            tool_outputs.append(output.strip())
                elif isinstance(item, str):
                    fallback_texts.append(item)
            if messages:
                recent_messages = messages[-12:]
                if (
                    tool_outputs
                    and len(recent_messages) == 1
                    and recent_messages[0]["role"] == "user"
                ):
                    prompt = f"当前用户：{recent_messages[0]['text']}"
                else:
                    prompt = self._format_messages(recent_messages)
                if tool_outputs:
                    prompt += "\n" + "\n".join(
                        f"工具结果：{output}" for output in tool_outputs[-4:]
                    )
                return prompt
            return "\n".join(fallback_texts)
        return str(inp)

    def _extract_content_text(self, content) -> str:
        texts = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    texts.append(c["text"])
                elif isinstance(c, str):
                    texts.append(c)
        elif isinstance(content, str):
            texts.append(content)
        return "\n".join(text.strip() for text in texts if text.strip())

    def _format_messages(self, messages: list[dict]) -> str:
        if len(messages) == 1 and messages[0]["role"] == "user":
            return messages[0]["text"]

        lines = []
        for msg in messages[:-1]:
            label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{label}：{msg['text']}")

        last = messages[-1]
        if last["role"] == "user":
            lines.append(f"当前用户：{last['text']}")
        else:
            lines.append(f"助手：{last['text']}")
        return "\n".join(lines)

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
            if isinstance(v, str) and target == "content":
                text += v
        return text

    def _detect_tool_call(self, text: str) -> dict | None:
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        if match:
            text = match.group(1)
        else:
            markdown_call = self._extract_markdown_calling_tool_call(text)
            if markdown_call:
                return markdown_call
            prefixed_call = self._extract_prefixed_tool_call(text)
            if prefixed_call:
                return prefixed_call
            embedded_call = self._extract_embedded_json_tool_call(text)
            if embedded_call:
                try:
                    obj = json.loads(embedded_call)
                except json.JSONDecodeError:
                    obj = self._parse_loose_json_object(embedded_call)
                if isinstance(obj, dict) and "tool" in obj:
                    return {
                        "name": obj["tool"],
                        "arguments": json.dumps(
                            obj.get("args", {}), ensure_ascii=False
                        ),
                    }
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool" in obj:
                return {
                    "name": obj["tool"],
                    "arguments": json.dumps(obj.get("args", {}), ensure_ascii=False),
                }
            if isinstance(obj, dict):
                raw_exec_call = self._build_raw_exec_command_call(obj)
                if raw_exec_call:
                    return raw_exec_call
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _build_raw_exec_command_call(self, arguments: dict) -> dict | None:
        tool_names = self._available_tool_names()
        if "exec_command" not in tool_names:
            return None
        if "command" not in arguments and "cmd" not in arguments:
            return None

        normalized = dict(arguments)
        if "command" in normalized and "cmd" not in normalized:
            normalized["cmd"] = normalized.pop("command")
        return {
            "name": "exec_command",
            "arguments": json.dumps(normalized, ensure_ascii=False),
        }

    def _available_tool_names(self) -> set[str]:
        names = set()
        for tool in getattr(self, "_tools", []) or []:
            func = tool.get("function", tool) if isinstance(tool, dict) else {}
            name = func.get("name")
            if name:
                names.add(name)
        return names

    def _extract_markdown_calling_tool_call(self, text: str) -> dict | None:
        match = re.search(
            r"\*\*Calling:\*\*\s*`?([A-Za-z_]\w*)`?\s*```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.S | re.I,
        )
        if not match:
            return None

        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            try:
                arguments = self._parse_loose_json_object(match.group(2))
            except json.JSONDecodeError:
                return None

        if not isinstance(arguments, dict):
            return None
        if (
            match.group(1) == "exec_command"
            and "command" in arguments
            and "cmd" not in arguments
        ):
            arguments["cmd"] = arguments.pop("command")
        return {
            "name": match.group(1),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _extract_embedded_json_tool_call(self, text: str) -> str | None:
        start = text.find('{"tool"')
        if start == -1:
            start = text.find('{"tool"'.replace('"', "'"))
        if start == -1:
            return None
        try:
            end = self._find_balanced_json_end(text, start)
            return text[start:end]
        except json.JSONDecodeError:
            return None

    def _extract_prefixed_tool_call(self, text: str) -> dict | None:
        match = re.search(r"(?:工具|tool)\s*[:：]\s*([A-Za-z_]\w*)\s*\(", text)
        if not match:
            return None

        start = match.end()
        depth = 1
        in_string = False
        escape = False
        end = start
        for index, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break

        if depth != 0:
            return None

        try:
            arguments = text[start:end].strip()
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = self._parse_loose_json_object(arguments)
            return {
                "name": match.group(1),
                "arguments": json.dumps(parsed_args, ensure_ascii=False),
            }
        except (json.JSONDecodeError, TypeError):
            return None

    def _parse_loose_json_object(self, text: str) -> dict:
        text = text.strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise json.JSONDecodeError("not object", text, 0)

        result = {}
        index = 1
        while index < len(text) - 1:
            index = self._skip_json_space_and_comma(text, index)
            if index >= len(text) - 1:
                break
            if text[index] != '"':
                raise json.JSONDecodeError("key expected", text, index)
            key_end = text.find('"', index + 1)
            if key_end == -1:
                raise json.JSONDecodeError("unterminated key", text, index)
            key = text[index + 1 : key_end]
            index = self._skip_json_space(text, key_end + 1)
            if index >= len(text) or text[index] != ":":
                raise json.JSONDecodeError("colon expected", text, index)
            index = self._skip_json_space(text, index + 1)
            value, index = self._parse_loose_json_value(text, index)
            result[key] = value
        return result

    def _parse_loose_json_value(self, text: str, index: int):
        if text[index] == '"':
            end = index + 1
            while end < len(text):
                if text[end] == '"':
                    next_index = self._skip_json_space(text, end + 1)
                    if next_index >= len(text) or text[next_index] in ",}":
                        return text[index + 1 : end], next_index
                end += 1
            raise json.JSONDecodeError("unterminated string", text, index)

        if text[index] == "{":
            end = self._find_balanced_json_end(text, index)
            return self._parse_loose_json_object(
                text[index:end]
            ), self._skip_json_space(text, end)

        if text[index] == "[":
            end = self._find_balanced_json_end(text, index)
            return json.loads(text[index:end]), self._skip_json_space(text, end)

        end = index
        while end < len(text) and text[end] not in ",}":
            end += 1
        raw = text[index:end].strip()
        if raw in ("true", "false", "null"):
            return json.loads(raw), self._skip_json_space(text, end)
        try:
            return json.loads(raw), self._skip_json_space(text, end)
        except json.JSONDecodeError:
            return raw, self._skip_json_space(text, end)

    def _find_balanced_json_end(self, text: str, index: int) -> int:
        pairs = {"{": "}", "[": "]"}
        stack = [pairs[text[index]]]
        in_string = False
        escape = False
        cursor = index + 1
        while cursor < len(text):
            char = text[cursor]
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char in pairs:
                stack.append(pairs[char])
            elif not in_string and stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return cursor + 1
            cursor += 1
        raise json.JSONDecodeError("unbalanced value", text, index)

    def _skip_json_space(self, text: str, index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def _skip_json_space_and_comma(self, text: str, index: int) -> int:
        while index < len(text) and (text[index].isspace() or text[index] == ","):
            index += 1
        return index

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
        if full_text and not tool_call:
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
