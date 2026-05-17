import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv

import opendeep as genai
from opendeep import pow as deepseek_pow
from opendeep.config import config

load_dotenv()

# 浏览器获取文件, 不设置默认使用 opendeep 包内置的 pow.wasm
deepseek_pow.WASM_PATH = os.environ.get("DEEPSEEK_WASM_PATH", "sha3_wasm_bg.wasm")
# 浏览器获取 Storage下的userToken
genai.configure(api_key=os.environ["DEEPSEEK_API_KEY"])

# 选择模型 'deepseek-chat' (V3) and 'deepseek-reasoner' (R1)
model = genai.GenerativeModel("deepseek-chat")


TOOL_PROMPT = """
你有以下工具可用，需要获取实时信息时请严格按 JSON 格式返回工具调用，不要包含其他文字：

可用工具：
- get_weather(city: str): 获取指定城市的实时天气
- get_current_time(timezone: str): 获取指定时区的当前时间

需要调用工具时，只返回以下格式：
{"tool": "工具名", "args": {"参数名": "参数值"}}

不需要工具时正常回答即可。
"""


def generate_content_raw(
    prompt: str,
    stream: bool = True,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    tool_enabled: bool = False,
) -> list[dict]:
    """获取原始 API 响应数据

    API 始终返回 SSE 格式，stream 参数控制客户端合并方式：
      True = 返回逐条原始事件列表
      False = 将所有事件合并为一条完整响应（拼接 content）
    """
    session = model.session
    headers = model._get_headers()
    chat_session_id = model._create_session()

    # 获取 PoW challenge
    pow_solver = deepseek_pow.DeepSeekPOW()
    pow_resp = session.post(
        f"{config.base_url}/chat/create_pow_challenge",
        headers=headers,
        json={"target_path": "/api/v0/chat/completion"},
    )
    if pow_resp.ok:
        challenge_data = (
            pow_resp.json().get("data", {}).get("biz_data", {}).get("challenge")
        )
        if challenge_data:
            headers["x-ds-pow-response"] = pow_solver.solve_challenge(challenge_data)

    # 发送请求（API 固定用 SSE 格式返回，不传 stream 参数）
    if tool_enabled:
        prompt = TOOL_PROMPT + "\n\n" + prompt
    payload = {
        "chat_session_id": chat_session_id,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
        "model_type": "default",
    }
    response = session.post(
        f"{config.base_url}/chat/completion", headers=headers, json=payload, stream=True
    )
    response.raise_for_status()

    events = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                raw = decoded[6:].strip()
                if raw:
                    events.append(json.loads(raw))

    if stream:
        return events

    # 非流式：重建完整响应状态
    state: dict = {}
    current_target = "content"
    for ev in events:
        if "v" not in ev:
            continue
        p = ev.get("p", "")
        v = ev["v"]

        if p:
            current_target = p.removeprefix("response/")

        if isinstance(v, str):
            if current_target in ("content", "thinking_content"):
                state.setdefault(current_target, "")
                state[current_target] += v
        elif p:
            state[current_target] = v

    # 去除 content/thinking_content 中的 citation 标记
    for k in ("content", "thinking_content"):
        if k in state and isinstance(state[k], str):
            state[k] = re.sub(r"\s*\[citation:\d+\]", "", state[k])

    # 检测工具调用：content 为 JSON 时解析并标记
    if tool_enabled and "content" in state:
        try:
            tc = json.loads(state["content"])
            if isinstance(tc, dict) and "tool" in tc:
                state["tool_call"] = tc["tool"]
                state["tool_args"] = tc.get("args", {})
        except (json.JSONDecodeError, TypeError):
            pass
    return [state]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Chat CLI")
    parser.add_argument(
        "prompt", nargs="?", default="请介绍一下深寻（DeepSeek）是什么？"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="非流式：合并为一条完整 JSON 响应"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="启用深度思考（R1 模式）"
    )
    parser.add_argument("--search", action="store_true", help="启用联网搜索")
    parser.add_argument(
        "--tool", action="store_true", help="启用工具调用（prompt 注入）"
    )
    args = parser.parse_args()

    raw_data = generate_content_raw(
        args.prompt,
        stream=not args.no_stream,
        thinking_enabled=args.thinking,
        search_enabled=args.search,
        tool_enabled=args.tool,
    )
    print(json.dumps(raw_data, ensure_ascii=False, indent=2, default=str))
