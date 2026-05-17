"""
DeepSeek 工具测试脚本
验证 SDK / CLI / HTTP 服务的基本功能

用法：
    python3 deepseek/test_deepseek.py
    python3 deepseek/test_deepseek.py --server-only    # 只测 HTTP 服务
    python3 deepseek/test_deepseek.py --no-server       # 跳过 HTTP 服务（不需启动 server）
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_PROJECT = _HERE.parent
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_HERE / ".env")

from deepseek_responses_api_sdk import DeepSeekResponses

PASS = "✅ PASS"
FAIL = "❌ FAIL"
TIMEOUT = 60


def log(status: str, msg: str):
    print(f"  {status}  {msg}")


def test_sdk_non_stream():
    client = DeepSeekResponses()
    resp = client.create(model="deepseek-chat", input="用一句话介绍Python")
    if isinstance(resp, dict) and resp.get("output"):
        return True, str(resp["output"][0].get("content", ""))[:60]
    return False, str(resp)[:200]


def test_sdk_stream():
    client = DeepSeekResponses()
    events = list(client.create(model="deepseek-chat", input="你好", stream=True))
    types = [e.get("type") for e in events]
    if "response.created" in types and "response.completed" in types:
        return True, f"{len(events)} events, first={types[0]}, last={types[-1]}"
    return False, f"events: {types[:5]}"


def test_cli_no_stream():
    result = subprocess.run(
        [
            sys.executable,
            str(_HERE / "deepseek_chat_cli.py"),
            "--no-stream",
            "用一句话介绍Python",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")},
    )
    if result.returncode != 0:
        return False, f"exit code {result.returncode}, stderr: {result.stderr[:200]}"
    try:
        data = json.loads(result.stdout)
        if data.get("output"):
            return True, "output ok"
        return False, f"no output: {result.stdout[:200]}"
    except json.JSONDecodeError as e:
        return False, f"json parse fail: {e}, stdout: {result.stdout[:200]}"


def test_cli_stream():
    result = subprocess.run(
        [sys.executable, str(_HERE / "deepseek_chat_cli.py"), "你好"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env={**os.environ, "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")},
    )
    if result.returncode != 0:
        return False, f"exit code {result.returncode}, stderr: {result.stderr[:200]}"
    try:
        lines = [
            json.loads(line)
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        types = [e.get("type") for e in lines]
        if "response.created" in types and "response.completed" in types:
            return True, f"{len(lines)} events ok"
        return False, f"unexpected events: {types[:5]}"
    except json.JSONDecodeError:
        pass
    if "response.created" in result.stdout:
        return True, "raw text contains response.created"
    return False, f"stdout: {result.stdout[:200]}"


def test_server_health(base_url: str):
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        data = json.loads(resp.read())
        if data.get("status") == "ok":
            return True, "health check ok"
        return False, str(data)
    except Exception as e:
        return False, str(e)


def test_server_non_stream(base_url: str, token: str):
    import urllib.request

    body = json.dumps(
        {"model": "deepseek-chat", "input": "用一句话介绍Python"}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/responses",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        data = json.loads(resp.read())
        if data.get("output"):
            return True, "output ok"
        return False, f"no output: {json.dumps(data, ensure_ascii=False)[:200]}"
    except Exception as e:
        return False, str(e)


def test_server_stream(base_url: str, token: str):
    import urllib.request

    body = json.dumps(
        {"model": "deepseek-chat", "input": "你好", "stream": True}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/responses",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        raw = resp.read().decode()
        if "response.created" in raw and "response.completed" in raw:
            return True, "SSE stream ok"
        return False, f"unexpected stream: {raw[:200]}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 工具测试脚本")
    parser.add_argument("--server-only", action="store_true", help="只测 HTTP 服务")
    parser.add_argument("--no-server", action="store_true", help="跳过 HTTP 服务测试")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print(f"\n{FAIL} 请在 deepseek/.env 中设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    total = 0
    passed = 0

    def run(name: str, fn, *fn_args):
        nonlocal total, passed
        total += 1
        ok, msg = fn(*fn_args)
        status = PASS if ok else FAIL
        log(status, f"{name}: {msg}")
        if ok:
            passed += 1

    # ── SDK 测试 ──
    if not args.server_only:
        print("\n━━━ SDK ━━━")
        run("非流式", test_sdk_non_stream)
        run("流式", test_sdk_stream)

        print("\n━━━ CLI ━━━")
        run("非流式", test_cli_no_stream)
        run("流式", test_cli_stream)

    # ── HTTP 服务测试 ──
    if not args.no_server:
        print("\n━━━ HTTP 服务 ━━━")
        port = os.environ.get("PORT", "8888")
        base_url = f"http://localhost:{port}"

        server_proc = None
        if not _is_server_running(base_url):
            print("  启动服务器...")
            log_file = _HERE / "deepseek_test_server.log"
            server_proc = subprocess.Popen(
                [sys.executable, str(_HERE / "deepseek_responses_api_server.py")],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
            )
            _wait_for_server(base_url, retries=10, delay=1.5)
            print("  服务器已启动")

        run("health", test_server_health, base_url)
        run("非流式", test_server_non_stream, base_url, api_key)
        run("流式", test_server_stream, base_url, api_key)

        if server_proc:
            server_proc.terminate()
            server_proc.wait(timeout=5)
            print("  服务器已关闭")

    # ── 结果 ──
    print(f"\n{'=' * 40}")
    log(PASS if passed == total else FAIL, f"{passed}/{total} 通过")
    return 0 if passed == total else 1


def _is_server_running(base_url: str) -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=2)
        return True
    except Exception:
        return False


def _wait_for_server(base_url: str, retries: int, delay: float):
    import urllib.request

    for i in range(retries):
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"服务器 {base_url} 启动超时")


if __name__ == "__main__":
    sys.exit(main())
