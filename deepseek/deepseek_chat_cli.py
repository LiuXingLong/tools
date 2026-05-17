import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from deepseek_responses_api_sdk import DeepSeekResponses

load_dotenv(dotenv_path=_HERE / ".env")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Chat CLI")
    parser.add_argument(
        "prompt", nargs="?", default="请介绍一下深寻（DeepSeek）是什么？"
    )
    parser.add_argument("--no-stream", action="store_true", help="非流式输出")
    parser.add_argument(
        "--thinking", action="store_true", help="启用深度思考（R1 模式）"
    )
    parser.add_argument("--search", action="store_true", help="启用联网搜索")
    parser.add_argument("--tool", action="store_true", help="启用工具调用")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("错误: 请在 .env 中设置 DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = DeepSeekResponses(api_key=api_key)
    model = "deepseek-reasoner" if args.thinking else "deepseek-chat"

    tools = []
    if args.tool:
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

    result = client.create(
        model=model,
        input=args.prompt,
        tools=tools,
        stream=not args.no_stream,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
