"""DeepSeek HTTP 服务单元测试"""

import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

import deepseek_responses_api_server as server
from deepseek_responses_api_sdk import DeepSeekResponses


class _FakeDeepSeekResponses:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def create(self, **kwargs):
        if kwargs.get("stream"):
            return [
                {"type": "response.created", "sequence_number": 0},
                {
                    "type": "response.output_text.delta",
                    "delta": "你好",
                    "sequence_number": 1,
                },
                {"type": "response.completed", "sequence_number": 2},
            ]
        return {
            "id": "resp_test",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "你好"}],
                }
            ],
        }


class DeepSeekServerUnitTest(unittest.TestCase):
    def setUp(self):
        self._old_get_client = server._get_client
        self._old_env_api_key = server.os.environ.get("DEEPSEEK_API_KEY")
        server._get_client = lambda api_key: _FakeDeepSeekResponses(api_key)
        self.client = TestClient(server.app)

    def tearDown(self):
        server._get_client = self._old_get_client
        if self._old_env_api_key is None:
            server.os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            server.os.environ["DEEPSEEK_API_KEY"] = self._old_env_api_key

    def test_server_prefers_env_key_over_openai_compatible_bearer(self):
        captured = []

        def fake_get_client(api_key):
            captured.append(api_key)
            return _FakeDeepSeekResponses(api_key)

        server.os.environ["DEEPSEEK_API_KEY"] = "real-deepseek-token"
        server._get_client = fake_get_client

        response = self.client.post(
            "/v1/responses",
            json={"model": "deepseek-chat", "input": "你好"},
            headers={"Authorization": "Bearer dummy-openai-token"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured, ["real-deepseek-token"])

    def test_codex_accept_event_stream_triggers_sse_without_stream_field(self):
        body = {
            "model": "deepseek-reasoner",
            "instructions": "You are Codex.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "你好"}],
                }
            ],
        }

        with self.client.stream(
            "POST",
            "/v1/responses",
            json=body,
            headers={
                "Authorization": "Bearer test-token",
                "Accept": "text/event-stream",
            },
        ) as response:
            raw = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: response.output_text.delta", raw)
        self.assertIn("data: [DONE]", raw)


class DeepSeekStreamSchemaTest(unittest.TestCase):
    def test_parse_input_skips_developer_messages_and_keeps_dialog_roles(self):
        client = DeepSeekResponses(api_key="test-token")

        prompt = client._parse_input(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "大量工具说明"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "今天北京天气怎么样"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "北京今天多云。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "深圳呢"}],
                },
            ]
        )

        self.assertNotIn("大量工具说明", prompt)
        self.assertIn("用户：今天北京天气怎么样", prompt)
        self.assertIn("助手：北京今天多云。", prompt)
        self.assertTrue(prompt.endswith("当前用户：深圳呢"))

    def test_parse_input_includes_responses_function_call_output(self):
        client = DeepSeekResponses(api_key="test-token")

        prompt = client._parse_input(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "深圳呢"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"curl wttr.in/Shenzhen"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "深圳：多云，气温 25-29 摄氏度。",
                },
            ]
        )

        self.assertIn("当前用户：深圳呢", prompt)
        self.assertIn("工具结果：深圳：多云，气温 25-29 摄氏度。", prompt)
        self.assertNotIn("不要再次输出工具调用 JSON", prompt)

    def test_create_keeps_tool_prompt_after_function_call_output(self):
        client = DeepSeekResponses(api_key="test-token")
        client._ensure_session = lambda: None
        client._create_session = lambda: "session_test"
        client._solve_pow = lambda: None
        captured = []
        client._do_request = lambda payload: captured.append(payload) or [
            {"p": "response/content", "v": "深圳今天多云。"}
        ]

        client.create(
            model="deepseek-reasoner",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "深圳呢"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "深圳：多云。",
                },
            ],
            tools=[{"type": "function", "name": "exec_command"}],
        )

        self.assertEqual(len(captured), 1)
        self.assertIn("需要调用工具时", captured[0]["prompt"])
        self.assertIn("工具结果：深圳：多云。", captured[0]["prompt"])

    def test_merge_events_excludes_reasoning_text_from_output_text(self):
        client = DeepSeekResponses(api_key="test-token")

        text = client._merge_events(
            [
                {"p": "response/thinking_content", "v": "我先分析用户意图。"},
                {"p": "response/content", "v": "你好"},
            ]
        )

        self.assertEqual(text, "你好")

    def test_stream_events_include_codex_required_item_and_part_payloads(self):
        client = DeepSeekResponses(api_key="test-token")
        client._model_name = "deepseek-chat"
        client._has_tools = False
        client._prepare = lambda: ("你好", "deepseek-chat", {})
        client._do_request = lambda payload: [{"p": "response/content", "v": "你好"}]

        events = list(client._stream())
        events_by_type = {event["type"]: event for event in events}

        self.assertIn("item", events_by_type["response.output_item.added"])
        self.assertEqual(
            events_by_type["response.output_item.added"]["item"]["type"], "message"
        )
        self.assertIn("part", events_by_type["response.content_part.added"])
        self.assertEqual(
            events_by_type["response.content_part.added"]["part"]["type"],
            "output_text",
        )
        self.assertIn("part", events_by_type["response.content_part.done"])
        self.assertIn("item", events_by_type["response.output_item.done"])

    def test_tool_call_response_does_not_echo_raw_tool_json_as_message(self):
        client = DeepSeekResponses(api_key="test-token")
        text = '{"tool":"exec_command","args":{"cmd":"date"}}'
        tool_call = client._detect_tool_call(text)

        response = client._build_response(
            "deepseek-reasoner", text, tool_call, has_tools=True
        )

        self.assertEqual(response["output"][0]["type"], "function_call")
        self.assertEqual(len(response["output"]), 1)

    def test_detect_tool_call_inside_tool_call_tags(self):
        client = DeepSeekResponses(api_key="test-token")
        text = """我先查看目录。

<tool_call>
{"tool":"exec_command","args":{"cmd":"ls -la","workdir":"/tmp"}}
</tool_call>
当前用户：分析项目
"""

        tool_call = client._detect_tool_call(text)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "exec_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]), {"cmd": "ls -la", "workdir": "/tmp"}
        )

    def test_detect_tool_call_in_chinese_tool_prefix_format(self):
        client = DeepSeekResponses(api_key="test-token")
        text = """继续查看源码结构。
工具：exec_command({"cmd":"find apps libs -type f -name \"*.ts\" | head -100","workdir":"/tmp"})
"""

        tool_call = client._detect_tool_call(text)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "exec_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {
                "cmd": 'find apps libs -type f -name "*.ts" | head -100',
                "workdir": "/tmp",
            },
        )

    def test_detect_embedded_json_tool_call_after_text(self):
        client = DeepSeekResponses(api_key="test-token")
        text = """我将分析项目。先查看文件结构。

{"tool":"exec_command","args":{"cmd":"find . -type f -name \"*.js\" | head -50","workdir":"/tmp"}}
"""

        tool_call = client._detect_tool_call(text)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "exec_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {"cmd": 'find . -type f -name "*.js" | head -50', "workdir": "/tmp"},
        )

    def test_detect_markdown_calling_tool_block(self):
        client = DeepSeekResponses(api_key="test-token")
        text = """我来继续深入分析项目的核心代码和架构。
**Calling:** `exec_command`
```
{"command": "ls -la apps/ai-chat/src/db", "max_output_tokens": 500, "workdir": "/tmp"}
```
"""

        tool_call = client._detect_tool_call(text)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "exec_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {
                "cmd": "ls -la apps/ai-chat/src/db",
                "max_output_tokens": 500,
                "workdir": "/tmp",
            },
        )

    def test_detect_raw_exec_command_arguments_when_tool_is_available(self):
        client = DeepSeekResponses(api_key="test-token")
        client._tools = [{"type": "function", "name": "exec_command"}]
        text = '{"command": "ls -la apps/ai-chat/src/controller", "max_output_tokens": 500, "workdir": "/tmp"}'

        tool_call = client._detect_tool_call(text)

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "exec_command")
        self.assertEqual(
            json.loads(tool_call["arguments"]),
            {
                "cmd": "ls -la apps/ai-chat/src/controller",
                "max_output_tokens": 500,
                "workdir": "/tmp",
            },
        )

    def test_stream_tool_call_emits_function_call_item_without_empty_message(self):
        client = DeepSeekResponses(api_key="test-token")
        client._model_name = "deepseek-reasoner"
        client._has_tools = True
        client._prepare = lambda: ("深圳呢", "deepseek-reasoner", {})
        client._do_request = lambda payload: [
            {
                "p": "response/content",
                "v": '{"tool":"exec_command","args":{"cmd":"date"}}',
            }
        ]

        events = list(client._stream())
        output_events = [
            event
            for event in events
            if event["type"].startswith("response.output_item")
        ]

        self.assertTrue(output_events)
        self.assertTrue(
            all(event["item"]["type"] == "function_call" for event in output_events)
        )
        self.assertFalse(
            any(event["type"] == "response.output_text.delta" for event in events)
        )

    def test_create_logs_complete_input_without_truncation(self):
        client = DeepSeekResponses(api_key="test-token")
        long_text = "长文本" * 50
        client._ensure_session = lambda: None
        client._non_stream = lambda: {"output": []}

        with patch("deepseek_responses_api_sdk.logger.info") as info:
            client.create(model="deepseek-chat", input=long_text)

        req_log = next(
            call.args[0]
            for call in info.call_args_list
            if call.args[0].get("type") == "REQ"
        )
        self.assertEqual(req_log["input"], long_text)


if __name__ == "__main__":
    unittest.main()
