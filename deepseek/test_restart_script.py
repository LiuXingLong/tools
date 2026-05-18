"""DeepSeek 服务重启脚本测试"""

import unittest
from pathlib import Path


class RestartScriptTest(unittest.TestCase):
    def test_restart_script_contains_required_commands(self):
        script = Path(__file__).with_name("restart_server.sh")
        text = script.read_text(encoding="utf-8")

        self.assertIn("deepseek_responses_api_server.py", text)
        self.assertIn("lsof -ti tcp:", text)
        self.assertIn("nohup", text)
        self.assertIn("/health", text)
        self.assertIn("deepseek_server.out", text)


if __name__ == "__main__":
    unittest.main()
