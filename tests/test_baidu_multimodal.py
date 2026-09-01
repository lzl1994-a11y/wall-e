import tempfile
import unittest
from pathlib import Path

import yaml

from services.llm_request_options import reasoning_request_options
from services.multimodal import create_multimodal
from services.multimodal.baidu_multimodal import BaiduMultimodal


class BaiduMultimodalTests(unittest.TestCase):
    def test_builds_qianfan_chat_audio_content(self):
        message = BaiduMultimodal().build_audio_message("aGVsbG8=")

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertIn("不得抄写本条提示文字", message["content"][0]["text"])
        self.assertEqual(message["content"][1], {
            "type": "input_audio",
            "input_audio": {"data": "aGVsbG8=", "format": "wav"},
        })

    def test_factory_selects_baidu_adapter(self):
        with tempfile.TemporaryDirectory(prefix="wali-baidu-") as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"llm": {"provider": "baidu"}}),
                encoding="utf-8",
            )

            adapter = create_multimodal(str(config_path))

        self.assertIsInstance(adapter, BaiduMultimodal)

    def test_fast_ernie_thinking_model_uses_qianfan_switch(self):
        options = reasoning_request_options({
            "provider": "baidu",
            "model": "ernie-5.0-thinking-preview",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {"extra_body": {"enable_thinking": False}})

    def test_fast_qianfan_deepseek_uses_thinking_object(self):
        options = reasoning_request_options({
            "provider": "qianfan",
            "model": "deepseek-v3.2",
            "reasoning_effort": "fast",
        })

        self.assertEqual(
            options,
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_unknown_qianfan_model_keeps_provider_default(self):
        options = reasoning_request_options({
            "provider": "baidu",
            "model": "custom-endpoint-id",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {})
